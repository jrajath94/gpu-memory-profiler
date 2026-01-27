"""Memory leak detection via growth-rate analysis and pattern recognition."""

import logging
from typing import List, Optional, Tuple

from gpu_memory_profiler.models import (
    AllocationEvent,
    AllocationEventType,
    LeakCandidate,
    LeakSeverity,
    MemorySnapshot,
    MemoryTimeline,
    ProfileConfig,
)

logger = logging.getLogger(__name__)

# Minimum number of snapshots needed for trend analysis
MIN_SNAPSHOTS_FOR_TREND = 5

# Growth rate thresholds (bytes per second)
GROWTH_RATE_LOW = 100 * 1024  # 100 KB/s
GROWTH_RATE_MEDIUM = 1024 * 1024  # 1 MB/s
GROWTH_RATE_HIGH = 10 * 1024 * 1024  # 10 MB/s
GROWTH_RATE_CRITICAL = 100 * 1024 * 1024  # 100 MB/s


class LeakDetector:
    """Detects memory leaks from profiling timeline data.

    Uses linear regression on memory snapshots to detect monotonic
    growth, allocation/free imbalance analysis, and pattern recognition
    for common leak scenarios.

    Args:
        config: Profiling configuration with leak detection thresholds.
    """

    def __init__(self, config: Optional[ProfileConfig] = None) -> None:
        self._config = config or ProfileConfig()

    def analyze(self, timeline: MemoryTimeline) -> List[LeakCandidate]:
        """Run all leak detection heuristics on a memory timeline.

        Args:
            timeline: Memory timeline with snapshots and events.

        Returns:
            List of detected leak candidates, sorted by severity.
        """
        leaks: List[LeakCandidate] = []

        # Heuristic 1: Monotonic growth detection
        growth_leak = self._detect_monotonic_growth(timeline.snapshots)
        if growth_leak is not None:
            leaks.append(growth_leak)

        # Heuristic 2: Allocation/free imbalance
        imbalance_leak = self._detect_allocation_imbalance(timeline)
        if imbalance_leak is not None:
            leaks.append(imbalance_leak)

        # Heuristic 3: Large persistent allocations
        persistent_leaks = self._detect_persistent_allocations(
            timeline.events, timeline.duration_seconds
        )
        leaks.extend(persistent_leaks)

        # Sort by severity (critical first)
        severity_order = {
            LeakSeverity.CRITICAL: 0,
            LeakSeverity.HIGH: 1,
            LeakSeverity.MEDIUM: 2,
            LeakSeverity.LOW: 3,
        }
        leaks.sort(key=lambda l: severity_order[l.severity])

        return leaks

    def _detect_monotonic_growth(
        self,
        snapshots: List[MemorySnapshot],
    ) -> Optional[LeakCandidate]:
        """Detect steady memory growth using linear regression.

        A positive slope that is consistently increasing (R^2 > 0.7)
        indicates a likely leak.

        Args:
            snapshots: Ordered list of memory snapshots.

        Returns:
            LeakCandidate if growth detected, None otherwise.
        """
        if len(snapshots) < MIN_SNAPSHOTS_FOR_TREND:
            return None

        times = [s.timestamp for s in snapshots]
        values = [s.allocated_bytes for s in snapshots]

        slope, r_squared = _linear_regression(times, values)

        # Growth must exceed threshold and be a consistent trend
        if slope < self._config.leak_growth_threshold_bytes:
            return None
        if r_squared < 0.7:
            return None

        duration = times[-1] - times[0]
        total_growth = int(slope * duration)
        severity = _classify_growth_rate(slope)

        return LeakCandidate(
            description=(
                f"Monotonic memory growth detected: "
                f"{slope / (1024 * 1024):.2f} MB/s "
                f"(R²={r_squared:.2f})"
            ),
            severity=severity,
            growth_rate_bytes_per_sec=slope,
            total_leaked_bytes=total_growth,
            first_seen=times[0],
            recommendation=(
                "Memory is growing linearly over time. Check for: "
                "1) Tensors appended to lists without clearing, "
                "2) Gradients accumulating (missing optimizer.zero_grad()), "
                "3) Tensors moved to CPU without detaching from graph."
            ),
        )

    def _detect_allocation_imbalance(
        self,
        timeline: MemoryTimeline,
    ) -> Optional[LeakCandidate]:
        """Detect when allocations significantly exceed frees.

        Args:
            timeline: Memory timeline with events.

        Returns:
            LeakCandidate if imbalance detected, None otherwise.
        """
        alloc_count = timeline.total_allocations
        free_count = timeline.total_frees

        if alloc_count == 0:
            return None

        imbalance_ratio = 1.0 - (free_count / alloc_count)

        # Significant imbalance: more than 20% of allocations not freed
        if imbalance_ratio < 0.2:
            return None

        # Calculate total leaked bytes from events
        alloc_bytes = sum(
            e.size_bytes
            for e in timeline.events
            if e.event_type == AllocationEventType.ALLOCATE
        )
        free_bytes = sum(
            e.size_bytes
            for e in timeline.events
            if e.event_type == AllocationEventType.FREE
        )
        leaked_bytes = max(0, alloc_bytes - free_bytes)

        duration = timeline.duration_seconds
        growth_rate = leaked_bytes / duration if duration > 0 else 0.0
        severity = _classify_growth_rate(growth_rate)

        return LeakCandidate(
            description=(
                f"Allocation/free imbalance: {alloc_count} allocations, "
                f"{free_count} frees ({imbalance_ratio:.0%} unfreed)"
            ),
            severity=severity,
            growth_rate_bytes_per_sec=growth_rate,
            total_leaked_bytes=leaked_bytes,
            first_seen=timeline.start_time,
            recommendation=(
                "More tensors are being allocated than freed. Check for: "
                "1) Tensors stored in global/class variables, "
                "2) Circular references preventing garbage collection, "
                "3) torch.no_grad() missing in inference paths."
            ),
        )

    def _detect_persistent_allocations(
        self,
        events: List[AllocationEvent],
        total_duration: float,
    ) -> List[LeakCandidate]:
        """Detect large allocations that persist for the entire session.

        Args:
            events: List of allocation events.
            total_duration: Total profiling duration in seconds.

        Returns:
            List of leak candidates for persistent allocations.
        """
        if total_duration <= 0:
            return []

        # Build allocation lifetime map
        alloc_times: dict[str, Tuple[float, int, Optional[str]]] = {}
        freed_keys: set[str] = set()

        for event in events:
            key = f"{event.device}:{event.size_bytes}:{event.tensor_dtype}"
            if event.event_type == AllocationEventType.ALLOCATE:
                if key not in alloc_times:
                    alloc_times[key] = (
                        event.timestamp,
                        event.size_bytes,
                        event.stack_trace,
                    )
            elif event.event_type == AllocationEventType.FREE:
                freed_keys.add(key)

        leaks: List[LeakCandidate] = []
        threshold_bytes = self._config.leak_growth_threshold_bytes

        for key, (ts, size, trace) in alloc_times.items():
            if key in freed_keys:
                continue
            if size < threshold_bytes:
                continue

            leaks.append(
                LeakCandidate(
                    description=(
                        f"Large persistent allocation: "
                        f"{size / (1024 * 1024):.1f} MB, never freed"
                    ),
                    severity=LeakSeverity.MEDIUM,
                    growth_rate_bytes_per_sec=0.0,
                    total_leaked_bytes=size,
                    first_seen=ts,
                    stack_trace=trace,
                    recommendation=(
                        "This allocation persists for the entire profiling "
                        "session. If this is a model or buffer, it may be "
                        "expected. Otherwise, ensure it is freed when no "
                        "longer needed."
                    ),
                )
            )

        return leaks


def _linear_regression(
    x: List[float],
    y: List[float],
) -> Tuple[float, float]:
    """Simple linear regression returning slope and R-squared.

    Args:
        x: Independent variable values.
        y: Dependent variable values.

    Returns:
        Tuple of (slope, r_squared).
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0

    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)
    sum_y2 = sum(yi * yi for yi in y)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-10:
        return 0.0, 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denom

    # R-squared
    mean_y = sum_y / n
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    if ss_tot < 1e-10:
        return slope, 1.0

    intercept = (sum_y - slope * sum_x) / n
    ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))
    r_squared = 1.0 - (ss_res / ss_tot)

    return slope, max(0.0, r_squared)


def _classify_growth_rate(rate: float) -> LeakSeverity:
    """Classify a memory growth rate into a severity level.

    Args:
        rate: Growth rate in bytes per second.

    Returns:
        Appropriate leak severity level.
    """
    if rate >= GROWTH_RATE_CRITICAL:
        return LeakSeverity.CRITICAL
    if rate >= GROWTH_RATE_HIGH:
        return LeakSeverity.HIGH
    if rate >= GROWTH_RATE_MEDIUM:
        return LeakSeverity.MEDIUM
    return LeakSeverity.LOW
