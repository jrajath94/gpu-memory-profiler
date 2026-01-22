"""Data models for GPU Memory Profiler."""

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class AllocationEventType(enum.Enum):
    """Types of memory allocation events."""

    ALLOCATE = "allocate"
    FREE = "free"
    RESIZE = "resize"
    OOM = "oom"


class LeakSeverity(enum.Enum):
    """Severity levels for detected memory leaks."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AllocationEvent:
    """A single memory allocation or deallocation event.

    Args:
        event_type: Type of memory event (allocate, free, resize, oom).
        size_bytes: Size of the allocation in bytes.
        timestamp: Unix timestamp of the event.
        tensor_shape: Shape of the tensor involved.
        tensor_dtype: Data type string of the tensor.
        device: Device string (e.g., 'cuda:0', 'cpu').
        stack_trace: Call stack at the time of allocation.
        tag: Optional user-provided label for the allocation.
    """

    event_type: AllocationEventType
    size_bytes: int
    timestamp: float = field(default_factory=time.time)
    tensor_shape: Optional[Tuple[int, ...]] = None
    tensor_dtype: Optional[str] = None
    device: str = "cpu"
    stack_trace: Optional[str] = None
    tag: Optional[str] = None

    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    @property
    def size_human(self) -> str:
        """Human-readable size string."""
        if self.size_bytes < 1024:
            return f"{self.size_bytes} B"
        if self.size_bytes < 1024 * 1024:
            return f"{self.size_bytes / 1024:.1f} KB"
        if self.size_bytes < 1024 * 1024 * 1024:
            return f"{self.size_mb:.1f} MB"
        return f"{self.size_bytes / (1024 ** 3):.2f} GB"


@dataclass
class MemorySnapshot:
    """A point-in-time snapshot of GPU memory state.

    Args:
        timestamp: Unix timestamp when snapshot was taken.
        allocated_bytes: Total allocated memory in bytes.
        reserved_bytes: Total reserved memory by the caching allocator.
        active_tensors: Number of live tensors.
        device: Device this snapshot is for.
        peak_allocated_bytes: Peak allocated memory since last reset.
        allocation_count: Cumulative allocation count.
        free_count: Cumulative free count.
    """

    timestamp: float
    allocated_bytes: int
    reserved_bytes: int
    active_tensors: int
    device: str = "cuda:0"
    peak_allocated_bytes: int = 0
    allocation_count: int = 0
    free_count: int = 0

    @property
    def allocated_mb(self) -> float:
        """Allocated memory in megabytes."""
        return self.allocated_bytes / (1024 * 1024)

    @property
    def reserved_mb(self) -> float:
        """Reserved memory in megabytes."""
        return self.reserved_bytes / (1024 * 1024)

    @property
    def fragmentation_ratio(self) -> float:
        """Ratio of reserved-but-unused memory to total reserved.

        Returns:
            Float between 0 and 1. Higher means more fragmentation.
        """
        if self.reserved_bytes == 0:
            return 0.0
        return 1.0 - (self.allocated_bytes / self.reserved_bytes)

    @property
    def net_allocations(self) -> int:
        """Net allocation count (allocations minus frees)."""
        return self.allocation_count - self.free_count


@dataclass
class MemoryTimeline:
    """A time series of memory snapshots with allocation events.

    Args:
        snapshots: Ordered list of memory snapshots.
        events: Ordered list of allocation events.
        start_time: When profiling started.
        device: Device being profiled.
    """

    snapshots: List[MemorySnapshot] = field(default_factory=list)
    events: List[AllocationEvent] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    device: str = "cuda:0"

    @property
    def duration_seconds(self) -> float:
        """Total profiling duration in seconds."""
        if not self.snapshots:
            return 0.0
        return self.snapshots[-1].timestamp - self.start_time

    @property
    def peak_memory_bytes(self) -> int:
        """Peak allocated memory across all snapshots."""
        if not self.snapshots:
            return 0
        return max(s.allocated_bytes for s in self.snapshots)

    @property
    def peak_memory_mb(self) -> float:
        """Peak allocated memory in megabytes."""
        return self.peak_memory_bytes / (1024 * 1024)

    @property
    def total_allocations(self) -> int:
        """Total number of allocation events."""
        return sum(
            1 for e in self.events if e.event_type == AllocationEventType.ALLOCATE
        )

    @property
    def total_frees(self) -> int:
        """Total number of free events."""
        return sum(1 for e in self.events if e.event_type == AllocationEventType.FREE)

    @property
    def oom_count(self) -> int:
        """Number of OOM events recorded."""
        return sum(1 for e in self.events if e.event_type == AllocationEventType.OOM)

    def memory_at_time(self, t: float) -> int:
        """Get allocated memory at a specific time.

        Args:
            t: Unix timestamp to query.

        Returns:
            Allocated bytes at or just before the given time.
        """
        result = 0
        for snapshot in self.snapshots:
            if snapshot.timestamp > t:
                break
            result = snapshot.allocated_bytes
        return result


@dataclass
class LeakCandidate:
    """A suspected memory leak identified by the detector.

    Args:
        description: Human-readable description of the leak.
        severity: How severe the leak appears.
        growth_rate_bytes_per_sec: Rate of memory growth in bytes/second.
        total_leaked_bytes: Estimated total leaked memory.
        first_seen: Timestamp when growth was first detected.
        stack_trace: Common allocation stack trace if available.
        recommendation: Suggested fix for the leak.
    """

    description: str
    severity: LeakSeverity
    growth_rate_bytes_per_sec: float
    total_leaked_bytes: int
    first_seen: float = 0.0
    stack_trace: Optional[str] = None
    recommendation: str = ""

    @property
    def growth_rate_mb_per_min(self) -> float:
        """Growth rate in MB per minute."""
        return (self.growth_rate_bytes_per_sec * 60) / (1024 * 1024)

    @property
    def leaked_mb(self) -> float:
        """Total leaked memory in megabytes."""
        return self.total_leaked_bytes / (1024 * 1024)


@dataclass
class ProfileConfig:
    """Configuration for the memory profiler.

    Args:
        device: CUDA device to profile.
        snapshot_interval_ms: Milliseconds between automatic snapshots.
        track_allocations: Whether to track individual allocation events.
        capture_stack_traces: Whether to capture Python stack traces.
        stack_trace_depth: Maximum depth of captured stack traces.
        detect_leaks: Whether to run leak detection.
        leak_growth_threshold_bytes: Minimum growth rate to flag as leak.
        cpu_fallback: Use CPU tensor tracking when CUDA unavailable.
        max_events: Maximum allocation events to store (ring buffer).
    """

    device: str = "cuda:0"
    snapshot_interval_ms: int = 100
    track_allocations: bool = True
    capture_stack_traces: bool = True
    stack_trace_depth: int = 10
    detect_leaks: bool = True
    leak_growth_threshold_bytes: int = 1024 * 1024  # 1 MB
    cpu_fallback: bool = True
    max_events: int = 100_000


@dataclass
class ProfileReport:
    """Complete profiling report with timeline, leaks, and summary.

    Args:
        timeline: Memory timeline with snapshots and events.
        leaks: Detected memory leak candidates.
        config: Configuration used for profiling.
        summary: Key-value summary statistics.
    """

    timeline: MemoryTimeline
    leaks: List[LeakCandidate] = field(default_factory=list)
    config: Optional[ProfileConfig] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    def build_summary(self) -> Dict[str, Any]:
        """Build summary statistics from the timeline data.

        Returns:
            Dictionary of summary statistics.
        """
        tl = self.timeline
        self.summary = {
            "duration_seconds": round(tl.duration_seconds, 2),
            "peak_memory_mb": round(tl.peak_memory_mb, 2),
            "total_allocations": tl.total_allocations,
            "total_frees": tl.total_frees,
            "net_allocations": tl.total_allocations - tl.total_frees,
            "oom_events": tl.oom_count,
            "snapshot_count": len(tl.snapshots),
            "event_count": len(tl.events),
            "leak_candidates": len(self.leaks),
            "device": tl.device,
        }
        return self.summary
