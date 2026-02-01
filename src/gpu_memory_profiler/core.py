"""Core memory profiler that orchestrates tracking, snapshotting, and analysis."""

import logging
import time
from pathlib import Path
from typing import Optional

from gpu_memory_profiler.exceptions import (
    ProfilerAlreadyStartedError,
    ProfilerNotStartedError,
)
from gpu_memory_profiler.leak_detector import LeakDetector
from gpu_memory_profiler.models import (
    MemoryTimeline,
    ProfileConfig,
    ProfileReport,
)
from gpu_memory_profiler.tracker import AllocationTracker
from gpu_memory_profiler.visualization import MemoryVisualizer

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """High-level GPU memory profiler with automatic snapshotting and leak detection.

    Provides a context manager interface for profiling code blocks.
    Automatically takes periodic snapshots, tracks allocations, and
    runs leak detection on stop.

    Args:
        config: Profiling configuration. Uses defaults if not provided.

    Example:
        >>> profiler = MemoryProfiler()
        >>> with profiler:
        ...     # Code to profile
        ...     tensor = torch.randn(1000, 1000, device='cuda')
        >>> report = profiler.report()
        >>> print(report.summary)
    """

    def __init__(self, config: Optional[ProfileConfig] = None) -> None:
        self._config = config or ProfileConfig()
        self._tracker = AllocationTracker(self._config)
        self._leak_detector = LeakDetector(self._config)
        self._timeline = MemoryTimeline(device=self._config.device)
        self._running = False
        self._report: Optional[ProfileReport] = None

    def start(self) -> None:
        """Start profiling.

        Raises:
            ProfilerAlreadyStartedError: If profiler is already running.
        """
        if self._running:
            raise ProfilerAlreadyStartedError()

        self._running = True
        self._timeline = MemoryTimeline(
            start_time=time.time(),
            device=self._config.device,
        )
        self._tracker.reset()
        self._report = None

        logger.info(
            "Memory profiler started (device=%s, interval=%dms)",
            self._config.device,
            self._config.snapshot_interval_ms,
        )

    def stop(self) -> ProfileReport:
        """Stop profiling and generate the report.

        Returns:
            Complete profiling report with timeline and leak analysis.

        Raises:
            ProfilerNotStartedError: If profiler hasn't been started.
        """
        if not self._running:
            raise ProfilerNotStartedError()

        self._running = False

        # Take final snapshot
        self._take_snapshot_internal()

        # Collect events into timeline
        self._timeline.events = self._tracker.events

        # Run leak detection
        leaks = []
        if self._config.detect_leaks:
            leaks = self._leak_detector.analyze(self._timeline)
            if leaks:
                logger.warning(
                    "Detected %d potential memory leak(s)", len(leaks)
                )

        # Build report
        self._report = ProfileReport(
            timeline=self._timeline,
            leaks=leaks,
            config=self._config,
        )
        self._report.build_summary()

        logger.info(
            "Memory profiler stopped. Peak: %.1f MB, %d events, %d leaks",
            self._timeline.peak_memory_mb,
            len(self._timeline.events),
            len(leaks),
        )

        return self._report

    def snapshot(self) -> None:
        """Take a manual memory snapshot.

        Raises:
            ProfilerNotStartedError: If profiler hasn't been started.
        """
        if not self._running:
            raise ProfilerNotStartedError()

        self._take_snapshot_internal()

    def _take_snapshot_internal(self) -> None:
        """Internal snapshot without running check."""
        snap = self._tracker.take_snapshot(device=self._config.device)
        self._timeline.snapshots.append(snap)

    def track_allocation(
        self,
        data_ptr: int,
        size_bytes: int,
        **kwargs: object,
    ) -> None:
        """Track a memory allocation event.

        Args:
            data_ptr: Memory address of the allocation.
            size_bytes: Size of allocation in bytes.
            **kwargs: Additional allocation metadata.
        """
        if not self._running:
            return

        self._tracker.record_allocation(
            data_ptr=data_ptr,
            size_bytes=size_bytes,
            tensor_shape=kwargs.get("tensor_shape"),  # type: ignore[arg-type]
            tensor_dtype=kwargs.get("tensor_dtype"),  # type: ignore[arg-type]
            device=kwargs.get("device", self._config.device),  # type: ignore[arg-type]
            tag=kwargs.get("tag"),  # type: ignore[arg-type]
        )

    def track_free(
        self,
        data_ptr: int,
        size_bytes: int,
        **kwargs: object,
    ) -> None:
        """Track a memory deallocation event.

        Args:
            data_ptr: Memory address being freed.
            size_bytes: Size of the freed allocation.
            **kwargs: Additional metadata.
        """
        if not self._running:
            return

        self._tracker.record_free(
            data_ptr=data_ptr,
            size_bytes=size_bytes,
            device=kwargs.get("device", self._config.device),  # type: ignore[arg-type]
        )

    def report(self) -> ProfileReport:
        """Get the profiling report.

        Returns:
            The generated report. Calls stop() if still running.

        Raises:
            ProfilerNotStartedError: If never started.
        """
        if self._running:
            return self.stop()
        if self._report is not None:
            return self._report
        raise ProfilerNotStartedError()

    def visualize(
        self,
        output_dir: Optional[Path] = None,
    ) -> MemoryVisualizer:
        """Create a visualizer for the profiling data.

        Args:
            output_dir: Optional directory to save HTML files.

        Returns:
            MemoryVisualizer instance.

        Raises:
            ProfilerNotStartedError: If no report available.
        """
        report = self.report()
        viz = MemoryVisualizer(report)

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            viz.generate_timeline_html(output_dir / "timeline.html")
            viz.generate_flame_graph_html(output_dir / "flamegraph.html")
            viz.generate_leak_report_html(output_dir / "leaks.html")
            logger.info("Visualizations saved to %s", output_dir)

        return viz

    @property
    def tracker(self) -> AllocationTracker:
        """Access the underlying allocation tracker."""
        return self._tracker

    @property
    def is_running(self) -> bool:
        """Whether the profiler is currently active."""
        return self._running

    def __enter__(self) -> "MemoryProfiler":
        """Start profiling on context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop profiling on context manager exit."""
        if self._running:
            self.stop()
