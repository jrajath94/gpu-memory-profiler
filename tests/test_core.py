"""Tests for core profiler, tracker, leak detector, and visualization."""

import time

import pytest

from gpu_memory_profiler.core import MemoryProfiler
from gpu_memory_profiler.exceptions import (
    ProfilerAlreadyStartedError,
    ProfilerNotStartedError,
)
from gpu_memory_profiler.leak_detector import LeakDetector, _linear_regression
from gpu_memory_profiler.models import (
    AllocationEvent,
    AllocationEventType,
    LeakSeverity,
    MemorySnapshot,
    MemoryTimeline,
    ProfileConfig,
    ProfileReport,
)
from gpu_memory_profiler.tracker import AllocationTracker
from gpu_memory_profiler.utils import (
    estimate_model_memory,
    format_bytes,
    simulate_training_loop,
)
from gpu_memory_profiler.visualization import MemoryVisualizer


class TestAllocationTracker:
    """Tests for AllocationTracker."""

    def test_record_allocation(self, tracker: AllocationTracker) -> None:
        """Recording an allocation increases counters."""
        tracker.record_allocation(
            data_ptr=0x1000,
            size_bytes=4096,
            tensor_shape=(32, 32),
            tensor_dtype="float32",
            device="cpu",
        )
        assert tracker.allocation_count == 1
        assert tracker.current_allocated_bytes == 4096
        assert 0x1000 in tracker.live_allocations

    def test_record_free(self, tracker: AllocationTracker) -> None:
        """Freeing reduces current allocated bytes."""
        tracker.record_allocation(data_ptr=0x1000, size_bytes=4096)
        tracker.record_free(data_ptr=0x1000, size_bytes=4096)
        assert tracker.free_count == 1
        assert tracker.current_allocated_bytes == 0
        assert 0x1000 not in tracker.live_allocations

    def test_peak_tracking(self, tracker: AllocationTracker) -> None:
        """Peak tracks the maximum allocated bytes."""
        tracker.record_allocation(data_ptr=0x1000, size_bytes=4096)
        tracker.record_allocation(data_ptr=0x2000, size_bytes=8192)
        assert tracker.peak_allocated_bytes == 4096 + 8192
        tracker.record_free(data_ptr=0x1000, size_bytes=4096)
        assert tracker.peak_allocated_bytes == 4096 + 8192  # Still peak
        assert tracker.current_allocated_bytes == 8192

    def test_take_snapshot(self, tracker: AllocationTracker) -> None:
        """Snapshot captures current state."""
        tracker.record_allocation(data_ptr=0x1000, size_bytes=1024)
        tracker.record_allocation(data_ptr=0x2000, size_bytes=2048)
        snap = tracker.take_snapshot(device="cpu")
        assert snap.allocated_bytes == 3072
        assert snap.active_tensors == 2
        assert snap.allocation_count == 2

    def test_top_allocations(self, tracker: AllocationTracker) -> None:
        """Top allocations returns largest first."""
        tracker.record_allocation(data_ptr=0x1000, size_bytes=100)
        tracker.record_allocation(data_ptr=0x2000, size_bytes=500)
        tracker.record_allocation(data_ptr=0x3000, size_bytes=300)
        top = tracker.top_allocations(n=2)
        assert len(top) == 2
        assert top[0].size_bytes == 500
        assert top[1].size_bytes == 300

    def test_reset(self, tracker: AllocationTracker) -> None:
        """Reset clears all state."""
        tracker.record_allocation(data_ptr=0x1000, size_bytes=4096)
        tracker.reset()
        assert tracker.allocation_count == 0
        assert tracker.current_allocated_bytes == 0
        assert len(tracker.events) == 0

    def test_record_oom(self, tracker: AllocationTracker) -> None:
        """OOM event is recorded."""
        tracker.record_oom(requested_bytes=1024 * 1024 * 1024, device="cpu")
        events = tracker.events
        assert len(events) == 1
        assert events[0].event_type == AllocationEventType.OOM

    def test_free_nonexistent_ptr(self, tracker: AllocationTracker) -> None:
        """Freeing unknown pointer doesn't crash."""
        tracker.record_free(data_ptr=0xDEAD, size_bytes=1024)
        assert tracker.free_count == 1
        assert tracker.current_allocated_bytes == 0

    def test_ring_buffer_limit(self) -> None:
        """Events are limited by max_events config."""
        config = ProfileConfig(max_events=5, capture_stack_traces=False)
        tracker = AllocationTracker(config)
        for i in range(10):
            tracker.record_allocation(data_ptr=i, size_bytes=100)
        assert len(tracker.events) == 5


class TestLeakDetector:
    """Tests for LeakDetector."""

    def test_detect_monotonic_growth(
        self, growing_timeline: MemoryTimeline
    ) -> None:
        """Growing timeline triggers monotonic growth detection."""
        detector = LeakDetector()
        leaks = detector.analyze(growing_timeline)
        # Should detect at least the monotonic growth
        growth_leaks = [
            l for l in leaks if "Monotonic" in l.description
        ]
        assert len(growth_leaks) >= 1
        assert growth_leaks[0].growth_rate_bytes_per_sec > 0

    def test_no_leak_stable(
        self, stable_timeline: MemoryTimeline
    ) -> None:
        """Stable timeline should not trigger monotonic growth."""
        detector = LeakDetector()
        leaks = detector.analyze(stable_timeline)
        growth_leaks = [
            l for l in leaks if "Monotonic" in l.description
        ]
        assert len(growth_leaks) == 0

    def test_allocation_imbalance_detection(
        self, growing_timeline: MemoryTimeline
    ) -> None:
        """Timeline with no frees triggers imbalance detection."""
        detector = LeakDetector()
        leaks = detector.analyze(growing_timeline)
        imbalance_leaks = [
            l for l in leaks if "imbalance" in l.description
        ]
        assert len(imbalance_leaks) >= 1

    def test_no_imbalance_stable(
        self, stable_timeline: MemoryTimeline
    ) -> None:
        """Balanced allocs/frees should not trigger imbalance."""
        detector = LeakDetector()
        leaks = detector.analyze(stable_timeline)
        imbalance_leaks = [
            l for l in leaks if "imbalance" in l.description
        ]
        assert len(imbalance_leaks) == 0

    def test_severity_ordering(
        self, growing_timeline: MemoryTimeline
    ) -> None:
        """Leaks should be sorted by severity (most severe first)."""
        detector = LeakDetector()
        leaks = detector.analyze(growing_timeline)
        if len(leaks) >= 2:
            severity_order = {
                LeakSeverity.CRITICAL: 0,
                LeakSeverity.HIGH: 1,
                LeakSeverity.MEDIUM: 2,
                LeakSeverity.LOW: 3,
            }
            for i in range(len(leaks) - 1):
                assert (
                    severity_order[leaks[i].severity]
                    <= severity_order[leaks[i + 1].severity]
                )

    def test_too_few_snapshots(self) -> None:
        """Less than 5 snapshots should not trigger growth detection."""
        tl = MemoryTimeline(
            snapshots=[
                MemorySnapshot(
                    timestamp=i,
                    allocated_bytes=i * 10_000_000,
                    reserved_bytes=i * 10_000_000,
                    active_tensors=i,
                )
                for i in range(3)
            ],
            start_time=0,
        )
        detector = LeakDetector()
        leaks = detector.analyze(tl)
        growth_leaks = [
            l for l in leaks if "Monotonic" in l.description
        ]
        assert len(growth_leaks) == 0


class TestLinearRegression:
    """Tests for the internal linear regression function."""

    def test_perfect_line(self) -> None:
        """y = 2x + 1 should give slope=2, R²=1."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 5.0, 7.0, 9.0, 11.0]
        slope, r_sq = _linear_regression(x, y)
        assert slope == pytest.approx(2.0, abs=1e-6)
        assert r_sq == pytest.approx(1.0, abs=1e-6)

    def test_constant_values(self) -> None:
        """Constant y values should give slope=0, R²=1."""
        x = [1.0, 2.0, 3.0]
        y = [5.0, 5.0, 5.0]
        slope, r_sq = _linear_regression(x, y)
        assert slope == pytest.approx(0.0, abs=1e-6)
        assert r_sq == pytest.approx(1.0, abs=1e-6)

    def test_single_point(self) -> None:
        """Single point should return zeros."""
        slope, r_sq = _linear_regression([1.0], [1.0])
        assert slope == 0.0
        assert r_sq == 0.0

    def test_noisy_positive_slope(self) -> None:
        """Noisy positive trend should have positive slope and R²<1."""
        x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        y = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 11.0]
        slope, r_sq = _linear_regression(x, y)
        assert slope > 0
        assert 0.5 < r_sq < 1.0


class TestMemoryProfiler:
    """Tests for the main MemoryProfiler class."""

    def test_context_manager(self, config: ProfileConfig) -> None:
        """Profiler works as context manager."""
        profiler = MemoryProfiler(config)
        with profiler:
            profiler.track_allocation(0x1000, 4096, device="cpu")
            profiler.snapshot()
        report = profiler.report()
        assert report.timeline.total_allocations == 1

    def test_start_stop(self, config: ProfileConfig) -> None:
        """Manual start/stop cycle."""
        profiler = MemoryProfiler(config)
        profiler.start()
        profiler.track_allocation(0x1000, 4096)
        profiler.snapshot()
        report = profiler.stop()
        assert report is not None
        assert len(report.timeline.snapshots) >= 1

    def test_double_start_raises(self, config: ProfileConfig) -> None:
        """Starting twice raises error."""
        profiler = MemoryProfiler(config)
        profiler.start()
        with pytest.raises(ProfilerAlreadyStartedError):
            profiler.start()
        profiler.stop()

    def test_stop_without_start_raises(
        self, config: ProfileConfig
    ) -> None:
        """Stopping without start raises error."""
        profiler = MemoryProfiler(config)
        with pytest.raises(ProfilerNotStartedError):
            profiler.stop()

    def test_snapshot_without_start_raises(
        self, config: ProfileConfig
    ) -> None:
        """Snapshot without start raises error."""
        profiler = MemoryProfiler(config)
        with pytest.raises(ProfilerNotStartedError):
            profiler.snapshot()

    def test_report_without_start_raises(
        self, config: ProfileConfig
    ) -> None:
        """Report without start raises error."""
        profiler = MemoryProfiler(config)
        with pytest.raises(ProfilerNotStartedError):
            profiler.report()

    def test_track_when_not_running(
        self, config: ProfileConfig
    ) -> None:
        """Tracking when not running is a no-op."""
        profiler = MemoryProfiler(config)
        profiler.track_allocation(0x1000, 4096)
        # Should not raise

    def test_is_running(self, config: ProfileConfig) -> None:
        """is_running reflects profiler state."""
        profiler = MemoryProfiler(config)
        assert not profiler.is_running
        profiler.start()
        assert profiler.is_running
        profiler.stop()
        assert not profiler.is_running

    def test_report_auto_stops(self, config: ProfileConfig) -> None:
        """Calling report() on a running profiler auto-stops it."""
        profiler = MemoryProfiler(config)
        profiler.start()
        profiler.track_allocation(0x1000, 4096)
        report = profiler.report()
        assert not profiler.is_running
        assert report.timeline.total_allocations == 1

    def test_leak_detection_integration(
        self, config: ProfileConfig
    ) -> None:
        """Full profiling with leak detection."""
        config.detect_leaks = True
        profiler = MemoryProfiler(config)
        with profiler:
            # Simulate a leak: allocate without freeing
            for i in range(20):
                profiler.track_allocation(
                    i, 10 * 1024 * 1024, device="cpu"
                )
                profiler.snapshot()
                time.sleep(0.01)

        report = profiler.report()
        assert report.summary["total_allocations"] == 20
        assert report.summary["total_frees"] == 0


class TestVisualization:
    """Tests for MemoryVisualizer."""

    def test_timeline_html_generation(
        self, growing_timeline: MemoryTimeline
    ) -> None:
        """Timeline HTML contains expected elements."""
        report = ProfileReport(timeline=growing_timeline)
        report.build_summary()
        viz = MemoryVisualizer(report)
        html = viz.generate_timeline_html()
        assert "GPU Memory Timeline" in html
        assert "canvas" in html

    def test_empty_timeline_html(self) -> None:
        """Empty timeline produces a message page."""
        report = ProfileReport(timeline=MemoryTimeline())
        viz = MemoryVisualizer(report)
        html = viz.generate_timeline_html()
        assert "No snapshots recorded" in html

    def test_flame_graph_no_traces(
        self, growing_timeline: MemoryTimeline
    ) -> None:
        """Flame graph with no stack traces shows message."""
        report = ProfileReport(timeline=growing_timeline)
        viz = MemoryVisualizer(report)
        html = viz.generate_flame_graph_html()
        assert "No stack traces captured" in html

    def test_leak_report_html(
        self, growing_timeline: MemoryTimeline
    ) -> None:
        """Leak report HTML contains expected structure."""
        report = ProfileReport(timeline=growing_timeline)
        report.build_summary()
        viz = MemoryVisualizer(report)
        html = viz.generate_leak_report_html()
        assert "Leak Detection Report" in html

    def test_summary_text(
        self, growing_timeline: MemoryTimeline
    ) -> None:
        """Text summary contains key metrics."""
        report = ProfileReport(timeline=growing_timeline)
        report.build_summary()
        viz = MemoryVisualizer(report)
        text = viz.generate_summary_text()
        assert "Peak Memory" in text
        assert "Allocations" in text


class TestUtils:
    """Tests for utility functions."""

    def test_simulate_training_loop(
        self, tracker: AllocationTracker
    ) -> None:
        """Training simulation produces events and snapshots."""
        timeline = simulate_training_loop(
            tracker=tracker,
            num_iterations=5,
            batch_size=8,
            hidden_dim=256,
        )
        assert len(timeline.snapshots) == 5
        assert len(timeline.events) > 0

    def test_simulate_with_leaks(
        self, tracker: AllocationTracker
    ) -> None:
        """Training simulation with leaks produces unfreed allocations."""
        timeline = simulate_training_loop(
            tracker=tracker,
            num_iterations=20,
            leak_probability=1.0,  # Always leak
        )
        # With leak_probability=1.0, every iteration leaks
        allocs = sum(
            1 for e in timeline.events
            if e.event_type == AllocationEventType.ALLOCATE
        )
        frees = sum(
            1 for e in timeline.events
            if e.event_type == AllocationEventType.FREE
        )
        assert allocs > frees

    @pytest.mark.parametrize(
        "size_bytes,expected",
        [
            (100, "100 B"),
            (2048, "2.0 KB"),
            (5 * 1024 * 1024, "5.0 MB"),
            (3 * 1024 * 1024 * 1024, "3.00 GB"),
        ],
    )
    def test_format_bytes(self, size_bytes: int, expected: str) -> None:
        """Verify byte formatting across scales."""
        assert format_bytes(size_bytes) == expected

    def test_estimate_model_memory_7b(self) -> None:
        """Estimate memory for a 7B parameter model."""
        breakdown = estimate_model_memory(
            param_count=7_000_000_000,
            dtype_bytes=2,  # fp16
            optimizer="adam",
        )
        # Parameters: 7B * 2 bytes = 14 GB
        assert breakdown["parameters"] == 7_000_000_000 * 2
        # Gradients same size
        assert breakdown["gradients"] == breakdown["parameters"]
        # Adam has 2x optimizer state
        assert breakdown["optimizer_states"] == breakdown["parameters"] * 2
        # Total should be substantial
        assert breakdown["total"] > breakdown["parameters"]

    def test_estimate_model_memory_sgd(self) -> None:
        """SGD has no optimizer state."""
        breakdown = estimate_model_memory(
            param_count=1_000_000,
            dtype_bytes=4,
            optimizer="sgd",
        )
        assert breakdown["optimizer_states"] == 0
