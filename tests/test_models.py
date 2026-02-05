"""Tests for data models."""

import time

import pytest

from gpu_memory_profiler.models import (
    AllocationEvent,
    AllocationEventType,
    LeakCandidate,
    LeakSeverity,
    MemorySnapshot,
    MemoryTimeline,
    ProfileConfig,
    ProfileReport,
)


class TestAllocationEvent:
    """Tests for AllocationEvent dataclass."""

    def test_size_mb_conversion(self) -> None:
        """Verify byte-to-MB conversion."""
        event = AllocationEvent(
            event_type=AllocationEventType.ALLOCATE,
            size_bytes=10 * 1024 * 1024,  # 10 MB
        )
        assert event.size_mb == pytest.approx(10.0)

    @pytest.mark.parametrize(
        "size_bytes,expected",
        [
            (512, "512 B"),
            (1536, "1.5 KB"),
            (10 * 1024 * 1024, "10.0 MB"),
            (2 * 1024 * 1024 * 1024, "2.00 GB"),
        ],
    )
    def test_size_human_formatting(
        self, size_bytes: int, expected: str
    ) -> None:
        """Verify human-readable size formatting across scales."""
        event = AllocationEvent(
            event_type=AllocationEventType.ALLOCATE,
            size_bytes=size_bytes,
        )
        assert event.size_human == expected

    def test_event_type_values(self) -> None:
        """Verify all event types are accessible."""
        assert AllocationEventType.ALLOCATE.value == "allocate"
        assert AllocationEventType.FREE.value == "free"
        assert AllocationEventType.RESIZE.value == "resize"
        assert AllocationEventType.OOM.value == "oom"

    def test_default_timestamp(self) -> None:
        """Verify timestamp defaults to current time."""
        before = time.time()
        event = AllocationEvent(
            event_type=AllocationEventType.ALLOCATE,
            size_bytes=1024,
        )
        after = time.time()
        assert before <= event.timestamp <= after


class TestMemorySnapshot:
    """Tests for MemorySnapshot dataclass."""

    def test_allocated_mb(self, sample_snapshot: MemorySnapshot) -> None:
        """Verify allocated MB calculation."""
        assert sample_snapshot.allocated_mb == pytest.approx(100.0)

    def test_reserved_mb(self, sample_snapshot: MemorySnapshot) -> None:
        """Verify reserved MB calculation."""
        assert sample_snapshot.reserved_mb == pytest.approx(128.0)

    def test_fragmentation_ratio(
        self, sample_snapshot: MemorySnapshot
    ) -> None:
        """Fragmentation = 1 - (allocated/reserved)."""
        expected = 1.0 - (100.0 / 128.0)
        assert sample_snapshot.fragmentation_ratio == pytest.approx(
            expected, abs=0.01
        )

    def test_fragmentation_zero_reserved(self) -> None:
        """Zero reserved bytes means zero fragmentation."""
        snap = MemorySnapshot(
            timestamp=0, allocated_bytes=0, reserved_bytes=0,
            active_tensors=0,
        )
        assert snap.fragmentation_ratio == 0.0

    def test_net_allocations(self, sample_snapshot: MemorySnapshot) -> None:
        """Net allocations = allocs - frees."""
        assert sample_snapshot.net_allocations == 42  # 200 - 158


class TestMemoryTimeline:
    """Tests for MemoryTimeline dataclass."""

    def test_duration_seconds(
        self, growing_timeline: MemoryTimeline
    ) -> None:
        """Duration is last snapshot time minus start time."""
        expected = 19 * 0.5  # 20 snapshots, 0.5s apart
        assert growing_timeline.duration_seconds == pytest.approx(expected)

    def test_peak_memory(self, growing_timeline: MemoryTimeline) -> None:
        """Peak memory is the maximum across all snapshots."""
        expected = 20 * 10 * 1024 * 1024
        assert growing_timeline.peak_memory_bytes == expected

    def test_total_allocations(
        self, growing_timeline: MemoryTimeline
    ) -> None:
        """Count of allocation events."""
        assert growing_timeline.total_allocations == 20

    def test_total_frees_growing(
        self, growing_timeline: MemoryTimeline
    ) -> None:
        """Growing timeline has no frees."""
        assert growing_timeline.total_frees == 0

    def test_total_frees_stable(
        self, stable_timeline: MemoryTimeline
    ) -> None:
        """Stable timeline has equal allocs and frees."""
        assert stable_timeline.total_frees == stable_timeline.total_allocations

    def test_empty_timeline(self) -> None:
        """Empty timeline has zero values."""
        tl = MemoryTimeline()
        assert tl.duration_seconds == 0.0
        assert tl.peak_memory_bytes == 0
        assert tl.total_allocations == 0

    def test_memory_at_time(self, growing_timeline: MemoryTimeline) -> None:
        """Query memory at a specific timestamp."""
        # After 5 snapshots at t=1000+2.5, should be 5*10MB = 50MB
        mem = growing_timeline.memory_at_time(1002.6)
        assert mem == 6 * 10 * 1024 * 1024  # 6th snapshot (0-indexed: t=1002.5)

    def test_oom_count(self) -> None:
        """OOM count reflects OOM events."""
        tl = MemoryTimeline(
            events=[
                AllocationEvent(
                    event_type=AllocationEventType.OOM,
                    size_bytes=1024,
                ),
                AllocationEvent(
                    event_type=AllocationEventType.ALLOCATE,
                    size_bytes=512,
                ),
            ]
        )
        assert tl.oom_count == 1


class TestLeakCandidate:
    """Tests for LeakCandidate dataclass."""

    def test_growth_rate_mb_per_min(self) -> None:
        """Growth rate conversion from bytes/sec to MB/min."""
        leak = LeakCandidate(
            description="test",
            severity=LeakSeverity.HIGH,
            growth_rate_bytes_per_sec=1024 * 1024,  # 1 MB/s
            total_leaked_bytes=0,
        )
        assert leak.growth_rate_mb_per_min == pytest.approx(60.0)

    def test_leaked_mb(self) -> None:
        """Leaked MB conversion."""
        leak = LeakCandidate(
            description="test",
            severity=LeakSeverity.LOW,
            growth_rate_bytes_per_sec=0,
            total_leaked_bytes=50 * 1024 * 1024,
        )
        assert leak.leaked_mb == pytest.approx(50.0)

    def test_severity_levels(self) -> None:
        """Verify all severity levels exist."""
        assert LeakSeverity.LOW.value == "low"
        assert LeakSeverity.MEDIUM.value == "medium"
        assert LeakSeverity.HIGH.value == "high"
        assert LeakSeverity.CRITICAL.value == "critical"


class TestProfileConfig:
    """Tests for ProfileConfig defaults."""

    def test_default_values(self) -> None:
        """Verify sensible defaults."""
        config = ProfileConfig()
        assert config.device == "cuda:0"
        assert config.snapshot_interval_ms == 100
        assert config.track_allocations is True
        assert config.capture_stack_traces is True
        assert config.detect_leaks is True
        assert config.max_events == 100_000

    def test_leak_threshold_default(self) -> None:
        """Default leak threshold is 1 MB."""
        config = ProfileConfig()
        assert config.leak_growth_threshold_bytes == 1024 * 1024


class TestProfileReport:
    """Tests for ProfileReport."""

    def test_build_summary(
        self, growing_timeline: MemoryTimeline
    ) -> None:
        """Build summary from timeline data."""
        report = ProfileReport(timeline=growing_timeline)
        summary = report.build_summary()
        assert summary["total_allocations"] == 20
        assert summary["total_frees"] == 0
        assert summary["net_allocations"] == 20
        assert summary["peak_memory_mb"] > 0
        assert summary["leak_candidates"] == 0
        assert summary["snapshot_count"] == 20

    def test_build_summary_with_leaks(
        self, growing_timeline: MemoryTimeline
    ) -> None:
        """Summary reflects leak count."""
        leaks = [
            LeakCandidate(
                description="test",
                severity=LeakSeverity.HIGH,
                growth_rate_bytes_per_sec=1e6,
                total_leaked_bytes=100 * 1024 * 1024,
            )
        ]
        report = ProfileReport(
            timeline=growing_timeline, leaks=leaks
        )
        summary = report.build_summary()
        assert summary["leak_candidates"] == 1
