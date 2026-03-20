"""Shared fixtures for GPU Memory Profiler tests."""


import pytest

from gpu_memory_profiler.models import (
    AllocationEvent,
    AllocationEventType,
    MemorySnapshot,
    MemoryTimeline,
    ProfileConfig,
)
from gpu_memory_profiler.tracker import AllocationTracker


@pytest.fixture
def config() -> ProfileConfig:
    """Default test configuration."""
    return ProfileConfig(
        device="cpu",
        cpu_fallback=True,
        snapshot_interval_ms=50,
        capture_stack_traces=False,
        max_events=10_000,
    )


@pytest.fixture
def tracker(config: ProfileConfig) -> AllocationTracker:
    """Allocation tracker with test config."""
    return AllocationTracker(config)


@pytest.fixture
def sample_snapshot() -> MemorySnapshot:
    """A sample memory snapshot."""
    return MemorySnapshot(
        timestamp=1000.0,
        allocated_bytes=100 * 1024 * 1024,  # 100 MB
        reserved_bytes=128 * 1024 * 1024,   # 128 MB
        active_tensors=42,
        device="cuda:0",
        peak_allocated_bytes=150 * 1024 * 1024,
        allocation_count=200,
        free_count=158,
    )


@pytest.fixture
def growing_timeline() -> MemoryTimeline:
    """Timeline with monotonically growing memory (simulated leak)."""
    base_time = 1000.0
    snapshots = []
    events = []

    for i in range(20):
        t = base_time + i * 0.5
        allocated = (i + 1) * 10 * 1024 * 1024  # 10 MB growth per step
        snapshots.append(
            MemorySnapshot(
                timestamp=t,
                allocated_bytes=allocated,
                reserved_bytes=allocated + 5 * 1024 * 1024,
                active_tensors=i + 1,
                device="cpu",
                allocation_count=i + 1,
                free_count=0,
            )
        )
        events.append(
            AllocationEvent(
                event_type=AllocationEventType.ALLOCATE,
                size_bytes=10 * 1024 * 1024,
                timestamp=t,
                device="cpu",
            )
        )

    return MemoryTimeline(
        snapshots=snapshots,
        events=events,
        start_time=base_time,
        device="cpu",
    )


@pytest.fixture
def stable_timeline() -> MemoryTimeline:
    """Timeline with stable memory usage (no leak)."""
    base_time = 1000.0
    snapshots = []
    events = []

    for i in range(20):
        t = base_time + i * 0.5
        # Oscillates around 100 MB
        allocated = 100 * 1024 * 1024 + (i % 3 - 1) * 1024 * 1024
        snapshots.append(
            MemorySnapshot(
                timestamp=t,
                allocated_bytes=allocated,
                reserved_bytes=128 * 1024 * 1024,
                active_tensors=42,
                device="cpu",
                allocation_count=i + 1,
                free_count=i + 1,
            )
        )
        # Balanced alloc/free
        events.append(
            AllocationEvent(
                event_type=AllocationEventType.ALLOCATE,
                size_bytes=1024 * 1024,
                timestamp=t,
                device="cpu",
            )
        )
        events.append(
            AllocationEvent(
                event_type=AllocationEventType.FREE,
                size_bytes=1024 * 1024,
                timestamp=t + 0.1,
                device="cpu",
            )
        )

    return MemoryTimeline(
        snapshots=snapshots,
        events=events,
        start_time=base_time,
        device="cpu",
    )
