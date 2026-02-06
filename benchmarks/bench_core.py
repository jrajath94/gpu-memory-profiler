"""Performance benchmarks for GPU Memory Profiler components."""

import logging
import time
from typing import Callable, Tuple

from gpu_memory_profiler.core import MemoryProfiler
from gpu_memory_profiler.leak_detector import LeakDetector
from gpu_memory_profiler.models import (
    AllocationEvent,
    AllocationEventType,
    MemorySnapshot,
    MemoryTimeline,
    ProfileConfig,
)
from gpu_memory_profiler.tracker import AllocationTracker
from gpu_memory_profiler.utils import simulate_training_loop
from gpu_memory_profiler.visualization import MemoryVisualizer

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def bench(
    name: str,
    fn: Callable[[], object],
    iterations: int = 1000,
) -> Tuple[float, float]:
    """Run a benchmark and return (ops/sec, avg_us).

    Args:
        name: Benchmark name.
        fn: Function to benchmark.
        iterations: Number of iterations.

    Returns:
        Tuple of (operations per second, average microseconds per op).
    """
    # Warmup
    for _ in range(min(100, iterations)):
        fn()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start

    ops_per_sec = iterations / elapsed
    avg_us = (elapsed / iterations) * 1e6
    return ops_per_sec, avg_us


def bench_allocation_tracking() -> None:
    """Benchmark allocation recording throughput."""
    config = ProfileConfig(
        capture_stack_traces=False,
        max_events=1_000_000,
    )
    tracker = AllocationTracker(config)
    ptr = 0

    def record_alloc() -> None:
        nonlocal ptr
        tracker.record_allocation(
            data_ptr=ptr,
            size_bytes=4096,
            tensor_shape=(32, 32),
            tensor_dtype="float32",
        )
        ptr += 4096

    ops, avg_us = bench("Allocation tracking", record_alloc, 100_000)
    print(f"  Allocation tracking:     {ops:>12,.0f} ops/s  ({avg_us:.1f} us/op)")


def bench_allocation_with_stack() -> None:
    """Benchmark allocation recording with stack trace capture."""
    config = ProfileConfig(
        capture_stack_traces=True,
        stack_trace_depth=5,
        max_events=100_000,
    )
    tracker = AllocationTracker(config)
    ptr = 0

    def record_alloc_stack() -> None:
        nonlocal ptr
        tracker.record_allocation(
            data_ptr=ptr,
            size_bytes=4096,
            tensor_shape=(32, 32),
            tensor_dtype="float32",
        )
        ptr += 4096

    ops, avg_us = bench("Alloc + stack trace", record_alloc_stack, 10_000)
    print(f"  Alloc + stack trace:     {ops:>12,.0f} ops/s  ({avg_us:.1f} us/op)")


def bench_snapshot() -> None:
    """Benchmark snapshot taking."""
    config = ProfileConfig(capture_stack_traces=False)
    tracker = AllocationTracker(config)
    # Pre-populate with allocations
    for i in range(100):
        tracker.record_allocation(data_ptr=i * 1000, size_bytes=4096)

    def take_snap() -> None:
        tracker.take_snapshot(device="cpu")

    ops, avg_us = bench("Snapshot", take_snap, 50_000)
    print(f"  Snapshot:                {ops:>12,.0f} ops/s  ({avg_us:.1f} us/op)")


def bench_leak_detection() -> None:
    """Benchmark leak detection analysis."""
    # Build a realistic timeline
    base_time = 1000.0
    snapshots = []
    events = []
    for i in range(100):
        t = base_time + i * 0.1
        snapshots.append(
            MemorySnapshot(
                timestamp=t,
                allocated_bytes=(i + 1) * 1024 * 1024,
                reserved_bytes=(i + 2) * 1024 * 1024,
                active_tensors=i + 1,
                allocation_count=i + 1,
                free_count=max(0, i - 5),
            )
        )
        events.append(
            AllocationEvent(
                event_type=AllocationEventType.ALLOCATE,
                size_bytes=1024 * 1024,
                timestamp=t,
            )
        )

    timeline = MemoryTimeline(
        snapshots=snapshots,
        events=events,
        start_time=base_time,
    )
    detector = LeakDetector()

    def detect() -> None:
        detector.analyze(timeline)

    ops, avg_us = bench("Leak detection (100 snaps)", detect, 1_000)
    print(f"  Leak detection:          {ops:>12,.0f} ops/s  ({avg_us:.1f} us/op)")


def bench_training_simulation() -> None:
    """Benchmark training loop simulation throughput."""
    config = ProfileConfig(capture_stack_traces=False, max_events=1_000_000)

    def simulate() -> None:
        tracker = AllocationTracker(config)
        simulate_training_loop(
            tracker=tracker,
            num_iterations=50,
            batch_size=32,
            hidden_dim=768,
        )

    ops, avg_us = bench("Training sim (50 iters)", simulate, 100)
    print(f"  Training sim (50 iter):  {ops:>12,.0f} ops/s  ({avg_us:.1f} us/op)")


def bench_full_pipeline() -> None:
    """Benchmark full profiling pipeline: track -> detect -> visualize."""
    config = ProfileConfig(
        device="cpu",
        cpu_fallback=True,
        detect_leaks=True,
        capture_stack_traces=False,
        max_events=100_000,
    )

    def full_pipeline() -> None:
        profiler = MemoryProfiler(config)
        with profiler:
            simulate_training_loop(
                tracker=profiler.tracker,
                num_iterations=20,
                batch_size=16,
                hidden_dim=512,
            )
        report = profiler.report()
        MemoryVisualizer(report).generate_summary_text()

    ops, avg_us = bench("Full pipeline (20 iters)", full_pipeline, 50)
    print(f"  Full pipeline:           {ops:>12,.0f} ops/s  ({avg_us:.1f} us/op)")


def main() -> None:
    """Run all benchmarks."""
    print("=" * 65)
    print("GPU Memory Profiler Benchmarks")
    print("=" * 65)
    print()

    bench_allocation_tracking()
    bench_allocation_with_stack()
    bench_snapshot()
    bench_leak_detection()
    bench_training_simulation()
    bench_full_pipeline()

    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
