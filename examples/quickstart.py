"""Quickstart example for GPU Memory Profiler."""

import logging
import time

from gpu_memory_profiler import (
    MemoryProfiler,
    ProfileConfig,
)
from gpu_memory_profiler.utils import (
    estimate_model_memory,
    format_bytes,
    simulate_training_loop,
)
from gpu_memory_profiler.visualization import MemoryVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def demo_basic_profiling() -> None:
    """Demonstrate basic memory profiling with context manager."""
    logger.info("=== Basic Profiling Demo ===")

    config = ProfileConfig(
        device="cpu",
        cpu_fallback=True,
        detect_leaks=True,
        capture_stack_traces=False,
    )

    profiler = MemoryProfiler(config)

    with profiler:
        # Simulate allocations
        for i in range(10):
            profiler.track_allocation(
                data_ptr=i * 0x1000,
                size_bytes=(i + 1) * 1024 * 1024,
                device="cpu",
                tensor_shape=((i + 1) * 256, 1024),
                tensor_dtype="float32",
            )
            profiler.snapshot()
            time.sleep(0.01)

        # Free some
        for i in range(0, 10, 2):
            profiler.track_free(
                data_ptr=i * 0x1000,
                size_bytes=(i + 1) * 1024 * 1024,
            )
            profiler.snapshot()

    report = profiler.report()
    viz = MemoryVisualizer(report)
    logger.info("\n%s", viz.generate_summary_text())


def demo_training_simulation() -> None:
    """Demonstrate training loop simulation with leak detection."""
    logger.info("=== Training Simulation Demo ===")

    config = ProfileConfig(
        device="cpu",
        cpu_fallback=True,
        detect_leaks=True,
        capture_stack_traces=False,
    )
    profiler = MemoryProfiler(config)

    with profiler:
        timeline = simulate_training_loop(
            tracker=profiler.tracker,
            num_iterations=15,
            batch_size=32,
            hidden_dim=768,
            leak_probability=0.2,
            device="cpu",
        )

    report = profiler.report()
    viz = MemoryVisualizer(report)
    logger.info("\n%s", viz.generate_summary_text())

    if report.leaks:
        logger.info("Leak details:")
        for leak in report.leaks:
            logger.info(
                "  [%s] %s (%.1f MB)",
                leak.severity.value.upper(),
                leak.description,
                leak.leaked_mb,
            )


def demo_memory_estimation() -> None:
    """Demonstrate model memory estimation."""
    logger.info("=== Memory Estimation Demo ===")

    models = [
        ("GPT-2 Small", 125_000_000),
        ("GPT-2 XL", 1_500_000_000),
        ("LLaMA 7B", 7_000_000_000),
        ("LLaMA 70B", 70_000_000_000),
    ]

    for name, params in models:
        breakdown = estimate_model_memory(
            param_count=params,
            dtype_bytes=2,  # fp16
            optimizer="adam",
            batch_size=1,
        )
        logger.info(
            "  %s (%s params): %s total",
            name,
            format_bytes(params),
            format_bytes(breakdown["total"]),
        )


if __name__ == "__main__":
    demo_basic_profiling()
    logger.info("")
    demo_training_simulation()
    logger.info("")
    demo_memory_estimation()
