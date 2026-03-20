"""CLI entry point for GPU Memory Profiler."""

import argparse
import logging
import sys
from pathlib import Path

from gpu_memory_profiler.core import MemoryProfiler
from gpu_memory_profiler.models import ProfileConfig
from gpu_memory_profiler.utils import (
    estimate_model_memory,
    format_bytes,
    simulate_training_loop,
)
from gpu_memory_profiler.visualization import MemoryVisualizer

logger = logging.getLogger(__name__)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GPU Memory Profiler with leak detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Demo command
    demo_parser = subparsers.add_parser(
        "demo", help="Run a demo profiling session"
    )
    demo_parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of training iterations to simulate",
    )
    demo_parser.add_argument(
        "--leak",
        action="store_true",
        help="Inject memory leaks into the simulation",
    )
    demo_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for HTML visualizations",
    )

    # Estimate command
    est_parser = subparsers.add_parser(
        "estimate", help="Estimate model memory requirements"
    )
    est_parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="Parameter count (e.g., '125M', '7B', '70B')",
    )
    est_parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16", "bf16", "int8"],
        default="fp32",
        help="Parameter data type",
    )
    est_parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam", "adamw"],
        default="adam",
        help="Optimizer type",
    )
    est_parser.add_argument(
        "--batch-size", type=int, default=1, help="Training batch size"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.command == "demo":
        _run_demo(args)
    elif args.command == "estimate":
        _run_estimate(args)
    else:
        parser.print_help()
        sys.exit(1)


def _run_demo(args: argparse.Namespace) -> None:
    """Run the demo profiling session."""
    config = ProfileConfig(
        device="cpu",
        cpu_fallback=True,
        detect_leaks=True,
        capture_stack_traces=True,
    )
    profiler = MemoryProfiler(config)

    with profiler:
        tracker = profiler.tracker
        simulate_training_loop(
            tracker=tracker,
            num_iterations=args.iterations,
            leak_probability=0.3 if args.leak else 0.0,
            device="cpu",
        )

    report = profiler.report()
    viz = MemoryVisualizer(report)

    logger.info("Profiling complete. Summary:\n%s", viz.generate_summary_text())

    if args.output:
        output_dir = Path(args.output)
        profiler.visualize(output_dir)
        logger.info("Visualizations saved to %s/", output_dir)


def _run_estimate(args: argparse.Namespace) -> None:
    """Run memory estimation."""
    param_count = _parse_param_count(args.params)
    dtype_bytes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}[args.dtype]

    breakdown = estimate_model_memory(
        param_count=param_count,
        dtype_bytes=dtype_bytes,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
    )

    lines = [f"Memory Estimate: {args.params} parameters ({args.dtype})", "=" * 50]
    for component, size in breakdown.items():
        lines.append(f"  {component:20s}: {format_bytes(size):>12s}")
    lines.append("=" * 50)
    logger.info("\n".join(lines))


def _parse_param_count(s: str) -> int:
    """Parse parameter count string like '7B' or '125M'.

    Args:
        s: Parameter count string.

    Returns:
        Integer parameter count.
    """
    s = s.strip().upper()
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(s)


if __name__ == "__main__":
    main()
