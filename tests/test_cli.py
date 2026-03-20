"""Tests for CLI entry point and utility edge cases."""

import sys
from unittest.mock import patch

import pytest

from gpu_memory_profiler.cli import _parse_param_count, main
from gpu_memory_profiler.models import ProfileConfig
from gpu_memory_profiler.tracker import AllocationTracker
from gpu_memory_profiler.utils import simulate_inference_batch


class TestParseParamCount:
    """Tests for parameter count parsing."""

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("125M", 125_000_000),
            ("7B", 7_000_000_000),
            ("1.5B", 1_500_000_000),
            ("100K", 100_000),
            ("1T", 1_000_000_000_000),
            ("1000", 1000),
        ],
    )
    def test_parse_various_formats(
        self, input_str: str, expected: int
    ) -> None:
        """Parse parameter count strings with various suffixes."""
        assert _parse_param_count(input_str) == expected


class TestCLIDemo:
    """Tests for CLI demo command."""

    def test_demo_command(self) -> None:
        """Demo runs without error."""
        test_args = ["gpu-profiler", "demo", "--iterations", "5"]
        with patch.object(sys, "argv", test_args):
            main()

    def test_demo_with_leak(self) -> None:
        """Demo with leak injection runs without error."""
        test_args = [
            "gpu-profiler", "demo", "--iterations", "5", "--leak",
        ]
        with patch.object(sys, "argv", test_args):
            main()

    def test_estimate_command(self) -> None:
        """Estimate command runs without error."""
        test_args = [
            "gpu-profiler", "estimate",
            "--params", "125M",
            "--dtype", "fp16",
            "--optimizer", "adam",
        ]
        with patch.object(sys, "argv", test_args):
            main()

    def test_no_command_exits(self) -> None:
        """No command prints help and exits."""
        test_args = ["gpu-profiler"]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                main()


class TestSimulateInferenceBatch:
    """Tests for inference batch simulation."""

    def test_inference_simulation(self) -> None:
        """Inference simulation produces balanced allocs/frees."""
        config = ProfileConfig(capture_stack_traces=False)
        tracker = AllocationTracker(config)
        timeline = simulate_inference_batch(
            tracker=tracker,
            num_requests=10,
            sequence_length=128,
            hidden_dim=256,
        )
        assert len(timeline.snapshots) == 10
        assert timeline.total_allocations == timeline.total_frees
