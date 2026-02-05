"""Tests for visualization module — flame graphs, timeline, leak reports."""

import tempfile
from pathlib import Path

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
from gpu_memory_profiler.visualization import (
    MemoryVisualizer,
    _build_flame_tree,
    _tree_to_list,
)


class TestFlameGraphBuilding:
    """Tests for flame graph tree construction."""

    def test_build_tree_single_event(self) -> None:
        """Single allocation produces a root with one child."""
        events = [
            AllocationEvent(
                event_type=AllocationEventType.ALLOCATE,
                size_bytes=1024,
                stack_trace='  File "test.py", line 10, in func\n    x = torch.randn(10)\n',
            )
        ]
        tree = _build_flame_tree(events)
        assert tree["value"] == 1024
        assert len(tree["children"]) > 0

    def test_build_tree_multiple_events_same_stack(self) -> None:
        """Multiple events with same stack accumulate size."""
        trace = '  File "model.py", line 5, in forward\n    out = self.linear(x)\n'
        events = [
            AllocationEvent(
                event_type=AllocationEventType.ALLOCATE,
                size_bytes=1000,
                stack_trace=trace,
            ),
            AllocationEvent(
                event_type=AllocationEventType.ALLOCATE,
                size_bytes=2000,
                stack_trace=trace,
            ),
        ]
        tree = _build_flame_tree(events)
        assert tree["value"] == 3000

    def test_build_tree_no_stack_trace(self) -> None:
        """Events without stack traces are skipped."""
        events = [
            AllocationEvent(
                event_type=AllocationEventType.ALLOCATE,
                size_bytes=1024,
                stack_trace=None,
            )
        ]
        tree = _build_flame_tree(events)
        assert tree["value"] == 0

    def test_tree_to_list_empty(self) -> None:
        """Empty tree produces empty list."""
        node = {"name": "root", "value": 0, "children": {}}
        result = _tree_to_list(node)
        assert result == []

    def test_tree_to_list_nested(self) -> None:
        """Nested tree converts to list format."""
        node = {
            "name": "root",
            "value": 100,
            "children": {
                "child1": {
                    "name": "child1",
                    "value": 60,
                    "children": {},
                },
                "child2": {
                    "name": "child2",
                    "value": 40,
                    "children": {},
                },
            },
        }
        result = _tree_to_list(node)
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert names == {"child1", "child2"}


class TestVisualizerFileOutput:
    """Test HTML file writing."""

    def _make_report_with_traces(self) -> ProfileReport:
        """Create a report with stack-traced events for flame graph."""
        base_time = 1000.0
        events = [
            AllocationEvent(
                event_type=AllocationEventType.ALLOCATE,
                size_bytes=5 * 1024 * 1024,
                timestamp=base_time + i * 0.1,
                stack_trace=(
                    f'  File "model.py", line {10 + i}, in forward\n'
                    f"    tensor = torch.randn({i})\n"
                ),
            )
            for i in range(5)
        ]
        snapshots = [
            MemorySnapshot(
                timestamp=base_time + i * 0.1,
                allocated_bytes=(i + 1) * 5 * 1024 * 1024,
                reserved_bytes=(i + 2) * 5 * 1024 * 1024,
                active_tensors=i + 1,
            )
            for i in range(5)
        ]
        timeline = MemoryTimeline(
            snapshots=snapshots,
            events=events,
            start_time=base_time,
        )
        leaks = [
            LeakCandidate(
                description="Test leak",
                severity=LeakSeverity.HIGH,
                growth_rate_bytes_per_sec=1e6,
                total_leaked_bytes=25 * 1024 * 1024,
                recommendation="Fix it.",
            ),
        ]
        report = ProfileReport(timeline=timeline, leaks=leaks)
        report.build_summary()
        return report

    def test_timeline_html_to_file(self) -> None:
        """Timeline HTML writes to file."""
        report = self._make_report_with_traces()
        viz = MemoryVisualizer(report)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "timeline.html"
            html = viz.generate_timeline_html(output_path=path)
            assert path.exists()
            assert "GPU Memory Timeline" in html

    def test_flame_graph_html_to_file(self) -> None:
        """Flame graph HTML writes to file."""
        report = self._make_report_with_traces()
        viz = MemoryVisualizer(report)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flame.html"
            html = viz.generate_flame_graph_html(output_path=path)
            assert path.exists()
            assert "Flame Graph" in html

    def test_leak_report_html_to_file(self) -> None:
        """Leak report HTML writes to file."""
        report = self._make_report_with_traces()
        viz = MemoryVisualizer(report)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "leaks.html"
            html = viz.generate_leak_report_html(output_path=path)
            assert path.exists()
            assert "Leak Detection Report" in html

    def test_summary_text_with_leaks(self) -> None:
        """Summary text includes leak details."""
        report = self._make_report_with_traces()
        viz = MemoryVisualizer(report)
        text = viz.generate_summary_text()
        assert "Detected Leaks" in text
        assert "HIGH" in text
        assert "Fix:" in text

    def test_flame_graph_with_traces_renders(self) -> None:
        """Flame graph actually renders when traces are present."""
        report = self._make_report_with_traces()
        viz = MemoryVisualizer(report)
        html = viz.generate_flame_graph_html()
        assert "container" in html
        assert "tooltip" in html
        assert "MB" in html


class TestVisualizerEdgeCases:
    """Edge cases for visualization."""

    def test_leak_report_no_leaks(self) -> None:
        """Leak report with no leaks shows appropriate message."""
        timeline = MemoryTimeline()
        report = ProfileReport(timeline=timeline)
        report.build_summary()
        viz = MemoryVisualizer(report)
        html = viz.generate_leak_report_html()
        assert "No leaks detected" in html

    @pytest.mark.parametrize(
        "severity,expected_color",
        [
            (LeakSeverity.CRITICAL, "#ff4757"),
            (LeakSeverity.HIGH, "#ff6348"),
            (LeakSeverity.MEDIUM, "#ffa502"),
            (LeakSeverity.LOW, "#2ed573"),
        ],
    )
    def test_leak_severity_colors(
        self, severity: LeakSeverity, expected_color: str
    ) -> None:
        """Each severity level has a distinct color."""
        timeline = MemoryTimeline()
        leaks = [
            LeakCandidate(
                description="test",
                severity=severity,
                growth_rate_bytes_per_sec=100,
                total_leaked_bytes=1000,
            )
        ]
        report = ProfileReport(timeline=timeline, leaks=leaks)
        report.build_summary()
        viz = MemoryVisualizer(report)
        html = viz.generate_leak_report_html()
        assert expected_color in html
