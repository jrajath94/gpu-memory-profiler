"""Visualization module for generating memory timeline charts and flame graphs."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from gpu_memory_profiler.models import (
    AllocationEvent,
    AllocationEventType,
    LeakCandidate,
    ProfileReport,
)

logger = logging.getLogger(__name__)


class MemoryVisualizer:
    """Generates HTML visualizations of memory profiling data.

    Produces standalone HTML files with embedded JavaScript charts.
    No external dependencies (plotly, matplotlib) required -- uses
    inline SVG and Canvas rendering for zero-dep deployment.

    Args:
        report: Profile report containing timeline and leak data.
    """

    def __init__(self, report: ProfileReport) -> None:
        self._report = report
        self._timeline = report.timeline

    def generate_timeline_html(self, output_path: Optional[Path] = None) -> str:
        """Generate an interactive memory timeline as standalone HTML.

        Args:
            output_path: Optional path to write the HTML file.

        Returns:
            HTML string of the timeline visualization.
        """
        snapshots = self._timeline.snapshots
        if not snapshots:
            return _empty_chart_html("No snapshots recorded")

        base_time = self._timeline.start_time
        times = [s.timestamp - base_time for s in snapshots]
        allocated = [s.allocated_mb for s in snapshots]
        reserved = [s.reserved_mb for s in snapshots]

        data = {
            "times": [round(t, 3) for t in times],
            "allocated": [round(a, 2) for a in allocated],
            "reserved": [round(r, 2) for r in reserved],
            "peak_mb": round(self._timeline.peak_memory_mb, 2),
            "duration": round(self._timeline.duration_seconds, 2),
        }

        html = _render_timeline_chart(data)

        if output_path is not None:
            output_path.write_text(html, encoding="utf-8")
            logger.info("Timeline saved to %s", output_path)

        return html

    def generate_flame_graph_html(
        self,
        output_path: Optional[Path] = None,
    ) -> str:
        """Generate a memory flame graph from allocation stack traces.

        Groups allocations by call stack and visualizes as nested
        rectangles sized by total memory.

        Args:
            output_path: Optional path to write the HTML file.

        Returns:
            HTML string of the flame graph.
        """
        events = [
            e
            for e in self._timeline.events
            if e.event_type == AllocationEventType.ALLOCATE
            and e.stack_trace is not None
        ]

        if not events:
            return _empty_chart_html("No stack traces captured")

        tree = _build_flame_tree(events)
        html = _render_flame_graph(tree)

        if output_path is not None:
            output_path.write_text(html, encoding="utf-8")
            logger.info("Flame graph saved to %s", output_path)

        return html

    def generate_leak_report_html(
        self,
        output_path: Optional[Path] = None,
    ) -> str:
        """Generate an HTML leak detection report.

        Args:
            output_path: Optional path to write the HTML file.

        Returns:
            HTML string of the leak report.
        """
        leaks = self._report.leaks
        summary = self._report.summary or self._report.build_summary()

        html = _render_leak_report(leaks, summary)

        if output_path is not None:
            output_path.write_text(html, encoding="utf-8")
            logger.info("Leak report saved to %s", output_path)

        return html

    def generate_summary_text(self) -> str:
        """Generate a plain-text profiling summary.

        Returns:
            Formatted text summary of the profiling session.
        """
        summary = self._report.summary or self._report.build_summary()
        lines = [
            "GPU Memory Profile Summary",
            "=" * 40,
            f"Duration:         {summary.get('duration_seconds', 0):.2f}s",
            f"Peak Memory:      {summary.get('peak_memory_mb', 0):.2f} MB",
            f"Allocations:      {summary.get('total_allocations', 0)}",
            f"Frees:            {summary.get('total_frees', 0)}",
            f"Net Allocations:  {summary.get('net_allocations', 0)}",
            f"OOM Events:       {summary.get('oom_events', 0)}",
            f"Leak Candidates:  {summary.get('leak_candidates', 0)}",
            f"Device:           {summary.get('device', 'N/A')}",
            "=" * 40,
        ]

        if self._report.leaks:
            lines.append("")
            lines.append("Detected Leaks:")
            lines.append("-" * 40)
            for i, leak in enumerate(self._report.leaks, 1):
                lines.append(
                    f"  [{leak.severity.value.upper()}] {leak.description}"
                )
                if leak.recommendation:
                    lines.append(f"    Fix: {leak.recommendation}")

        return "\n".join(lines)


def _empty_chart_html(message: str) -> str:
    """Return a minimal HTML page with a message."""
    return f"""<!DOCTYPE html>
<html><head><title>GPU Memory Profiler</title></head>
<body style="font-family:monospace;padding:40px">
<h2>{message}</h2>
</body></html>"""


def _render_timeline_chart(data: Dict[str, Any]) -> str:
    """Render memory timeline as standalone HTML with inline JS chart."""
    data_json = json.dumps(data)
    return f"""<!DOCTYPE html>
<html>
<head>
<title>GPU Memory Timeline</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
h1 {{ color: #00d4aa; }}
.stats {{ display: flex; gap: 30px; margin: 20px 0; }}
.stat {{ background: #16213e; padding: 15px 25px; border-radius: 8px; }}
.stat-value {{ font-size: 24px; font-weight: bold; color: #00d4aa; }}
.stat-label {{ font-size: 12px; color: #888; }}
canvas {{ background: #16213e; border-radius: 8px; }}
</style>
</head>
<body>
<h1>GPU Memory Timeline</h1>
<div class="stats">
  <div class="stat"><div class="stat-value" id="peak">--</div><div class="stat-label">Peak (MB)</div></div>
  <div class="stat"><div class="stat-value" id="duration">--</div><div class="stat-label">Duration (s)</div></div>
</div>
<canvas id="chart" width="1200" height="400"></canvas>
<script>
const data = {data_json};
document.getElementById('peak').textContent = data.peak_mb;
document.getElementById('duration').textContent = data.duration;
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;
const pad = {{l:60, r:20, t:20, b:40}};
const pW = W-pad.l-pad.r, pH = H-pad.t-pad.b;
const maxT = Math.max(...data.times), maxM = Math.max(...data.allocated,...data.reserved)*1.1;
function tx(t) {{ return pad.l + (t/maxT)*pW; }}
function ty(m) {{ return pad.t + pH - (m/maxM)*pH; }}
ctx.strokeStyle = '#333'; ctx.lineWidth = 0.5;
for(let i=0;i<=5;i++) {{ let y=pad.t+(pH/5)*i; ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y); ctx.stroke(); }}
ctx.fillStyle = 'rgba(0,100,200,0.15)';
ctx.beginPath(); ctx.moveTo(tx(data.times[0]),ty(0));
data.times.forEach((t,i) => ctx.lineTo(tx(t),ty(data.reserved[i])));
ctx.lineTo(tx(data.times[data.times.length-1]),ty(0)); ctx.fill();
ctx.strokeStyle = '#00d4aa'; ctx.lineWidth = 2;
ctx.beginPath(); data.times.forEach((t,i) => {{ i===0?ctx.moveTo(tx(t),ty(data.allocated[i])):ctx.lineTo(tx(t),ty(data.allocated[i])); }});
ctx.stroke();
ctx.fillStyle = '#888'; ctx.font = '11px monospace';
ctx.fillText('Time (s)', W/2-20, H-5);
ctx.save(); ctx.translate(12, H/2); ctx.rotate(-Math.PI/2); ctx.fillText('Memory (MB)', 0, 0); ctx.restore();
for(let i=0;i<=5;i++) {{ ctx.fillText((maxM*(5-i)/5).toFixed(0), 5, pad.t+(pH/5)*i+4); }}
</script>
</body></html>"""


def _build_flame_tree(events: List[AllocationEvent]) -> Dict[str, Any]:
    """Build a hierarchical tree from allocation stack traces.

    Args:
        events: Allocation events with stack traces.

    Returns:
        Nested dictionary representing the flame graph tree.
    """
    root: Dict[str, Any] = {"name": "root", "value": 0, "children": {}}

    for event in events:
        if event.stack_trace is None:
            continue

        lines = event.stack_trace.strip().split("\n")
        frames = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("File "):
                parts = stripped.split(",")
                if len(parts) >= 2:
                    frames.append(
                        parts[0].replace("File ", "").strip('"')
                        + parts[1].strip()
                    )
            elif stripped and not stripped.startswith("File"):
                frames.append(stripped)

        if not frames:
            frames = ["<unknown>"]

        node = root
        node["value"] += event.size_bytes
        for frame in frames:
            if frame not in node["children"]:
                node["children"][frame] = {
                    "name": frame,
                    "value": 0,
                    "children": {},
                }
            node = node["children"][frame]
            node["value"] += event.size_bytes

    return root


def _render_flame_graph(tree: Dict[str, Any]) -> str:
    """Render a flame graph tree as standalone HTML."""
    tree_json = json.dumps(_tree_to_list(tree))
    return f"""<!DOCTYPE html>
<html>
<head>
<title>GPU Memory Flame Graph</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
h1 {{ color: #ff6b6b; }}
.bar {{ position: absolute; overflow: hidden; white-space: nowrap; font-size: 11px;
        line-height: 20px; padding: 0 4px; box-sizing: border-box; cursor: pointer; }}
.bar:hover {{ opacity: 0.8; }}
#container {{ position: relative; width: 100%; }}
#tooltip {{ position: fixed; background: #16213e; padding: 8px 12px; border-radius: 4px;
            border: 1px solid #333; display: none; pointer-events: none; z-index: 100; }}
</style>
</head>
<body>
<h1>GPU Memory Flame Graph</h1>
<div id="tooltip"></div>
<div id="container"></div>
<script>
const tree = {tree_json};
const container = document.getElementById('container');
const tooltip = document.getElementById('tooltip');
const colors = ['#ff6b6b','#ffa502','#ff6348','#ff4757','#e84393','#fd79a8','#e17055','#d63031'];
function render(nodes, x, w, depth) {{
  nodes.forEach(node => {{
    const div = document.createElement('div');
    div.className = 'bar';
    div.style.left = x+'%'; div.style.width = Math.max(w,0.5)+'%';
    div.style.top = (depth*22)+'px'; div.style.height = '20px';
    div.style.background = colors[depth % colors.length];
    div.textContent = node.name + ' (' + (node.value/(1024*1024)).toFixed(1) + ' MB)';
    div.addEventListener('mousemove', e => {{
      tooltip.style.display = 'block';
      tooltip.style.left = e.clientX+10+'px'; tooltip.style.top = e.clientY+10+'px';
      tooltip.textContent = node.name + ': ' + (node.value/(1024*1024)).toFixed(2) + ' MB';
    }});
    div.addEventListener('mouseout', () => {{ tooltip.style.display = 'none'; }});
    container.appendChild(div);
    if(node.children && node.children.length) {{
      let cx = x;
      const total = node.children.reduce((s,c) => s+c.value, 0);
      node.children.forEach(child => {{
        const cw = (child.value / total) * w;
        render([child], cx, cw, depth+1);
        cx += cw;
      }});
    }}
  }});
}}
render(tree, 0, 100, 0);
container.style.height = '600px';
</script>
</body></html>"""


def _tree_to_list(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert tree dict to a list format for JSON serialization."""
    children_list = []
    for child in node.get("children", {}).values():
        children_list.append(
            {
                "name": child["name"],
                "value": child["value"],
                "children": _tree_to_list(child),
            }
        )
    return children_list


def _render_leak_report(
    leaks: List[LeakCandidate],
    summary: Dict[str, Any],
) -> str:
    """Render leak detection results as HTML."""
    severity_colors = {
        "critical": "#ff4757",
        "high": "#ff6348",
        "medium": "#ffa502",
        "low": "#2ed573",
    }

    leak_rows = ""
    for leak in leaks:
        color = severity_colors.get(leak.severity.value, "#888")
        leak_rows += (
            "<tr>"
            f'<td><span style="color:{color};font-weight:bold">'
            f"{leak.severity.value.upper()}</span></td>"
            f"<td>{leak.description}</td>"
            f"<td>{leak.leaked_mb:.1f} MB</td>"
            f"<td>{leak.growth_rate_mb_per_min:.2f} MB/min</td>"
            f'<td style="font-size:11px">{leak.recommendation}</td>'
            "</tr>"
        )

    if not leak_rows:
        leak_rows = '<tr><td colspan="5">No leaks detected</td></tr>'

    return f"""<!DOCTYPE html>
<html>
<head>
<title>GPU Memory Leak Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
h1 {{ color: #ffa502; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #333; }}
th {{ background: #16213e; color: #00d4aa; }}
.summary {{ display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0; }}
.card {{ background: #16213e; padding: 12px 20px; border-radius: 6px; min-width: 140px; }}
.card-val {{ font-size: 20px; font-weight: bold; color: #00d4aa; }}
.card-lbl {{ font-size: 11px; color: #888; }}
</style>
</head>
<body>
<h1>Memory Leak Detection Report</h1>
<div class="summary">
  <div class="card"><div class="card-val">{summary.get('peak_memory_mb', 0):.1f} MB</div><div class="card-lbl">Peak Memory</div></div>
  <div class="card"><div class="card-val">{summary.get('duration_seconds', 0):.1f}s</div><div class="card-lbl">Duration</div></div>
  <div class="card"><div class="card-val">{summary.get('total_allocations', 0)}</div><div class="card-lbl">Allocations</div></div>
  <div class="card"><div class="card-val">{summary.get('leak_candidates', 0)}</div><div class="card-lbl">Leaks Found</div></div>
</div>
<h2>Leak Candidates</h2>
<table>
<tr><th>Severity</th><th>Description</th><th>Leaked</th><th>Growth Rate</th><th>Recommendation</th></tr>
{leak_rows}
</table>
</body></html>"""
