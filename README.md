# gpu-memory-profiler

A GPU memory profiler with leak detection. Flame graphs, timeline charts, and
leak heuristics for PyTorch GPU allocations, via a manual instrumentation API.

**Verified 2026-09-21:** 95 tests pass, 96% line coverage, on CPU with no GPU
needed for the suite.

## Why this exists

PyTorch's built-in memory tools show total bytes allocated, not where the
bytes came from or why they are not being freed. Every ML engineer has hit an
OOM with no actionable information.

The root cause is a visibility gap. PyTorch's caching allocator requests large
chunks from CUDA (512 MB at a time) and subdivides them internally. When you
`del` a tensor, PyTorch marks the block as free but does not return it to
CUDA, so `nvidia-smi` shows high memory even after freeing tensors. Real
leaks (retained computation graphs, global tensor references, DataLoader
workers initializing CUDA contexts) hide behind this noise.

This profiler records allocation and free events and turns them into three
diagnostics: a timeline of memory over the run, a flame graph of where
memory was allocated, and a leak report from three heuristics.

## How it works

You instrument the code you want to watch. The API is a context manager plus
explicit calls:

```
from pathlib import Path

from gpu_memory_profiler import MemoryProfiler, ProfileConfig

profiler = MemoryProfiler(ProfileConfig(device="cuda:0", detect_leaks=True))
with profiler:
    # your PyTorch code
    profiler.track_allocation(ptr, size, device="cuda:0")
    # ... later ...
    profiler.track_free(ptr, size)
    profiler.snapshot()

report = profiler.report()
profiler.visualize(output_dir=Path("./output"))   # timeline.html, flamegraph.html, leaks.html
```

The `AllocationTracker` is the hot path. Every event flows through it under
one lock: events, the live-allocation map, and the counters move together so
concurrent DataLoader workers cannot corrupt the accounting. Events are stored
in a ring buffer (`deque(maxlen=N)`), so the event log stays bounded during
a long session. (Snapshots and the live-allocation map still grow with
activity.) Stack traces are captured per allocation of 512 bytes or more,
with profiler-internal frames filtered out, so traces show your code.

`snapshot()` captures point-in-time state. The `LeakDetector` runs three
heuristics over the snapshots, documented as heuristics, not oracles:

1. Growth-rate regression: fits a line to memory over time; flags the leak
   when the slope is positive and R^2 > 0.7.
2. Allocation/free imbalance: counts total allocs vs total frees; flags when
   more than 20% of allocations go unfreed.
3. Persistent large allocations: flags allocations over the leak-growth
   threshold (1 MB by default) that are never freed during the session.

The `MemoryVisualizer` generates standalone HTML: a Canvas timeline chart,
an HTML flame graph, and a leak report page. No server, no external
dependencies. A CLI ships two commands: `gpu-profiler demo` (runs a scripted
example; add `--leak` to inject leaks) and `gpu-profiler estimate`
(estimates memory for a model size and dtype).

## What it is not

This profiler does **not** attach to PyTorch's allocation hooks. There is no
automatic interception of allocations. You call `track_allocation` and
`track_free` yourself, or wrap the provided context manager. Implementing real
hook integration (`torch.cuda.memory` hooks or `torch.profiler` kineto
callbacks that auto-register allocations) is the top item in "What I'd
improve".

Two known gaps. `take_snapshot` reports `reserved_bytes` equal to allocated
bytes. It is a placeholder, not a measurement of PyTorch's caching-allocator
behavior. And `record_free` for a pointer that was never tracked still bumps
the free counters, corrupting the accounting silently (the live total is
clamped at zero, so it cannot go negative). Both are tracked fixes.

## Architecture

```
User code -->|track_allocation / track_free| AllocationTracker
AllocationTracker -->|ring buffer events| MemoryTimeline
AllocationTracker -->|point-in-time snapshots| MemoryTimeline
MemoryTimeline --> LeakDetector --> LeakCandidates --> Leak report
MemoryTimeline --> MemoryVisualizer --> standalone HTML (timeline, flame graph, leak report)
CLI -->|gpu-profiler demo| AllocationTracker
CLI -->|gpu-profiler estimate| Memory Estimator
```

## Memory estimation for inference workloads

`gpu-profiler estimate --params 7B --dtype fp16 --optimizer adam` prints a
per-component breakdown (parameters, gradients, optimizer states,
activations). Example output:

```
Memory Estimate: 7B parameters (fp16)
==================================================
  parameters          :     13.04 GB
  gradients           :     13.04 GB
  optimizer_states    :     26.08 GB
  activations         :      2.17 GB
  total               :     54.33 GB
==================================================
```

A rough sizing step before renting GPU time: parameters, gradients, optimizer
states, and activations, computed from the estimator's formula.

## Performance

Run `make bench` to generate benchmarks on your hardware. Example results
(self-reported; fill in your own hardware line below):

| Component | Throughput | Latency (avg) | Conditions |
| --- | --- | --- | --- |
| Allocation tracking | 386,543 ops/s | 2.6 us | no stack traces, ring buffer |
| Alloc + stack trace | 30,739 ops/s | 32.5 us | 5-frame trace capture |
| Snapshot | 756,293 ops/s | 1.3 us | 100 live allocations |
| Leak detection | 5,859 ops/s | 170.7 us | 100 snapshots, 3 heuristics |
| Training simulation | 252 sims/s | 3,961.9 us | 50 iterations, 6 layers |
| Full pipeline | 382 runs/s | 2,620.6 us | track + detect + visualize |

Hardware for the numbers above: [GPU model, CPU, Python/PyTorch versions,
date - fill in].

## Testing

```
make test    # 95 tests, 96% coverage (verified 2026-09-21)
make bench   # component throughput benchmarks
make lint    # ruff + mypy
```

## Key design decisions

- **Ring buffer for events.** The event log stays bounded; snapshots and the
  live-allocation map still grow with activity. Tradeoff: oldest events are
  lost.
- **Linear regression for leak detection.** Interpretable (slope =
  bytes/sec), fast, no extra dependencies. Tradeoff: misses some edge cases
  that an ML approach might catch.
- **Standalone HTML visualization.** Zero deps, shareable, works offline.
  Tradeoff: less pretty than Plotly or TensorBoard.
- **Thread-safe tracker with one lock.** DataLoader workers run in threads;
  profiling must not crash training. Lock contention adds ~2 us per event;
  correctness wins.
- **Severity classification (LOW/MEDIUM/HIGH/CRITICAL).** Maps to how
  engineers triage. Tradeoff: slightly more complex than binary leak/no-leak.

## What I'd improve

- **Real PyTorch hook integration.** Auto-register allocations via
  `torch.cuda.memory` hooks or `torch.profiler` kineto callbacks, so users do
  not have to instrument manually. This is the fix for the current manual-API
  limitation.
- **Separate reserved-bytes tracking.** Make `take_snapshot` distinguish
  allocated from reserved, so the profiler can show the exact `nvidia-smi` vs
  `del tensor` gap the "Why this exists" section describes.
- **Cross-GPU tracking.** Multi-GPU runs allocate across devices; show which
  operations allocate on which GPU.
- **`torch.compile` awareness.** `torch.compile` fuses operations and changes
  allocation patterns; handle both eager and compiled modes.

## License

MIT
