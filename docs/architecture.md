# Architecture: gpu-memory-profiler

## Component Diagram

```
User Code
    │
    ▼
┌──────────────────────────────────┐
│         MemoryProfiler           │
│  - Context manager interface     │
│  - Orchestrates all components   │
└──────────┬───────────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────────┐  ┌──────────────────┐
│ Allocation │  │   Leak Detector  │
│  Tracker   │  │  - Growth rate   │
│            │  │  - Imbalance     │
│ Ring buffer│  │  - Persistent    │
│ Thread-safe│  └────────┬─────────┘
└─────┬──────┘           │
      │                  │
      ▼                  ▼
┌─────────────┐   ┌──────────────┐
│   Memory    │   │    Leak      │
│  Timeline   │   │  Candidates  │
│             │   │              │
│ snapshots[] │   │ severity     │
│ events[]    │   │ growth_rate  │
└──────┬──────┘   └──────┬───────┘
       │                 │
       └────────┬────────┘
                ▼
       ┌────────────────┐
       │  Visualizer    │
       │                │
       │ timeline.html  │
       │ flamegraph.html│
       │ leaks.html     │
       └────────────────┘
```

## Leak Detection Pipeline

```
Snapshots (time series)
    │
    ▼
┌──────────────────────────────┐
│  Heuristic 1: Growth Rate   │
│  Linear regression on        │
│  allocated_bytes vs time     │
│  Flag if slope > threshold   │
│  AND R² > 0.7               │
├──────────────────────────────┤
│  Heuristic 2: Imbalance     │
│  allocs vs frees ratio       │
│  Flag if > 20% unfreed      │
├──────────────────────────────┤
│  Heuristic 3: Persistent    │
│  Large allocs never freed    │
│  Flag if > threshold size    │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│   Severity Classification    │
│                              │
│   < 100 KB/s  → LOW         │
│   < 1 MB/s   → MEDIUM      │
│   < 10 MB/s  → HIGH        │
│   ≥ 100 MB/s → CRITICAL    │
└──────────────────────────────┘
```

## Thread Safety

The AllocationTracker uses a threading.Lock to protect all mutable state.
This is necessary because PyTorch DataLoader workers run in separate threads
and may trigger allocations concurrently with the main training loop.

## Ring Buffer Design

Events are stored in a `collections.deque(maxlen=N)` to bound memory usage.
When the buffer is full, the oldest events are automatically discarded.
This prevents the profiler itself from causing OOM during long profiling sessions.
