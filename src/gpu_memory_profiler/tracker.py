"""Allocation tracker that records individual tensor allocation/free events."""

import logging
import threading
import time
import traceback
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from gpu_memory_profiler.models import (
    AllocationEvent,
    AllocationEventType,
    MemorySnapshot,
    ProfileConfig,
)

logger = logging.getLogger(__name__)

# Allocation size thresholds for filtering noise
MIN_TRACKED_BYTES = 512


class AllocationTracker:
    """Tracks GPU memory allocations and frees at the tensor level.

    Uses a ring buffer to store allocation events and maintains a live
    allocation table mapping tensor data pointers to their metadata.

    Args:
        config: Profiling configuration.
    """

    def __init__(self, config: Optional[ProfileConfig] = None) -> None:
        self._config = config or ProfileConfig()
        self._events: Deque[AllocationEvent] = deque(
            maxlen=self._config.max_events
        )
        self._live_allocations: Dict[int, AllocationEvent] = {}
        self._lock = threading.Lock()
        self._allocation_count = 0
        self._free_count = 0
        self._total_allocated = 0
        self._total_freed = 0
        self._peak_allocated = 0
        self._current_allocated = 0

    def record_allocation(
        self,
        data_ptr: int,
        size_bytes: int,
        tensor_shape: Optional[Tuple[int, ...]] = None,
        tensor_dtype: Optional[str] = None,
        device: str = "cpu",
        tag: Optional[str] = None,
    ) -> AllocationEvent:
        """Record a new tensor allocation.

        Args:
            data_ptr: Memory address of the allocation.
            size_bytes: Size of allocation in bytes.
            tensor_shape: Shape of the allocated tensor.
            tensor_dtype: Data type of the tensor.
            device: Device where allocation occurred.
            tag: Optional label for this allocation.

        Returns:
            The recorded allocation event.
        """
        stack_trace = None
        if self._config.capture_stack_traces and size_bytes >= MIN_TRACKED_BYTES:
            stack_trace = _capture_stack_trace(self._config.stack_trace_depth)

        event = AllocationEvent(
            event_type=AllocationEventType.ALLOCATE,
            size_bytes=size_bytes,
            timestamp=time.time(),
            tensor_shape=tensor_shape,
            tensor_dtype=tensor_dtype,
            device=device,
            stack_trace=stack_trace,
            tag=tag,
        )

        with self._lock:
            self._events.append(event)
            self._live_allocations[data_ptr] = event
            self._allocation_count += 1
            self._total_allocated += size_bytes
            self._current_allocated += size_bytes
            if self._current_allocated > self._peak_allocated:
                self._peak_allocated = self._current_allocated

        return event

    def record_free(
        self,
        data_ptr: int,
        size_bytes: int,
        device: str = "cpu",
    ) -> AllocationEvent:
        """Record a tensor deallocation.

        Args:
            data_ptr: Memory address being freed.
            size_bytes: Size of the freed allocation.
            device: Device where free occurred.

        Returns:
            The recorded free event.
        """
        event = AllocationEvent(
            event_type=AllocationEventType.FREE,
            size_bytes=size_bytes,
            timestamp=time.time(),
            device=device,
        )

        with self._lock:
            self._events.append(event)
            self._live_allocations.pop(data_ptr, None)
            self._free_count += 1
            self._total_freed += size_bytes
            self._current_allocated = max(
                0, self._current_allocated - size_bytes
            )

        return event

    def record_oom(
        self,
        requested_bytes: int,
        device: str = "cpu",
    ) -> AllocationEvent:
        """Record an out-of-memory event.

        Args:
            requested_bytes: Size of the failed allocation.
            device: Device where OOM occurred.

        Returns:
            The recorded OOM event.
        """
        stack_trace = _capture_stack_trace(self._config.stack_trace_depth)

        event = AllocationEvent(
            event_type=AllocationEventType.OOM,
            size_bytes=requested_bytes,
            timestamp=time.time(),
            device=device,
            stack_trace=stack_trace,
        )

        with self._lock:
            self._events.append(event)

        logger.warning(
            "OOM event: requested %d bytes on %s",
            requested_bytes,
            device,
        )
        return event

    def take_snapshot(self, device: str = "cpu") -> MemorySnapshot:
        """Capture a point-in-time memory snapshot.

        Args:
            device: Device to snapshot.

        Returns:
            Current memory state as a snapshot.
        """
        with self._lock:
            return MemorySnapshot(
                timestamp=time.time(),
                allocated_bytes=self._current_allocated,
                reserved_bytes=self._current_allocated,
                active_tensors=len(self._live_allocations),
                device=device,
                peak_allocated_bytes=self._peak_allocated,
                allocation_count=self._allocation_count,
                free_count=self._free_count,
            )

    @property
    def events(self) -> List[AllocationEvent]:
        """Return a copy of all recorded events."""
        with self._lock:
            return list(self._events)

    @property
    def live_allocations(self) -> Dict[int, AllocationEvent]:
        """Return a copy of currently live allocations."""
        with self._lock:
            return dict(self._live_allocations)

    @property
    def current_allocated_bytes(self) -> int:
        """Current total allocated bytes."""
        with self._lock:
            return self._current_allocated

    @property
    def peak_allocated_bytes(self) -> int:
        """Peak allocated bytes seen."""
        with self._lock:
            return self._peak_allocated

    @property
    def allocation_count(self) -> int:
        """Total allocation count."""
        with self._lock:
            return self._allocation_count

    @property
    def free_count(self) -> int:
        """Total free count."""
        with self._lock:
            return self._free_count

    def reset(self) -> None:
        """Clear all tracked state."""
        with self._lock:
            self._events.clear()
            self._live_allocations.clear()
            self._allocation_count = 0
            self._free_count = 0
            self._total_allocated = 0
            self._total_freed = 0
            self._peak_allocated = 0
            self._current_allocated = 0

    def top_allocations(self, n: int = 10) -> List[AllocationEvent]:
        """Return the N largest live allocations.

        Args:
            n: Number of allocations to return.

        Returns:
            List of allocation events sorted by size descending.
        """
        with self._lock:
            allocations = list(self._live_allocations.values())
        allocations.sort(key=lambda e: e.size_bytes, reverse=True)
        return allocations[:n]


def _capture_stack_trace(depth: int) -> str:
    """Capture a filtered Python stack trace.

    Filters out internal profiler frames to show only user code.

    Args:
        depth: Maximum number of stack frames to capture.

    Returns:
        Formatted stack trace string.
    """
    frames = traceback.extract_stack()
    # Filter out profiler internals
    filtered = [
        f
        for f in frames
        if "gpu_memory_profiler" not in f.filename
        and "traceback" not in f.filename
    ]
    # Take the most recent frames up to depth
    relevant = filtered[-depth:] if len(filtered) > depth else filtered
    return "".join(traceback.format_list(relevant))
