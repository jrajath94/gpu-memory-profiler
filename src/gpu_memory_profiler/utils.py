"""Utility functions for memory profiling simulation and analysis."""

import logging
import random
import time
from typing import List, Optional, Tuple

from gpu_memory_profiler.models import (
    AllocationEvent,
    AllocationEventType,
    MemorySnapshot,
    MemoryTimeline,
    ProfileConfig,
)
from gpu_memory_profiler.tracker import AllocationTracker

logger = logging.getLogger(__name__)

# Common tensor sizes for simulation (bytes)
COMMON_TENSOR_SIZES = [
    4 * 768 * 768,          # Small attention matrix: 2.25 MB
    4 * 1024 * 1024,        # 1K x 1K float32: 4 MB
    4 * 2048 * 2048,        # 2K x 2K float32: 16 MB
    4 * 4096 * 4096,        # 4K x 4K float32: 64 MB
    4 * 1024 * 768 * 12,    # Multi-head attention: 36 MB
    4 * 50257 * 768,        # GPT-2 embedding: 148 MB
    2 * 1024 * 1024,        # 1K x 1K float16: 2 MB
]


def simulate_training_loop(
    tracker: AllocationTracker,
    num_iterations: int = 20,
    batch_size: int = 32,
    hidden_dim: int = 768,
    leak_probability: float = 0.0,
    device: str = "cpu",
) -> MemoryTimeline:
    """Simulate a training loop with realistic allocation patterns.

    Creates allocations mimicking forward pass, backward pass,
    and optimizer step with optional leak injection for testing.

    Args:
        tracker: Allocation tracker to record events.
        num_iterations: Number of training iterations.
        batch_size: Simulated batch size.
        hidden_dim: Simulated hidden dimension.
        leak_probability: Chance of a "leaked" allocation per iteration.
        device: Device label for allocations.

    Returns:
        Memory timeline from the simulated training.
    """
    timeline = MemoryTimeline(start_time=time.time(), device=device)
    ptr_counter = 0x1000_0000
    leaked_ptrs: List[int] = []

    for iteration in range(num_iterations):
        # Forward pass: allocate activations
        forward_ptrs = []
        for layer in range(6):
            size = 4 * batch_size * hidden_dim  # Activation tensor
            ptr = ptr_counter
            ptr_counter += size
            tracker.record_allocation(
                data_ptr=ptr,
                size_bytes=size,
                tensor_shape=(batch_size, hidden_dim),
                tensor_dtype="float32",
                device=device,
                tag=f"layer_{layer}_activation",
            )
            forward_ptrs.append((ptr, size))

        # Backward pass: allocate gradients, free activations
        grad_ptrs = []
        for layer in range(5, -1, -1):
            size = 4 * batch_size * hidden_dim
            ptr = ptr_counter
            ptr_counter += size
            tracker.record_allocation(
                data_ptr=ptr,
                size_bytes=size,
                tensor_shape=(batch_size, hidden_dim),
                tensor_dtype="float32",
                device=device,
                tag=f"layer_{layer}_gradient",
            )
            grad_ptrs.append((ptr, size))

        # Free activations after backward
        for ptr, size in forward_ptrs:
            tracker.record_free(ptr, size, device=device)

        # Optimizer step: update weights (small allocs)
        opt_ptrs = []
        for layer in range(6):
            size = 4 * hidden_dim * hidden_dim // 4  # Optimizer state
            ptr = ptr_counter
            ptr_counter += size
            tracker.record_allocation(
                data_ptr=ptr,
                size_bytes=size,
                tensor_shape=(hidden_dim, hidden_dim // 4),
                tensor_dtype="float32",
                device=device,
                tag=f"optimizer_state_{layer}",
            )
            opt_ptrs.append((ptr, size))

        # Free gradients and optimizer state
        for ptr, size in grad_ptrs:
            tracker.record_free(ptr, size, device=device)
        for ptr, size in opt_ptrs:
            tracker.record_free(ptr, size, device=device)

        # Inject leak: allocate but never free
        if leak_probability > 0 and random.random() < leak_probability:
            leak_size = random.choice(COMMON_TENSOR_SIZES)
            leak_ptr = ptr_counter
            ptr_counter += leak_size
            tracker.record_allocation(
                data_ptr=leak_ptr,
                size_bytes=leak_size,
                tensor_shape=(leak_size // 4,),
                tensor_dtype="float32",
                device=device,
                tag="LEAKED_TENSOR",
            )
            leaked_ptrs.append(leak_ptr)

        # Take snapshot after each iteration
        snapshot = tracker.take_snapshot(device=device)
        timeline.snapshots.append(snapshot)

    timeline.events = tracker.events

    if leaked_ptrs:
        logger.info(
            "Simulation injected %d leaked allocations", len(leaked_ptrs)
        )

    return timeline


def simulate_inference_batch(
    tracker: AllocationTracker,
    num_requests: int = 50,
    sequence_length: int = 512,
    hidden_dim: int = 768,
    device: str = "cpu",
) -> MemoryTimeline:
    """Simulate inference batching with KV cache allocations.

    Args:
        tracker: Allocation tracker to record events.
        num_requests: Number of inference requests.
        sequence_length: Tokens per request.
        hidden_dim: Model hidden dimension.
        device: Device label.

    Returns:
        Memory timeline from simulated inference.
    """
    timeline = MemoryTimeline(start_time=time.time(), device=device)
    ptr_counter = 0x2000_0000

    for req in range(num_requests):
        # KV cache allocation per request
        kv_size = 2 * 4 * sequence_length * hidden_dim  # K + V
        kv_ptr = ptr_counter
        ptr_counter += kv_size
        tracker.record_allocation(
            data_ptr=kv_ptr,
            size_bytes=kv_size,
            tensor_shape=(2, sequence_length, hidden_dim),
            tensor_dtype="float32",
            device=device,
            tag=f"kv_cache_req_{req}",
        )

        # Output logits
        vocab_size = 50257
        logit_size = 4 * vocab_size
        logit_ptr = ptr_counter
        ptr_counter += logit_size
        tracker.record_allocation(
            data_ptr=logit_ptr,
            size_bytes=logit_size,
            tensor_shape=(vocab_size,),
            tensor_dtype="float32",
            device=device,
            tag=f"logits_req_{req}",
        )

        # Free after request completes
        tracker.record_free(kv_ptr, kv_size, device=device)
        tracker.record_free(logit_ptr, logit_size, device=device)

        snapshot = tracker.take_snapshot(device=device)
        timeline.snapshots.append(snapshot)

    timeline.events = tracker.events
    return timeline


def format_bytes(size_bytes: int) -> str:
    """Format byte count as human-readable string.

    Args:
        size_bytes: Number of bytes.

    Returns:
        Formatted string (e.g., '1.5 GB', '256 MB').
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 ** 3):.2f} GB"


def estimate_model_memory(
    param_count: int,
    dtype_bytes: int = 4,
    optimizer: str = "adam",
    batch_size: int = 1,
    seq_length: int = 512,
    hidden_dim: int = 768,
) -> dict[str, int]:
    """Estimate GPU memory required for a model.

    Args:
        param_count: Number of model parameters.
        dtype_bytes: Bytes per parameter (4 for fp32, 2 for fp16).
        optimizer: Optimizer type ('adam', 'sgd', 'adamw').
        batch_size: Training batch size.
        seq_length: Sequence length.
        hidden_dim: Hidden dimension.

    Returns:
        Dictionary with memory breakdown in bytes.
    """
    # Model parameters
    param_memory = param_count * dtype_bytes

    # Gradients (same size as parameters)
    gradient_memory = param_memory

    # Optimizer states
    optimizer_multipliers = {
        "sgd": 0,       # No extra state
        "sgd_momentum": 1,  # Momentum buffer
        "adam": 2,       # First + second moment
        "adamw": 2,      # Same as Adam
    }
    opt_mult = optimizer_multipliers.get(optimizer, 2)
    optimizer_memory = param_count * dtype_bytes * opt_mult

    # Activations (rough estimate)
    num_layers = max(1, param_count // (hidden_dim * hidden_dim * 4))
    activation_memory = (
        batch_size * seq_length * hidden_dim * dtype_bytes * num_layers
    )

    return {
        "parameters": param_memory,
        "gradients": gradient_memory,
        "optimizer_states": optimizer_memory,
        "activations": activation_memory,
        "total": param_memory + gradient_memory + optimizer_memory + activation_memory,
    }
