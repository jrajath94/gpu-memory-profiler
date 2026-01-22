"""Custom exception types for GPU Memory Profiler."""


class GPUMemoryProfilerError(Exception):
    """Base exception for all GPU memory profiler errors."""


class CUDANotAvailableError(GPUMemoryProfilerError):
    """Raised when CUDA is not available but GPU profiling is requested."""

    def __init__(self) -> None:
        super().__init__(
            "CUDA is not available. GPU memory profiling requires a CUDA-capable device. "
            "Use cpu_fallback=True for CPU tensor tracking."
        )


class ProfilerNotStartedError(GPUMemoryProfilerError):
    """Raised when operations require an active profiler that hasn't been started."""

    def __init__(self) -> None:
        super().__init__(
            "Profiler has not been started. Call profiler.start() or use as context manager."
        )


class ProfilerAlreadyStartedError(GPUMemoryProfilerError):
    """Raised when attempting to start a profiler that's already running."""

    def __init__(self) -> None:
        super().__init__(
            "Profiler is already running. Call profiler.stop() before starting again."
        )


class InvalidSnapshotError(GPUMemoryProfilerError):
    """Raised when a memory snapshot is invalid or corrupted."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Invalid memory snapshot: {reason}")


class VisualizationError(GPUMemoryProfilerError):
    """Raised when visualization generation fails."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Visualization error: {reason}")
