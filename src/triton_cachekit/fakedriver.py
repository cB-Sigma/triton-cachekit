"""A stand-in GPU driver so Triton can compile for a GPU that is not there.

Triton asks ``triton.runtime.driver.driver.active`` for three things before it
compiles a kernel: the current device, the current stream and the current target.
None of them needs hardware once the target is known, so a driver object that
returns constants is enough to run the whole compilation pipeline (the bundled
``ptxas`` and the AMD code generator run on the CPU).

What this does *not* give you: anything that touches a device. Launching a kernel
through this driver fails by design.
"""

from __future__ import annotations

from typing import Any

from ._util import parse_target

#: Warp size the real CUDA driver always reports (``CudaDriver.get_current_target``).
CUDA_WARP_SIZE = 32


class FakeDriverError(RuntimeError):
    pass


def make_target(spec: str) -> Any:
    """Build a ``GPUTarget`` from ``backend:arch[:warp_size]``."""
    from triton.backends.compiler import GPUTarget

    backend, arch, warp = parse_target(spec)
    if backend == "cuda":
        if not arch.isdigit():
            raise FakeDriverError(
                f"invalid CUDA target {spec!r}: the architecture is the compute capability "
                "as an integer, e.g. 80 (A100), 89 (L4/RTX 40), 90 (H100), 100 (B200)"
            )
        return GPUTarget("cuda", int(arch), warp if warp is not None else CUDA_WARP_SIZE)
    if warp is None:
        raise FakeDriverError(
            f"invalid HIP target {spec!r}: give the warp size explicitly "
            "(the real driver reads it from the device), e.g. 'hip:gfx942:64'"
        )
    return GPUTarget("hip", arch, warp)


def format_target(target: Any) -> str:
    return f"{target.backend}:{target.arch}:{target.warp_size}"


def install_fake_driver(spec: str) -> Any:
    """Make Triton believe a device with the given target is current. Returns the target."""
    from triton.backends.driver import DriverBase
    from triton.runtime.driver import driver

    target = make_target(spec)

    class FakeDriver(DriverBase):  # type: ignore[misc]
        """Reports a fixed target; cannot launch kernels."""

        @classmethod
        def is_active(cls) -> bool:
            return True

        def map_python_to_cpp_type(self, ty: str) -> str:
            raise FakeDriverError("the fake driver cannot build launchers")

        def get_current_target(self) -> Any:
            return target

        def get_current_device(self) -> int:
            return 0

        def get_current_stream(self, device: int) -> int:
            return 0

        def set_current_device(self, device: int) -> None:
            return None

        def get_active_torch_device(self) -> Any:
            raise FakeDriverError("the fake driver has no device")

        def get_benchmarker(self) -> Any:
            raise FakeDriverError("the fake driver cannot benchmark (autotuning needs a GPU)")

    driver.set_active(FakeDriver())
    return target


class _DType:
    """Mimics a ``torch.dtype`` just enough for Triton's type canonicalisation."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return f"torch.{self.name}"

    __repr__ = __str__

    def __hash__(self) -> int:
        return hash(("triton_cachekit.dtype", self.name))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _DType) and other.name == self.name


class PrewarmTensor:
    """A tensor argument for ``kernel.warmup``: a dtype and a pointer alignment.

    Triton's specialisation looks at exactly two things on a tensor argument: its
    ``dtype`` and whether ``data_ptr()`` is divisible by 16.
    """

    def __init__(self, dtype: str, *, aligned: bool = True) -> None:
        self.dtype = _DType(dtype)
        self.aligned = aligned
        self.shape = [1]

    def data_ptr(self) -> int:
        return 0 if self.aligned else 4

    def stride(self) -> tuple[int, ...]:
        return (1,)

    def __repr__(self) -> str:
        return f"tensor:{self.dtype.name}" + ("" if self.aligned else ":unaligned")


def parse_manifest_arg(value: Any) -> Any:
    """Turn a manifest argument into the object passed to ``kernel.warmup``.

    * ``"tensor:<dtype>"`` / ``"tensor:<dtype>:unaligned"`` -> :class:`PrewarmTensor`
    * ``{"tensor": "<dtype>", "aligned": false}``           -> :class:`PrewarmTensor`
    * ``{"value": x}``                                        -> ``x`` (escape hatch)
    * lists                                                   -> tuples
    * anything else (int, float, bool, str)                   -> itself
    """
    if isinstance(value, str) and value.startswith("tensor:"):
        parts = value.split(":")
        if len(parts) == 2:
            return PrewarmTensor(parts[1])
        if len(parts) == 3 and parts[2] in ("aligned", "unaligned"):
            return PrewarmTensor(parts[1], aligned=parts[2] == "aligned")
        raise ValueError(f"invalid tensor argument {value!r}: use 'tensor:<dtype>[:unaligned]'")
    if isinstance(value, dict):
        if "tensor" in value:
            return PrewarmTensor(str(value["tensor"]), aligned=bool(value.get("aligned", True)))
        if set(value) == {"value"}:
            return value["value"]
        raise ValueError(f"invalid argument table {value!r}: expected 'tensor' or 'value'")
    if isinstance(value, list):
        return tuple(parse_manifest_arg(v) for v in value)
    return value
