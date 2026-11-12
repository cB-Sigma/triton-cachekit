"""A self-contained scenario used by the ``--demo`` flags and by the README.

It runs real Triton compilations (through the fake driver, so no GPU is needed):

1. *build*: prewarm four small kernels for ``cuda:80`` into a fresh cache.
2. *server*: run the same workload against that cache after three realistic
   changes - a called helper was edited, two comment lines were added above one
   kernel, and one kernel is called with a different block size.

One kernel still hits; the other three miss, each for a different reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .prewarm import Job, run_jobs
from .recording import Recording, load_recording

DEMO_TARGET = "cuda:80:32"

KERNELS_V1 = """\
import triton
import triton.language as tl

EPS = tl.constexpr(1e-6)


@triton.jit
def safe_div(num, den):
    return num / (den + EPS)


@triton.jit
def scale(x_ptr, out_ptr, n, factor, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * factor, mask=mask)


@triton.jit
def normalize(x_ptr, out_ptr, n, total, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, safe_div(x, total), mask=mask)


@triton.jit
def row_sum(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row, tl.sum(x, axis=0))


@triton.jit
def saxpy(x_ptr, y_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, alpha * x + y, mask=mask)
"""

# Same kernels after a "deploy": safe_div edited, two comment lines above saxpy.
KERNELS_V2 = KERNELS_V1.replace("return num / (den + EPS)", "return num / (den + 2 * EPS)").replace(
    "@triton.jit\ndef saxpy(",
    "# saxpy: out = alpha * x + y\n# (comment added in the new release)\n@triton.jit\ndef saxpy(",
)

_T = "tensor:float32"


def _jobs(kernel_file: Path, row_sum_block: int) -> list[Job]:
    def job(name: str, args: list[object], block: int = 128) -> Job:
        return Job(
            kernel={"file": str(kernel_file), "name": name},
            specs=[{"args": args, "kwargs": {"BLOCK": block}}],
        )

    return [
        job("scale", [_T, _T, 4096, 0.5]),
        job("normalize", [_T, _T, 4096, 10.0]),
        job("row_sum", [_T, _T, 96], row_sum_block),
        job("saxpy", [_T, _T, _T, 4096, 2.0]),
    ]


@dataclass
class Demo:
    workdir: Path
    cache_dir: Path
    build_jobs: list[Job]
    server_jobs: list[Job] | None
    build_recording: Recording
    server_recording: Recording | None


def run_demo(workdir: Path, *, server: bool = True, timeout: float = 300.0) -> Demo:
    """Run the scenario inside ``workdir`` (which should be empty)."""
    cache_dir = workdir / "cache"
    build_src = workdir / "build" / "demo_kernels.py"
    server_src = workdir / "server" / "demo_kernels.py"
    for path, text in ((build_src, KERNELS_V1), (server_src, KERNELS_V2)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    # Keep the demo independent of whatever the caller has exported.
    env = {"TRITON_DEFAULT_FP_FUSION": "0"}

    build_jobs = _jobs(build_src, 128)
    run_jobs(
        build_jobs,
        target=DEMO_TARGET,
        fake_driver=True,
        cache_dir=cache_dir,
        record_dir=workdir / "build.rec",
        timeout=timeout,
        extra_env=env,
    )
    _raise_on_error(build_jobs)
    build_recording = load_recording(workdir / "build.rec")
    server_jobs = None
    server_recording = None
    if server:
        server_jobs = _jobs(server_src, 256)
        run_jobs(
            server_jobs,
            target=DEMO_TARGET,
            fake_driver=True,
            cache_dir=cache_dir,
            record_dir=workdir / "server.rec",
            timeout=timeout,
            extra_env=env,
        )
        _raise_on_error(server_jobs)
        server_recording = load_recording(workdir / "server.rec")
    return Demo(workdir, cache_dir, build_jobs, server_jobs, build_recording, server_recording)


def _raise_on_error(jobs: list[Job]) -> None:
    for job in jobs:
        error = job.result.get("error") or next(
            (s.get("error") for s in job.result.get("specs", []) if s.get("error")), None
        )
        if error:
            detail = job.result.get("traceback", "")
            raise RuntimeError(f"demo compilation failed for {job.label}: {error}\n{detail}")
