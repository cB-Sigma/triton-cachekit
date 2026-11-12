"""Shared fixtures.

Tests that need a working Triton are marked ``triton`` and skipped when it cannot
be imported. Everything they compile goes through the fake driver, so no GPU is
needed; nothing is written outside pytest's temporary directories.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
HAS_TRITON = importlib.util.find_spec("triton") is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if HAS_TRITON:
        return
    skip = pytest.mark.skip(reason="Triton is not installed")
    for item in items:
        if "triton" in item.keywords:
            item.add_marker(skip)


def _cli_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    paths = [str(SRC)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in list(env):
        if name.startswith("TRITON_") or name == "PTXAS_OPTIONS":
            del env[name]
    env.update(extra or {})
    return env


@pytest.fixture
def cli() -> Callable[..., subprocess.CompletedProcess[str]]:
    """Run ``python -m triton_cachekit`` in a subprocess."""

    def run(
        *args: str, env: dict[str, str] | None = None, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "triton_cachekit", *map(str, args)],
            capture_output=True,
            text=True,
            env=_cli_env(env),
            cwd=cwd,
            timeout=600,
            check=False,
        )

    return run


# --------------------------------------------------------------- synthetic caches
KEY_A = "FDFR7LPCZJFRUESSI3JYNFXHVZDCWYNXSU76DLOVGYOT3BVEIZ3A"
HASH_A = "28cb1fade2ca4b1a125246d38696e7ae462b61b7953fe1add5361d3d86a44676"
KEY_B = "65SSM42BE5Z3UIWADJGE6TMZLWWMMDLBCM3HCA3MDTU6ZHKZWRLA"
HASH_B = "f7652673412773ba22c01a4c4f4d995dacc60d61133671036c1ce9ec9d59b456"


def make_entry(
    cache: Path,
    key: str = KEY_A,
    digest: str = HASH_A,
    name: str = "axpy",
    *,
    backend: str = "cuda",
    arch: Any = 80,
    group: str = "absolute",
    triton_version: str | None = "3.9.0",
    env: dict[str, str] | None = None,
    group_root: Path | None = None,
    source_path: str = "/src/pkg/kernels.py",
) -> Path:
    """Write a small text-only entry in the same shape Triton writes."""
    directory = cache / key
    directory.mkdir(parents=True, exist_ok=True)
    exts = ["source", "ttir", "ttgir", "llir"] + (
        ["ptx", "cubin"] if backend == "cuda" else ["amdgcn", "hsaco"]
    )
    for ext in exts:
        body = f'#loc = loc("{source_path}":11:1)\nmodule {{ }} // {ext}\n'
        (directory / f"{name}.{ext}").write_text(body)
    meta: dict[str, Any] = {
        "hash": digest,
        "target": {"backend": backend, "arch": arch, "warp_size": 32 if backend == "cuda" else 64},
        "num_warps": 4,
        "num_stages": 3,
        "num_ctas": 1,
        "enable_fp_fusion": True,
        "debug": False,
        "name": name,
        "shared": 0,
        **(env or {}),
    }
    if triton_version is not None:
        meta["triton_version"] = triton_version
    (directory / f"{name}.json").write_text(json.dumps(meta))
    children = [f"{name}.{ext}" for ext in exts] + [f"{name}.json"]
    root = (group_root or cache) / key
    if group == "absolute":
        payload: Any = {c: str(root / c) for c in children}
    elif group == "relative":
        payload = sorted(children)
    else:
        payload = None
    if payload is not None:
        (directory / f"__grp__{name}.json").write_text(json.dumps({"child_paths": payload}))
    return directory


@pytest.fixture
def synthetic_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    make_entry(cache, KEY_A, HASH_A, "axpy")
    make_entry(cache, KEY_B, HASH_B, "axpy", backend="hip", arch="gfx942", group="relative")
    return cache


# ------------------------------------------------------------------ real Triton
@pytest.fixture(scope="session")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """The built-in scenario, compiled once per test session with the real Triton."""
    from triton_cachekit.demo import run_demo

    return run_demo(tmp_path_factory.mktemp("demo"))


KERNEL_SOURCE = """\
import triton
import triton.language as tl

GAIN = tl.constexpr(3)


@triton.jit
def boost(x):
    return x * GAIN


@triton.jit
def blend(a_ptr, b_ptr, out_ptr, n, weight, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, boost(a) * weight + b, mask=mask)
"""

BLEND_ARGS = ["tensor:float32", "tensor:float32", "tensor:float32", 1024, 0.25]


def compile_once(
    source_file: Path,
    cache_dir: Path,
    record_dir: Path,
    *,
    name: str = "blend",
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    target: str = "cuda:80:32",
    env: dict[str, str] | None = None,
    source: str | None = KERNEL_SOURCE,
) -> Any:
    """Compile one kernel in a fresh subprocess (fake driver) and record it.

    Returns ``(job, recording)``.
    """
    from triton_cachekit.prewarm import Job, run_jobs
    from triton_cachekit.recording import load_recording

    if source is not None:
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text(source)
    job = Job(
        kernel={"file": str(source_file), "name": name},
        specs=[
            {"args": list(BLEND_ARGS if args is None else args), "kwargs": kwargs or {"BLOCK": 64}}
        ],
    )
    base_env = {"TRITON_DEFAULT_FP_FUSION": "0"}
    base_env.update(env or {})
    saved = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("TRITON_")}
    try:
        run_jobs(
            [job],
            target=target,
            fake_driver=True,
            cache_dir=cache_dir,
            record_dir=record_dir,
            timeout=300,
            extra_env=base_env,
        )
    finally:
        os.environ.update(saved)
    assert not job.result.get("error"), job.result
    assert job.result["specs"][0]["status"] != "error", job.result
    return job, load_recording(record_dir)
