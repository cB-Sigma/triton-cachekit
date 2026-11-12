"""Subprocess entry point used by ``prewarm`` and ``verify-hit``.

Reads one JSON job from stdin, compiles the requested specialisations of one
kernel through Triton's own JIT entry points (``kernel.warmup`` for manifest
specs, ``kernel.preload`` for specs taken from a recording) and prints one JSON
result line prefixed with :data:`RESULT_MARKER`.

Running in a subprocess gives each kernel a fresh interpreter (no shared JIT
state, a crash or hang in the compiler only loses one job).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import time
import traceback
from typing import Any

RESULT_MARKER = "@@triton-cachekit-result@@ "


def _load_kernel(kernel: dict[str, Any]) -> Any:
    name = str(kernel["name"])
    module_name = kernel.get("module")
    file = kernel.get("file")
    module = None
    if module_name and module_name != "__main__":
        try:
            module = importlib.import_module(str(module_name))
        except ImportError:
            if not file:
                raise
    if module is None:
        if not file:
            raise ImportError(f"kernel {name!r}: neither an importable module nor a file was given")
        path = os.path.abspath(str(file))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"kernel file not found: {path}")
        alias = str(module_name) if module_name and module_name != "__main__" else None
        alias = alias or os.path.splitext(os.path.basename(path))[0]
        if alias == "__main__":
            alias = "__tck_main__"
        # Like Triton's own tools/compile.py: the file's directory becomes importable.
        sys.path.insert(0, os.path.dirname(path))
        spec = importlib.util.spec_from_file_location(alias, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    obj: Any = module
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def _unwrap_jit_function(obj: Any) -> Any:
    """Autotuner and similar wrappers keep the JITFunction in ``.fn``."""
    seen = 0
    while not hasattr(obj, "preload") and hasattr(obj, "fn") and seen < 8:
        obj = obj.fn
        seen += 1
    return obj


def _run(job: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"specs": [], "error": None}
    events: list[dict[str, Any]] = []

    import triton
    from triton import knobs

    from . import recorder as recorder_module
    from .fakedriver import format_target, install_fake_driver, parse_manifest_arg

    result["triton_version"] = str(triton.__version__)

    if job.get("fake_driver"):
        target = install_fake_driver(str(job["target"]))
        result["target"] = format_target(target)
        result["driver"] = "fake"
    else:
        from triton.runtime.driver import driver

        try:
            target = driver.active.get_current_target()
        except Exception as exc:
            result["error"] = (
                "no usable GPU driver in this process "
                f"({type(exc).__name__}: {exc}). verify-hit must run on the GPU host; "
                "use --fake-driver only to exercise the tool itself."
            )
            return result
        result["target"] = format_target(target)
        result["driver"] = type(driver.active).__name__
        wanted = job.get("target")
        if wanted and not format_target(target).startswith(str(wanted)):
            result["target_mismatch"] = {"expected": wanted, "actual": format_target(target)}

    def listener(
        *, src: Any, metadata: Any, metadata_group: Any, times: Any, cache_hit: bool
    ) -> None:
        events.append(
            {
                "hash": str(dict(metadata).get("hash")),
                "cache_hit": bool(cache_hit),
                "us": int(times.total),
            }
        )

    knobs.compilation.listener = listener
    if job.get("record_to"):
        recorder_module.install(
            job["record_to"], record_source=bool(job.get("record_source", True))
        )
    result["fingerprint"] = recorder_module.collect_fingerprint()

    try:
        loaded = _load_kernel(job["kernel"])
    except BaseException as exc:
        result["error"] = f"cannot load kernel: {type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=8)
        return result

    for index, spec in enumerate(job["specs"]):
        before = len(events)
        started = time.perf_counter()
        outcome: dict[str, Any] = {"index": index, "status": "error", "compilations": []}
        try:
            if "specialization_data" in spec:
                kernel = _unwrap_jit_function(loaded)
                data = json.loads(spec["specialization_data"])
                # The qualified name is not part of the key; it only has to match the
                # object we loaded (which may live under another module name here).
                data["name"] = kernel._fn_name
                kernel.preload(json.dumps(data))
            else:
                args = [parse_manifest_arg(a) for a in spec.get("args", [])]
                kwargs = {
                    str(k): parse_manifest_arg(v) for k, v in dict(spec.get("kwargs", {})).items()
                }
                loaded.warmup(*args, grid=(1,), **kwargs)
            new = events[before:]
            outcome["compilations"] = new
            if not new:
                outcome["status"] = "duplicate"
            elif all(e["cache_hit"] for e in new):
                outcome["status"] = "hit"
            elif not any(e["cache_hit"] for e in new):
                outcome["status"] = "compiled"
            else:
                outcome["status"] = "partial"
        except BaseException as exc:
            outcome["status"] = "error"
            outcome["error"] = f"{type(exc).__name__}: {exc}"
            outcome["compilations"] = events[before:]
        outcome["seconds"] = round(time.perf_counter() - started, 4)
        result["specs"].append(outcome)
    return result


def main() -> int:
    job = json.loads(sys.stdin.read())
    try:
        result = _run(job)
    except BaseException as exc:
        result = {
            "specs": [],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=12),
        }
    sys.stdout.flush()
    print(RESULT_MARKER + json.dumps(result, default=repr), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
