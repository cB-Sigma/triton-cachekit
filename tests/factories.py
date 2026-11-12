"""Builders for synthetic recordings (the shape the recorder writes)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from triton_cachekit._util import base32_key

PTXAS = "ptxas: NVIDIA (R) Ptx optimizing assembler\nCuda compilation tools, release 12.9, V12.9.86\n-80"


def session(
    pid: int = 100, *, version: str = "3.9.0", policy: tuple[str, ...] = ("line",), **over: Any
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "type": "session",
        "schema": 1,
        "pid": pid,
        "seq": 1,
        "triton": {
            "version": version,
            "key_digest": "t" * 64,
            "files": {"runtime/cache.py": "a" * 64, "_C/libtriton.so": "b" * 64},
            "files_verified": True,
        },
        "location_in_key": list(policy),
        "env": {"cache_invalidating": {}, "triton": {}},
        "implementation": "cpython-313",
    }
    record.update(over)
    return record


def fn(name: str = "pkg.kernels.blend", **over: Any) -> dict[str, Any]:
    src = over.pop("src", "def blend(x):\n    return x\n")
    info: dict[str, Any] = {
        "name": name,
        "file": "/app/pkg/kernels.py",
        "line": 10,
        "def_line": 11,
        "def_col": 1,
        "noinline": False,
        "src": src,
        "src_sha256": hashlib.sha256(src.encode()).hexdigest(),
        "constexpr_globals": [],
        "deps": [],
    }
    info.update(over)
    info["cache_key"] = _digest({k: v for k, v in info.items() if k != "cache_key"})
    return info


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def compile_record(pid: int = 100, **over: Any) -> dict[str, Any]:
    """A ``compile`` record; keyword overrides use dotted paths into ``components``."""
    function = over.pop("fn", None) or fn()
    components: dict[str, Any] = {
        "triton": {"version": "3.9.0", "key_digest": "t" * 64},
        "source": {
            "kind": "ast",
            "fn": function,
            "signature": {"x": "*fp32", "n": "i32", "BLOCK": "constexpr"},
            "constants": [{"path": [2], "name": "BLOCK", "value": "64", "key": "64"}],
            "attrs": "{(0,): [['tt.divisibility', 16]]}",
        },
        "backend": {
            "hash": PTXAS,
            "class": "CUDABackend",
            "target": {"backend": "cuda", "arch": 80, "warp_size": 32},
        },
        "options": {
            "hash": None,
            "values": {"num_warps": 4, "num_stages": 3, "enable_fp_fusion": False},
            "extern_libs": {"libdevice": {"path": "/t/libdevice.10.bc", "sha256": "d" * 64}},
        },
        "env": {},
    }
    record: dict[str, Any] = {
        "type": "compile",
        "pid": pid,
        "kernel": function["name"],
        "name": function["name"].rpartition(".")[2],
        "cache_dir": "/cache",
        "custom_pipeline": False,
        "via_jit": True,
        "specialization_data": None,
        "components": components,
    }
    for dotted, value in over.items():
        target: Any = record if dotted in record else components
        parts = dotted.split("__")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    source = components["source"]
    source["hash"] = _digest(
        [source["fn"]["cache_key"], source["attrs"], source["signature"], source["constants"]]
    )
    components["options"]["hash"] = _digest(
        [components["options"]["values"], components["options"]["extern_libs"]]
    )
    record["hash"] = hashlib.sha256(
        _digest(
            [
                components["triton"]["key_digest"],
                source["hash"],
                components["backend"]["hash"],
                components["options"]["hash"],
                sorted(components["env"].items()),
                record["custom_pipeline"],
            ]
        ).encode()
    ).hexdigest()
    record["entry"] = base32_key(record["hash"])
    return record


def outcome(
    compile_rec: dict[str, Any],
    *,
    hit: bool,
    total: int = 50_000,
    ir: str | None = None,
    ir_no_loc: str | None = None,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    if ir is not None:
        artifacts["source"] = {"size": 1, "sha256": ir, "sha256_no_loc": ir_no_loc or ir}
    lowering = (
        {}
        if hit
        else {
            "ttir": total // 10,
            "ttgir": total // 5,
            "llir": total // 5,
            "ptx": total // 10,
            "cubin": total // 5,
        }
    )
    return {
        "type": "outcome",
        "pid": compile_rec["pid"],
        "hash": compile_rec["hash"],
        "kernel": compile_rec["kernel"],
        "cache_hit": hit,
        "times_us": {
            "total": total,
            "ir_initialization": total - sum(lowering.values()),
            "lowering": lowering,
            "store_results": 0,
        },
        "artifacts": artifacts,
    }


def write_recording(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for index, record in enumerate(records, 1):
            handle.write(json.dumps({"seq": index, **copy.deepcopy(record)}) + "\n")
    return path


def recording(
    path: Path, compiles: list[tuple[dict[str, Any], bool | None]], **session_over: Any
) -> Path:
    """Write a session plus ``(compile, hit)`` pairs; ``hit=None`` omits the outcome."""
    records: list[dict[str, Any]] = [session(**session_over)]
    for rec, hit in compiles:
        records.append(rec)
        if hit is not None:
            records.append(outcome(rec, hit=hit, total=5_000 if hit else 50_000))
    return write_recording(path, records)
