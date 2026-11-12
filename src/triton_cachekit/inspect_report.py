"""Rendering for ``triton-cachekit inspect``."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from ._util import format_time, human_size, render_table, shorten
from .layout import CacheEntry, CacheScan

_STAGE_ABBREV = {
    "source": "src",
    "ttir": "ttir",
    "ttgir": "ttgir",
    "llir": "llir",
    "ptx": "ptx",
    "cubin": "cubin",
    "amdgcn": "amdgcn",
    "hsaco": "hsaco",
}


def _options_cell(entry: CacheEntry) -> str:
    opts = entry.summary_options()
    parts = []
    short = {
        "num_warps": "warps",
        "num_stages": "stages",
        "num_ctas": "ctas",
        "waves_per_eu": "waves",
        "enable_fp_fusion": "fpfuse",
    }
    for key, value in opts.items():
        if key == "num_ctas" and value == 1:
            continue
        if key == "waves_per_eu" and value == 0:
            continue
        if key == "debug" and not value:
            continue
        if isinstance(value, bool):
            value = int(value)
        parts.append(f"{short.get(key, key)}={value}")
    for name, value in sorted(entry.env.items()):
        parts.append(f"{name}={value}")
    return " ".join(parts) or "-"


def _stages_cell(entry: CacheEntry) -> str:
    stages = entry.stages
    if not stages:
        return "-"
    return ",".join(_STAGE_ABBREV[s] for s in stages)


def filter_entries(
    scan: CacheScan,
    *,
    kernel: str | None = None,
    target: str | None = None,
    only_problems: bool = False,
) -> list[CacheEntry]:
    import fnmatch

    selected = []
    for entry in scan.entries:
        if kernel and not (entry.name and fnmatch.fnmatchcase(entry.name, kernel)):
            continue
        if target and not matches_target(entry, target):
            continue
        if only_problems and entry.status == "ok" and not entry.problems:
            continue
        selected.append(entry)
    return selected


def matches_target(entry: CacheEntry, target: str) -> bool:
    """``target`` is ``backend`` or ``backend:arch`` (a third ``:warp`` field is ignored)."""
    parts = target.split(":")
    if entry.backend != parts[0]:
        return False
    return len(parts) < 2 or entry.arch == parts[1]


def totals(entries: list[CacheEntry]) -> dict[str, Any]:
    by_kernel: dict[str, dict[str, int]] = defaultdict(lambda: {"entries": 0, "size": 0})
    by_target: dict[str, dict[str, int]] = defaultdict(lambda: {"entries": 0, "size": 0})
    by_version: dict[str, dict[str, int]] = defaultdict(lambda: {"entries": 0, "size": 0})
    by_status: dict[str, int] = defaultdict(int)
    by_kind: dict[str, int] = defaultdict(int)
    for entry in entries:
        by_kind[entry.kind] += 1
        if entry.kind != "kernel":
            continue
        by_status[entry.status] += 1
        for table, label in (
            (by_kernel, entry.name or "?"),
            (by_target, entry.target),
            (by_version, entry.triton_version or "unknown"),
        ):
            table[label]["entries"] += 1
            table[label]["size"] += entry.size
    return {
        "entries": len(entries),
        "size": sum(e.size for e in entries),
        "files": sum(len(e.files) for e in entries),
        "by_kind": dict(sorted(by_kind.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_kernel": dict(sorted(by_kernel.items())),
        "by_target": dict(sorted(by_target.items())),
        "by_triton_version": dict(sorted(by_version.items())),
    }


def render_json(scan: CacheScan, entries: list[CacheEntry]) -> str:
    payload = {
        "cache_dir": str(scan.root),
        "totals": totals(entries),
        "entries": [e.to_dict() for e in entries],
        "stray_files": scan.stray,
    }
    return json.dumps(payload, indent=2, sort_keys=False, default=str)


def render_table_report(
    scan: CacheScan,
    entries: list[CacheEntry],
    *,
    wide: bool = False,
    show_location: bool = False,
) -> str:
    out: list[str] = []
    summary = totals(entries)
    out.append(
        f"Triton cache: {scan.root}  "
        f"({summary['entries']} entries, {summary['files']} files, {human_size(summary['size'])})"
    )
    out.append("")
    kernels = [e for e in entries if e.kind == "kernel"]
    others = [e for e in entries if e.kind != "kernel"]

    if kernels:
        headers = [
            "KEY",
            "KERNEL",
            "TARGET",
            "TRITON",
            "OPTIONS",
            "STAGES",
            "SIZE",
            "MODIFIED",
            "STATUS",
        ]
        rows = []
        for entry in sorted(kernels, key=lambda e: (e.name or "", e.target, e.mtime or 0, e.key)):
            rows.append(
                [
                    entry.key if wide else entry.key[:10],
                    entry.name or "?",
                    entry.target,
                    entry.triton_version or "?",
                    _options_cell(entry) if wide else shorten(_options_cell(entry), 44),
                    _stages_cell(entry),
                    human_size(entry.size),
                    format_time(entry.mtime),
                    entry.status,
                ]
            )
        out.append(render_table(headers, rows, right_align=(6,)))
        out.append("")
        if show_location:
            out.append("Kernel source locations (from MLIR debug info):")
            loc_rows = [
                [
                    e.key[:10],
                    e.name or "?",
                    f"{e.source_file}:{e.source_line}" if e.source_file else "-",
                ]
                for e in sorted(kernels, key=lambda e: (e.name or "", e.key))
            ]
            out.append(render_table(["KEY", "KERNEL", "DEFINED AT"], loc_rows))
            out.append("")

        out.append("By kernel:")
        out.append(
            render_table(
                ["KERNEL", "ENTRIES", "SIZE"],
                [
                    [k, str(v["entries"]), human_size(v["size"])]
                    for k, v in summary["by_kernel"].items()
                ],
                right_align=(1, 2),
            )
        )
        out.append("")
        out.append("By target:")
        out.append(
            render_table(
                ["TARGET", "ENTRIES", "SIZE"],
                [
                    [k, str(v["entries"]), human_size(v["size"])]
                    for k, v in summary["by_target"].items()
                ],
                right_align=(1, 2),
            )
        )
        out.append("")
        if len(summary["by_triton_version"]) > 1:
            out.append("By Triton version:")
            out.append(
                render_table(
                    ["TRITON", "ENTRIES", "SIZE"],
                    [
                        [k, str(v["entries"]), human_size(v["size"])]
                        for k, v in summary["by_triton_version"].items()
                    ],
                    right_align=(1, 2),
                )
            )
            out.append("")

    if others:
        out.append("Other entries:")
        out.append(
            render_table(
                ["KEY", "KIND", "NAME", "SIZE", "MODIFIED"],
                [
                    [
                        e.key if wide else e.key[:10],
                        e.kind,
                        e.name or "-",
                        human_size(e.size),
                        format_time(e.mtime),
                    ]
                    for e in others
                ],
                right_align=(3,),
            )
        )
        out.append("")

    problem_entries = [e for e in entries if e.problems]
    if problem_entries:
        out.append(f"Problems ({len(problem_entries)} entries):")
        for entry in problem_entries:
            out.append(f"  {entry.key[:10]}  {entry.name or entry.kind}  [{entry.status}]")
            for problem in entry.problems:
                out.append(f"    - {problem.severity}: {problem.message}")
        out.append("")
    elif kernels:
        out.append("No incomplete or corrupt entries found.")
        out.append("")

    if scan.stray:
        out.append(
            "Stray files in the cache root (ignored by Triton): " + ", ".join(scan.stray[:10])
        )
        out.append("")
    if not entries:
        out.append("(no cache entries)")
        out.append("")
    return "\n".join(out).rstrip() + "\n"
