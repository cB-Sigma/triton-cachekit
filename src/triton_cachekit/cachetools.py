"""``diff`` and ``prune`` for cache directories."""

from __future__ import annotations

import fnmatch
import shutil
import time
from dataclasses import dataclass, field
from typing import Any

from ._util import format_time, human_size, render_table, sha256_file
from .inspect_report import matches_target
from .layout import CacheEntry, CacheScan
from .pack import remove_entry_files


# ------------------------------------------------------------------------ diff
@dataclass
class CacheDiff:
    only_a: list[CacheEntry]
    only_b: list[CacheEntry]
    common: int
    content_differs: list[tuple[CacheEntry, CacheEntry, list[str]]]
    hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "only_a": [e.to_dict() for e in self.only_a],
            "only_b": [e.to_dict() for e in self.only_b],
            "common": self.common,
            "content_differs": [
                {"key": a.key, "kernel": a.name, "files": files}
                for a, _, files in self.content_differs
            ],
            "hints": self.hints,
        }


def diff_caches(a: CacheScan, b: CacheScan, *, compare_content: bool = True) -> CacheDiff:
    index_a = {(e.key, e.name): e for e in a.kernels}
    index_b = {(e.key, e.name): e for e in b.kernels}
    only_a = [index_a[k] for k in sorted(set(index_a) - set(index_b), key=str)]
    only_b = [index_b[k] for k in sorted(set(index_b) - set(index_a), key=str)]
    differs: list[tuple[CacheEntry, CacheEntry, list[str]]] = []
    common = sorted(set(index_a) & set(index_b), key=str)
    if compare_content:
        for key in common:
            ea, eb = index_a[key], index_b[key]
            files: list[str] = []
            for ext in ea.stages:
                if ext not in eb.stages:
                    files.append(f"{ext}: missing in B")
                    continue
                name = f"{ea.name}.{ext}"
                if ea.stages[ext] != eb.stages[ext] or sha256_file(ea.path / name) != sha256_file(
                    eb.path / name
                ):
                    files.append(f"{ext}: content differs")
            files.extend(f"{ext}: missing in A" for ext in eb.stages if ext not in ea.stages)
            if files:
                differs.append((ea, eb, files))
    return CacheDiff(only_a, only_b, len(common), differs, _diff_hints(only_a, only_b))


def _diff_hints(only_a: list[CacheEntry], only_b: list[CacheEntry]) -> list[str]:
    """What the on-disk metadata can say about why keys differ (often: not much)."""
    hints: list[str] = []
    by_name_b: dict[tuple[str | None, str], list[CacheEntry]] = {}
    for entry in only_b:
        by_name_b.setdefault((entry.name, entry.target), []).append(entry)
    unexplained = 0
    for entry in only_a:
        for other in by_name_b.get((entry.name, entry.target), []):
            reasons = []
            if entry.triton_version != other.triton_version:
                reasons.append(f"Triton {entry.triton_version} vs {other.triton_version}")
            if entry.env != other.env:
                reasons.append(f"env {entry.env} vs {other.env}")
            changed = sorted(
                k
                for k in set(entry.options) | set(other.options)
                if entry.options.get(k) != other.options.get(k)
                and k not in ("shared", "tmem_size", "global_scratch_size", "tensordesc_meta")
            )
            if changed:
                reasons.append("options differ: " + ", ".join(changed))
            if entry.source_file != other.source_file or entry.source_line != other.source_line:
                reasons.append(
                    f"defined at {entry.source_file}:{entry.source_line} vs "
                    f"{other.source_file}:{other.source_line}"
                )
            if reasons:
                hints.append(
                    f"{entry.name} ({entry.target}) {entry.key[:10]} vs {other.key[:10]}: "
                    + "; ".join(reasons)
                )
            else:
                unexplained += 1
    if unexplained:
        hints.append(
            f"{unexplained} pair(s) of same-named entries have identical metadata: the difference "
            "is in something the cache does not store (kernel source, a called function, the "
            "specialisation, or the Triton files). Record both runs and use 'triton-cachekit "
            "explain' to see which."
        )
    return hints


def render_diff(diff: CacheDiff, label_a: str, label_b: str) -> str:
    out = [f"A: {label_a}", f"B: {label_b}", ""]
    out.append(
        f"{diff.common} entries in both, {len(diff.only_a)} only in A, {len(diff.only_b)} only in B, "
        f"{len(diff.content_differs)} in both with different content"
    )
    for title, entries in (("Only in A", diff.only_a), ("Only in B", diff.only_b)):
        if entries:
            out.append("")
            out.append(f"{title}:")
            out.append(
                render_table(
                    ["KEY", "KERNEL", "TARGET", "TRITON", "SIZE", "MODIFIED"],
                    [
                        [
                            e.key[:10],
                            e.name or "?",
                            e.target,
                            e.triton_version or "?",
                            human_size(e.size),
                            format_time(e.mtime),
                        ]
                        for e in entries
                    ],
                    right_align=(4,),
                )
            )
    if diff.content_differs:
        out.append("")
        out.append("Same key, different content (non-deterministic build, or a key collision):")
        for entry, _, files in diff.content_differs:
            out.append(f"  {entry.key[:10]}  {entry.name}: " + "; ".join(files))
    if diff.hints:
        out.append("")
        out.append("Hints:")
        out.extend(f"  - {hint}" for hint in diff.hints)
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------- prune
@dataclass
class PrunePlan:
    remove: list[tuple[CacheEntry, str]]
    keep: list[CacheEntry]

    @property
    def freed(self) -> int:
        return sum(e.size for e, _ in self.remove)


def plan_prune(
    scan: CacheScan,
    *,
    older_than: float | None = None,
    max_size: int | None = None,
    targets: list[str] | None = None,
    kernel: str | None = None,
    incomplete: bool = False,
    triton_version_not: str | None = None,
    now: float | None = None,
) -> PrunePlan:
    """Decide what to delete.

    Selectors (``older_than``, ``targets``, ``kernel``, ``incomplete``,
    ``triton_version_not``) are OR-ed. ``max_size`` then evicts the least
    recently modified remaining entries until the cache fits. Triton never
    updates timestamps on a hit, so "modified" means "compiled".
    """
    current = time.time() if now is None else now
    remove: list[tuple[CacheEntry, str]] = []
    keep: list[CacheEntry] = []
    for entry in scan.entries:
        reason = None
        mtime = entry.mtime
        if incomplete and (
            entry.kind in ("empty", "unknown") or (entry.kind == "kernel" and not entry.loadable)
        ):
            reason = f"not loadable ({entry.status})"
        elif older_than is not None and mtime is not None and current - mtime > older_than:
            reason = f"older than cutoff (modified {format_time(mtime)})"
        elif targets and entry.kind == "kernel" and any(matches_target(entry, t) for t in targets):
            reason = f"target {entry.target}"
        elif kernel and entry.name and fnmatch.fnmatchcase(entry.name, kernel):
            reason = f"kernel matches {kernel!r}"
        elif (
            triton_version_not
            and entry.kind == "kernel"
            and entry.triton_version is not None
            and entry.triton_version != triton_version_not
        ):
            reason = f"written by Triton {entry.triton_version}, not {triton_version_not}"
        if reason:
            remove.append((entry, reason))
        else:
            keep.append(entry)
    if max_size is not None:
        total = sum(e.size for e in keep)
        for entry in sorted(keep, key=lambda e: e.mtime or 0.0):
            if total <= max_size:
                break
            remove.append((entry, "evicted to meet --max-size (oldest first)"))
            total -= entry.size
        removed = {id(e) for e, _ in remove}
        keep = [e for e in keep if id(e) not in removed]
    return PrunePlan(remove=remove, keep=keep)


def apply_prune(plan: PrunePlan) -> None:
    for entry, _ in plan.remove:
        if entry.kind == "empty":
            shutil.rmtree(entry.path, ignore_errors=True)
        else:
            remove_entry_files(entry)
            if entry.path.exists() and not any(
                p.is_file() and p.name != "lock" for p in entry.path.iterdir()
            ):
                shutil.rmtree(entry.path, ignore_errors=True)


def render_prune(plan: PrunePlan, *, applied: bool) -> str:
    out = []
    if plan.remove:
        out.append(
            render_table(
                ["KEY", "KERNEL", "TARGET", "SIZE", "REASON"],
                [
                    [e.key[:10], e.name or e.kind, e.target, human_size(e.size), reason]
                    for e, reason in plan.remove
                ],
                right_align=(3,),
            )
        )
        out.append("")
    verb = "Removed" if applied else "Would remove"
    out.append(
        f"{verb} {len(plan.remove)} entries ({human_size(plan.freed)}); "
        f"{len(plan.keep)} entries ({human_size(sum(e.size for e in plan.keep))}) kept."
    )
    if not applied and plan.remove:
        out.append("Dry run: nothing was deleted. Re-run with --yes to delete.")
    return "\n".join(out) + "\n"
