"""Parser for Triton's on-disk compilation cache.

This module reads the directory layout that ``triton.runtime.cache.FileCacheManager``
writes. It deliberately does not import Triton, so a cache copied from another
machine can be inspected anywhere.

Layout (checked against real entries written by the Triton 3.8.0 and 3.9.0 wheels,
and against the source of ``main`` at 8a439de)::

    <cache dir>/
      <KEY>/                      base32(sha256(key string)), "=" padding stripped
        <kernel>.source           TTIR as produced by the frontend
        <kernel>.ttir .ttgir .llir
        <kernel>.ptx .cubin       NVIDIA   (or .amdgcn .hsaco for AMD)
        <kernel>.json             metadata: hash, target, options, env vars, ...
        __grp__<kernel>.json      {"child_paths": ...} - the entry is only valid
                                  once this file exists and every child exists
        tmp.pid_<pid>_<uuid>/     staging directory of an in-flight write

Two encodings of ``child_paths`` exist:

* ``{"name": "/absolute/path", ...}`` (3.8.0, 3.9.0): absolute paths. A cache
  directory that was copied or mounted at another path is a *miss* for every entry
  unless the original paths still exist (upstream issue #11999).
* ``["name", ...]`` (``main`` after the fix for #11999): names relative to the
  entry directory.

The same directory also holds entries that are not kernels: autotuner results
(``<fn>.autotune.json``) and natively compiled launcher stubs (``*.so``).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._util import key_to_hex

GROUP_PREFIX = "__grp__"
#: Lowering stages in pipeline order. ``source`` is the frontend output.
STAGE_ORDER: tuple[str, ...] = (
    "source",
    "ttir",
    "ttgir",
    "llir",
    "ptx",
    "cubin",
    "amdgcn",
    "hsaco",
)
BINARY_EXT: dict[str, str] = {"cuda": "cubin", "hip": "hsaco"}
#: Options worth showing in a one-line summary, in display order.
SUMMARY_OPTIONS: tuple[str, ...] = (
    "num_warps",
    "num_stages",
    "num_ctas",
    "waves_per_eu",
    "enable_fp_fusion",
    "debug",
    "maxnreg",
)
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_LOC_RE = re.compile(r'loc\("([^"\n]+)":(\d+):(\d+)')
_TMP_DIR_RE = re.compile(r"^tmp\.pid_(\d+)_")
_SOURCE_PROBE_BYTES = 64 * 1024


@dataclass(frozen=True)
class FileInfo:
    name: str
    size: int
    mtime: float


@dataclass(frozen=True)
class Problem:
    """Something wrong (or noteworthy) about a cache entry."""

    code: str
    message: str
    severity: str = "error"  # "error": Triton will not load it; "warning": it may

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "severity": self.severity, "message": self.message}


@dataclass
class CacheEntry:
    """One logical entry (one kernel, autotune result or native module) in a key dir."""

    key: str
    path: Path
    kind: str  # "kernel" | "autotune" | "native" | "empty" | "unknown"
    name: str | None = None
    key_hex: str | None = None
    metadata: dict[str, Any] | None = None
    group_format: str | None = None  # "absolute" | "relative" | "invalid" | None
    group_children: dict[str, str] = field(default_factory=dict)
    files: list[FileInfo] = field(default_factory=list)
    problems: list[Problem] = field(default_factory=list)
    source_file: str | None = None
    source_line: int | None = None

    # ---- derived properties -------------------------------------------------
    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def mtime(self) -> float | None:
        return max((f.mtime for f in self.files), default=None)

    @property
    def created(self) -> float | None:
        return min((f.mtime for f in self.files), default=None)

    @property
    def backend(self) -> str | None:
        target = (self.metadata or {}).get("target")
        if isinstance(target, dict):
            value = target.get("backend")
            return str(value) if value is not None else None
        return None

    @property
    def arch(self) -> str | None:
        target = (self.metadata or {}).get("target")
        if isinstance(target, dict) and target.get("arch") is not None:
            return str(target["arch"])
        return None

    @property
    def warp_size(self) -> int | None:
        target = (self.metadata or {}).get("target")
        if isinstance(target, dict) and isinstance(target.get("warp_size"), int):
            return int(target["warp_size"])
        return None

    @property
    def target(self) -> str:
        if self.backend is None:
            return "-"
        return f"{self.backend}:{self.arch}"

    @property
    def triton_version(self) -> str | None:
        value = (self.metadata or {}).get("triton_version")
        return str(value) if value is not None else None

    @property
    def options(self) -> dict[str, Any]:
        """Compiler options stored in the metadata (everything that is not env/target)."""
        meta = self.metadata or {}
        skip = {"hash", "target", "triton_version", "name"}
        return {
            k: v
            for k, v in meta.items()
            if k not in skip and not _ENV_NAME_RE.match(k) and not isinstance(v, dict)
        }

    @property
    def env(self) -> dict[str, str]:
        """Cache-invalidating environment variables that were set when compiling.

        ``compile()`` merges them into the metadata under their own (upper-case)
        names; option names are lower-case, which is how they are told apart.
        """
        meta = self.metadata or {}
        return {k: str(v) for k, v in meta.items() if _ENV_NAME_RE.match(k)}

    @property
    def stages(self) -> dict[str, int]:
        """Artifact stages present, mapped to their size in bytes."""
        if self.name is None:
            return {}
        found: dict[str, int] = {}
        prefix = self.name + "."
        for info in self.files:
            if info.name.startswith(prefix):
                ext = info.name[len(prefix) :]
                if ext in STAGE_ORDER:
                    found[ext] = info.size
        return {ext: found[ext] for ext in STAGE_ORDER if ext in found}

    @property
    def binary_ext(self) -> str | None:
        backend = self.backend
        if backend in BINARY_EXT:
            return BINARY_EXT[backend]
        for ext in ("cubin", "hsaco"):
            if ext in self.stages:
                return ext
        return None

    @property
    def status(self) -> str:
        """``ok`` | ``stale-paths`` | ``incomplete`` | ``corrupt``."""
        codes = {p.code for p in self.problems if p.severity == "error"}
        if codes & {"metadata-invalid", "group-invalid", "hash-mismatch"}:
            return "corrupt"
        if codes - {"group-stale-paths"}:
            return "incomplete"
        if "group-stale-paths" in codes:
            return "stale-paths"
        return "ok"

    @property
    def loadable(self) -> bool:
        """Would Triton's ``get_group`` accept this entry where it sits now?"""
        return self.status == "ok"

    def summary_options(self) -> dict[str, Any]:
        opts = self.options
        return {k: opts[k] for k in SUMMARY_OPTIONS if k in opts and opts[k] is not None}

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "key_hex": self.key_hex,
            "path": str(self.path),
            "kind": self.kind,
            "kernel": self.name,
            "status": self.status,
            "backend": self.backend,
            "arch": self.arch,
            "warp_size": self.warp_size,
            "triton_version": self.triton_version,
            "options": self.options,
            "env": self.env,
            "stages": self.stages,
            "size": self.size,
            "files": [{"name": f.name, "size": f.size, "mtime": f.mtime} for f in self.files],
            "created": self.created,
            "modified": self.mtime,
            "group_format": self.group_format,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "problems": [p.to_dict() for p in self.problems],
        }


@dataclass
class CacheScan:
    root: Path
    entries: list[CacheEntry]
    stray: list[str] = field(default_factory=list)  # top-level names that are not entries

    @property
    def kernels(self) -> list[CacheEntry]:
        return [e for e in self.entries if e.kind == "kernel"]

    @property
    def total_size(self) -> int:
        return sum(e.size for e in self.entries)


def looks_like_cache_dir(path: Path) -> bool:
    """Cheap test used for helpful error messages."""
    if not path.is_dir():
        return False
    for child in _iter_dirs(path):
        try:
            names = os.listdir(child)
        except OSError:
            continue
        if any(n.startswith(GROUP_PREFIX) or n.endswith(".json") for n in names):
            return True
    return False


def default_cache_dir(env: dict[str, str] | None = None) -> Path:
    """Where Triton puts its cache by default (``knobs.cache`` in Triton 3.9.0).

    ``TRITON_CACHE_DIR`` wins; otherwise ``$TRITON_HOME/.triton/cache`` with
    ``TRITON_HOME`` defaulting to the home directory.
    """
    environ = os.environ if env is None else env
    explicit = environ.get("TRITON_CACHE_DIR")
    if explicit:
        return Path(explicit)
    home = environ.get("TRITON_HOME") or os.path.expanduser("~/")
    return Path(home) / ".triton" / "cache"


def _iter_dirs(root: Path) -> Iterator[Path]:
    try:
        with os.scandir(root) as it:
            for item in sorted(it, key=lambda e: e.name):
                if item.is_dir(follow_symlinks=False):
                    yield Path(item.path)
    except OSError:
        return


def scan_cache(root: Path | str) -> CacheScan:
    """Scan ``root`` and return every entry found, including broken ones."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"cache directory not found: {root_path}")
    entries: list[CacheEntry] = []
    stray: list[str] = []
    with os.scandir(root_path) as it:
        children = sorted(it, key=lambda e: e.name)
    for child in children:
        if child.is_dir(follow_symlinks=False):
            entries.extend(read_entry_dir(Path(child.path)))
        else:
            stray.append(child.name)
    return CacheScan(root=root_path, entries=entries, stray=stray)


def _list_files(directory: Path) -> tuple[list[FileInfo], list[str]]:
    files: list[FileInfo] = []
    tmp_dirs: list[str] = []
    try:
        with os.scandir(directory) as it:
            for item in sorted(it, key=lambda e: e.name):
                if item.is_dir(follow_symlinks=False):
                    if _TMP_DIR_RE.match(item.name):
                        tmp_dirs.append(item.name)
                    continue
                try:
                    stat = item.stat(follow_symlinks=False)
                except OSError:
                    continue
                files.append(FileInfo(item.name, stat.st_size, stat.st_mtime))
    except OSError:
        pass
    return files, tmp_dirs


def _load_json(path: Path) -> tuple[Any, str | None]:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle), None
    except (OSError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _probe_source_location(directory: Path, name: str) -> tuple[str | None, int | None]:
    """Recover the kernel's Python file and line from MLIR debug locations.

    The frontend emits ``#loc = loc("/path/to/file.py":LINE:COL)`` for the kernel
    definition; nothing else in the entry records where the kernel came from.
    Absent when the cache was written with ``TRITON_DISABLE_LINE_INFO=1`` or
    ``TRITON_STORE_BINARY_ONLY=1``.
    """
    for ext in ("source", "ttir"):
        path = directory / f"{name}.{ext}"
        try:
            with open(path, "rb") as handle:
                head = handle.read(_SOURCE_PROBE_BYTES).decode("utf-8", errors="replace")
        except OSError:
            continue
        # Prefer the definition's own "#loc = loc(...)" line when it is present.
        for line in head.splitlines():
            if line.startswith("#loc = "):
                match = _LOC_RE.search(line)
                if match:
                    return match.group(1), int(match.group(2))
        match = _LOC_RE.search(head)
        if match:
            return match.group(1), int(match.group(2))
    return None, None


def _kernel_names(files: list[FileInfo]) -> list[str]:
    names: set[str] = set()
    for info in files:
        if info.name.startswith(GROUP_PREFIX) and info.name.endswith(".json"):
            names.add(info.name[len(GROUP_PREFIX) : -len(".json")])
    for info in files:
        if info.name.startswith(GROUP_PREFIX):
            continue
        stem, dot, ext = info.name.rpartition(".")
        if dot and ext in STAGE_ORDER:
            names.add(stem)
    # A bare "<name>.json" with nothing else is metadata without artifacts.
    for info in files:
        if info.name.startswith(GROUP_PREFIX) or not info.name.endswith(".json"):
            continue
        stem = info.name[: -len(".json")]
        if stem.endswith(".autotune"):
            continue
        names.add(stem)
    return sorted(names)


def read_entry_dir(directory: Path) -> list[CacheEntry]:
    """Parse one ``<cache>/<KEY>`` directory into entries (usually exactly one)."""
    key = directory.name
    key_hex = key_to_hex(key)
    files, tmp_dirs = _list_files(directory)
    payload = [f for f in files if f.name != "lock"]

    if not payload:
        entry = CacheEntry(key=key, path=directory, kind="empty", key_hex=key_hex, files=files)
        if tmp_dirs:
            entry.problems.append(
                Problem(
                    "interrupted-write",
                    f"only staging directories present ({len(tmp_dirs)}): a compilation was "
                    "interrupted or is still running",
                )
            )
        else:
            entry.problems.append(Problem("empty", "directory holds no cache files"))
        return [entry]

    autotune = [f for f in payload if f.name.endswith(".autotune.json")]
    native = [f for f in payload if f.name.endswith((".so", ".pyd", ".dylib"))]
    kernel_files = [f for f in payload if f not in autotune and f not in native]
    entries: list[CacheEntry] = []

    for info in autotune:
        entries.append(
            CacheEntry(
                key=key,
                path=directory,
                kind="autotune",
                name=info.name[: -len(".autotune.json")],
                key_hex=key_hex,
                files=[info],
            )
        )
    if native:
        entries.append(
            CacheEntry(
                key=key,
                path=directory,
                kind="native",
                name=native[0].name.split(".")[0],
                key_hex=key_hex,
                files=native + [f for f in kernel_files if f.name.endswith((".c", ".cpp", ".o"))],
            )
        )
        kernel_files = [f for f in kernel_files if not f.name.endswith((".c", ".cpp", ".o"))]

    names = _kernel_names(kernel_files)
    for name in names:
        own = [
            f
            for f in kernel_files
            if f.name == f"{GROUP_PREFIX}{name}.json" or f.name.rpartition(".")[0] == name
        ]
        entries.append(_read_kernel(directory, key, key_hex, name, own, tmp_dirs))

    claimed = {f.name for e in entries for f in e.files}
    leftovers = [f for f in payload if f.name not in claimed]
    if leftovers and not entries:
        entries.append(
            CacheEntry(
                key=key,
                path=directory,
                kind="unknown",
                key_hex=key_hex,
                files=leftovers,
                problems=[
                    Problem(
                        "unrecognised",
                        "no Triton cache files recognised in this directory",
                        "warning",
                    )
                ],
            )
        )
    return entries


def _read_kernel(
    directory: Path,
    key: str,
    key_hex: str | None,
    name: str,
    files: list[FileInfo],
    tmp_dirs: list[str],
) -> CacheEntry:
    entry = CacheEntry(
        key=key, path=directory, kind="kernel", name=name, key_hex=key_hex, files=files
    )
    present = {f.name for f in files}
    problems = entry.problems

    # ---- metadata -----------------------------------------------------------
    meta_name = f"{name}.json"
    if meta_name in present:
        data, error = _load_json(directory / meta_name)
        if error is not None or not isinstance(data, dict):
            problems.append(
                Problem(
                    "metadata-invalid",
                    f"{meta_name} is not valid JSON ({error or 'not an object'})",
                )
            )
        else:
            entry.metadata = data
            recorded = data.get("hash")
            if isinstance(recorded, str) and key_hex is not None and recorded != key_hex:
                problems.append(
                    Problem(
                        "hash-mismatch",
                        f"metadata hash {recorded[:12]}... does not match the directory name "
                        f"({key_hex[:12]}...): files were moved between entries",
                    )
                )
    else:
        problems.append(
            Problem("metadata-missing", f"{meta_name} is missing (compilation did not finish)")
        )

    # ---- group file: this is what Triton checks for a hit --------------------
    group_name = f"{GROUP_PREFIX}{name}.json"
    if group_name in present:
        data, error = _load_json(directory / group_name)
        children = data.get("child_paths") if isinstance(data, dict) else None
        if error is not None or children is None:
            entry.group_format = "invalid"
            problems.append(
                Problem(
                    "group-invalid",
                    f"{group_name} is unreadable or has no 'child_paths' ({error or 'bad shape'})",
                )
            )
        elif isinstance(children, dict):
            entry.group_format = "absolute"
            entry.group_children = {str(k): str(v) for k, v in children.items()}
            _check_absolute_group(entry, directory, present)
        elif isinstance(children, list):
            entry.group_format = "relative"
            entry.group_children = {str(c): str(c) for c in children}
            missing = [c for c in entry.group_children if not (directory / c).exists()]
            if missing:
                problems.append(
                    Problem(
                        "group-child-missing",
                        "group lists files that are missing: " + ", ".join(sorted(missing)),
                    )
                )
        else:
            entry.group_format = "invalid"
            problems.append(
                Problem("group-invalid", f"{group_name}: unexpected 'child_paths' type")
            )
    else:
        problems.append(
            Problem(
                "group-missing",
                f"{group_name} is missing: Triton treats the entry as absent and recompiles",
            )
        )

    # ---- artifacts ------------------------------------------------------------
    binary = entry.binary_ext
    stages = entry.stages
    if binary is not None and binary not in stages:
        problems.append(
            Problem("binary-missing", f"{name}.{binary} (the compiled binary) is missing")
        )
    elif binary is not None and stages.get(binary) == 0:
        problems.append(Problem("binary-empty", f"{name}.{binary} is empty"))
    if binary is None and entry.metadata is not None:
        problems.append(
            Problem("binary-missing", "no compiled binary (.cubin/.hsaco) found", "error")
        )
    if tmp_dirs:
        problems.append(
            Problem(
                "interrupted-write",
                f"{len(tmp_dirs)} staging director{'y' if len(tmp_dirs) == 1 else 'ies'} "
                "left behind (tmp.pid_*): a writer crashed or is still running",
                "warning",
            )
        )
    entry.source_file, entry.source_line = _probe_source_location(directory, name)
    return entry


def _check_absolute_group(entry: CacheEntry, directory: Path, present: set[str]) -> None:
    """Replicate ``FileCacheManager.get_group`` for the absolute-path encoding."""
    here = os.path.abspath(directory)
    missing_local = [n for n in entry.group_children if n not in present]
    if missing_local:
        entry.problems.append(
            Problem(
                "group-child-missing",
                "group lists files that are missing: " + ", ".join(sorted(missing_local)),
            )
        )
        return
    elsewhere = [
        p for p in entry.group_children.values() if os.path.dirname(os.path.abspath(p)) != here
    ]
    if not elsewhere:
        return
    original = os.path.dirname(os.path.dirname(elsewhere[0]))
    if all(os.path.exists(p) for p in entry.group_children.values()):
        entry.problems.append(
            Problem(
                "group-foreign-paths",
                f"group file points at another cache directory that still exists ({original}): "
                "Triton would load the files from there, not from this directory. "
                "Run 'triton-cachekit relocate' to fix.",
                "warning",
            )
        )
    else:
        entry.problems.append(
            Problem(
                "group-stale-paths",
                f"group file records absolute paths under {original}, which does not exist here: "
                "Triton releases that store absolute paths (3.8.0 and 3.9.0 verified) treat this "
                "entry as a miss and recompile. Run 'triton-cachekit relocate' to fix.",
            )
        )
