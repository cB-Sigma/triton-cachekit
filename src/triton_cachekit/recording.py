"""Load recordings written by :mod:`triton_cachekit.recorder`."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class RecordingError(ValueError):
    """The file is not a usable triton-cachekit recording."""


@dataclass
class Compile:
    """One call to ``triton.compile`` as seen by the recorder."""

    raw: dict[str, Any]
    outcome: dict[str, Any] | None = None
    session: dict[str, Any] | None = None

    @property
    def kernel(self) -> str:
        return str(self.raw.get("kernel", "?"))

    @property
    def name(self) -> str:
        return str(self.raw.get("name", self.kernel.rpartition(".")[2]))

    @property
    def hash(self) -> str:
        return str(self.raw["hash"])

    @property
    def entry(self) -> str:
        return str(self.raw.get("entry", ""))

    @property
    def components(self) -> dict[str, Any]:
        value = self.raw.get("components")
        return value if isinstance(value, dict) else {}

    @property
    def cache_hit(self) -> bool | None:
        if self.outcome is None:
            return None
        return bool(self.outcome.get("cache_hit"))

    @property
    def total_us(self) -> int | None:
        if self.outcome is None:
            return None
        return int(self.outcome.get("times_us", {}).get("total", 0))

    @property
    def stage_us(self) -> dict[str, int]:
        if self.outcome is None:
            return {}
        times = self.outcome.get("times_us", {})
        stages = {"frontend": int(times.get("ir_initialization", 0))}
        stages.update({str(k): int(v) for k, v in dict(times.get("lowering", {})).items()})
        stages["store"] = int(times.get("store_results", 0))
        return stages

    @property
    def target(self) -> str:
        target = self.components.get("backend", {}).get("target") or {}
        if isinstance(target, dict) and target:
            return f"{target.get('backend')}:{target.get('arch')}"
        return "-"

    @property
    def cache_dir(self) -> str | None:
        value = self.raw.get("cache_dir")
        return str(value) if value else None

    @property
    def specialization_data(self) -> str | None:
        value = self.raw.get("specialization_data")
        return value if isinstance(value, str) else None

    def describe_specialization(self) -> str:
        """Short human-readable description: constexpr values and pointer types."""
        source = self.components.get("source", {})
        parts = []
        for const in source.get("constants", []) or []:
            label = const.get("name") or ".".join(map(str, const.get("path", [])))
            parts.append(f"{label}={const.get('value')}")
        return ", ".join(parts)


@dataclass
class Recording:
    path: str
    sessions: list[dict[str, Any]] = field(default_factory=list)
    compiles: list[Compile] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def session(self) -> dict[str, Any] | None:
        """The first session (recordings usually hold exactly one process)."""
        return self.sessions[0] if self.sessions else None

    def by_hash(self) -> dict[str, list[Compile]]:
        table: dict[str, list[Compile]] = {}
        for item in self.compiles:
            table.setdefault(item.hash, []).append(item)
        return table

    @property
    def label(self) -> str:
        return Path(self.path).name


def _iter_lines(paths: Iterable[Path]) -> Iterable[tuple[Path, int, str]]:
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    yield path, number, line


def expand_paths(path: str | Path) -> list[Path]:
    """A recording is a ``.jsonl`` file or a directory of them."""
    target = Path(path)
    if target.is_dir():
        found = sorted(target.glob("*.jsonl"))
        if not found:
            raise RecordingError(f"{target}: directory contains no .jsonl recordings")
        return found
    if not target.is_file():
        raise RecordingError(f"{target}: recording not found")
    return [target]


def load_recording(path: str | Path) -> Recording:
    recording = Recording(path=str(path))
    sessions: dict[tuple[str, int], dict[str, Any]] = {}
    open_compiles: dict[tuple[str, int, str], Compile] = {}
    for file, number, line in _iter_lines(expand_paths(path)):
        try:
            record = json.loads(line)
        except ValueError as exc:
            # A process killed mid-write leaves a truncated last line; keep the rest.
            recording.warnings.append(f"{file}:{number}: skipped unparsable line ({exc})")
            continue
        if not isinstance(record, dict) or "type" not in record:
            recording.warnings.append(f"{file}:{number}: skipped line without a 'type'")
            continue
        kind = record["type"]
        pid = int(record.get("pid", 0))
        if kind == "session":
            schema = record.get("schema")
            if schema != 1:
                raise RecordingError(
                    f"{file}:{number}: unsupported recording schema {schema!r} "
                    "(this triton-cachekit reads schema 1)"
                )
            sessions[(str(file), pid)] = record
            recording.sessions.append(record)
        elif kind == "compile":
            item = Compile(raw=record, session=sessions.get((str(file), pid)))
            recording.compiles.append(item)
            open_compiles[(str(file), pid, str(record.get("hash")))] = item
        elif kind == "outcome":
            match = open_compiles.pop((str(file), pid, str(record.get("hash"))), None)
            if match is not None:
                match.outcome = record
        elif kind == "warning":
            recording.warnings.append(str(record.get("message")))
    if not recording.sessions and not recording.compiles:
        raise RecordingError(f"{path}: no triton-cachekit records found (is this a recording?)")
    return recording
