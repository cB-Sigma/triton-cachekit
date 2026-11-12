"""Small helpers shared by the command implementations (standard library only)."""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

_SIZE_UNITS = ("B", "KiB", "MiB", "GiB", "TiB")
_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)(i?b)?\s*$", re.IGNORECASE)
_AGE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw])\s*$", re.IGNORECASE)
_AGE_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def human_size(num: float) -> str:
    """Format a byte count using binary units."""
    value = float(num)
    for unit in _SIZE_UNITS:
        if abs(value) < 1024.0 or unit == _SIZE_UNITS[-1]:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def parse_size(text: str) -> int:
    """Parse ``500M``, ``2G``, ``1024`` ... into bytes (binary multiples)."""
    match = _SIZE_RE.match(text)
    if not match:
        raise ValueError(f"cannot parse size {text!r} (examples: 500M, 2G, 1048576)")
    number = float(match.group(1))
    exponent = " kmgt".index(match.group(2).lower() or " ")
    return int(number * (1024**exponent))


def parse_age(text: str) -> float:
    """Parse ``30d``, ``12h``, ``90m`` ... into seconds."""
    match = _AGE_RE.match(text)
    if not match:
        raise ValueError(f"cannot parse age {text!r} (examples: 30d, 12h, 90m, 2w)")
    return float(match.group(1)) * _AGE_SECONDS[match.group(2).lower()]


def human_duration_us(microseconds: float) -> str:
    """Format a duration given in microseconds."""
    if microseconds < 1000:
        return f"{microseconds:.0f} us"
    millis = microseconds / 1000.0
    if millis < 1000:
        return f"{millis:.1f} ms"
    return f"{millis / 1000.0:.2f} s"


def format_time(epoch: float | None) -> str:
    """Format a timestamp as local ``YYYY-MM-DD HH:MM``."""
    if epoch is None:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(epoch))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def base32_key(hex_digest: str) -> str:
    """Directory name Triton derives from a hex digest (base32, padding stripped)."""
    return base64.b32encode(bytes.fromhex(hex_digest)).decode("ascii").rstrip("=")


def key_to_hex(key: str) -> str | None:
    """Invert :func:`base32_key`; ``None`` when ``key`` is not a base32 digest."""
    if re.fullmatch(r"[0-9a-f]{32}|[0-9a-f]{64}", key):
        return key  # very old Triton releases used the hex digest directly
    if not re.fullmatch(r"[A-Z2-7]+", key):
        return None
    padded = key + "=" * (-len(key) % 8)
    try:
        return base64.b32decode(padded).hex()
    except (binascii.Error, ValueError):
        return None


def render_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    right_align: Sequence[int] = (),
) -> str:
    """Render a plain-text table with two-space column gaps."""
    materialised = [list(map(str, row)) for row in rows]
    widths = [len(h) for h in headers]
    for row in materialised:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    right = set(right_align)

    def fmt(cells: Sequence[str]) -> str:
        parts = []
        for index, cell in enumerate(cells):
            parts.append(cell.rjust(widths[index]) if index in right else cell.ljust(widths[index]))
        return "  ".join(parts).rstrip()

    lines = [fmt(headers), fmt(["-" * w for w in widths])]
    lines.extend(fmt(row) for row in materialised)
    return "\n".join(lines)


def shorten(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def parse_target(text: str) -> tuple[str, str, int | None]:
    """Parse ``backend:arch[:warp_size]`` (the format Triton's AOT tool uses).

    Returns ``(backend, arch, warp_size)``; ``arch`` stays a string here.
    """
    parts = text.split(":")
    if len(parts) not in (2, 3) or not all(parts):
        raise ValueError(
            f"invalid target {text!r}: expected '<backend>:<arch>[:<warp-size>]', "
            "e.g. 'cuda:80', 'cuda:90:32' or 'hip:gfx942:64'"
        )
    backend, arch = parts[0], parts[1]
    if backend not in ("cuda", "hip"):
        raise ValueError(f"invalid target {text!r}: backend must be 'cuda' or 'hip'")
    warp: int | None = None
    if len(parts) == 3:
        try:
            warp = int(parts[2])
        except ValueError as exc:
            raise ValueError(f"invalid target {text!r}: warp size must be an integer") from exc
    return backend, arch, warp


def target_label(backend: object, arch: object) -> str:
    return f"{backend}:{arch}"
