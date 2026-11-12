from __future__ import annotations

import hashlib

import pytest

from triton_cachekit import _util


def test_base32_key_round_trip() -> None:
    digest = hashlib.sha256(b"some key").hexdigest()
    key = _util.base32_key(digest)
    assert len(key) == 52 and "=" not in key
    assert _util.key_to_hex(key) == digest


def test_key_to_hex_accepts_legacy_hex_and_rejects_noise() -> None:
    assert (
        _util.key_to_hex("cff628804055ab05f902072733c9ab2d") == "cff628804055ab05f902072733c9ab2d"
    )
    assert _util.key_to_hex("not-a-key") is None
    assert _util.key_to_hex("tmp.pid_1_x") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [("1024", 1024), ("1k", 1024), ("500M", 500 * 1024**2), ("2G", 2 * 1024**3), ("1.5KiB", 1536)],
)
def test_parse_size(text: str, expected: int) -> None:
    assert _util.parse_size(text) == expected


def test_parse_size_and_age_reject_garbage() -> None:
    with pytest.raises(ValueError):
        _util.parse_size("lots")
    with pytest.raises(ValueError):
        _util.parse_age("30 days")


def test_parse_age() -> None:
    assert _util.parse_age("30d") == 30 * 86400
    assert _util.parse_age("12h") == 12 * 3600
    assert _util.parse_age("2w") == 14 * 86400


def test_human_formats() -> None:
    assert _util.human_size(0) == "0 B"
    assert _util.human_size(1536) == "1.5 KiB"
    assert _util.human_duration_us(999) == "999 us"
    assert _util.human_duration_us(12_345) == "12.3 ms"
    assert _util.human_duration_us(2_500_000) == "2.50 s"


def test_parse_target() -> None:
    assert _util.parse_target("cuda:80") == ("cuda", "80", None)
    assert _util.parse_target("hip:gfx942:64") == ("hip", "gfx942", 64)
    for bad in ("cuda", "metal:1", "cuda:80:x", "cuda::32"):
        with pytest.raises(ValueError):
            _util.parse_target(bad)


def test_render_table_alignment() -> None:
    table = _util.render_table(["A", "NUM"], [["x", "1"], ["long", "100"]], right_align=(1,))
    assert table.splitlines() == ["A     NUM", "----  ---", "x       1", "long  100"]
