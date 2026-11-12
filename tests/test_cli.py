"""CLI behaviour that does not need Triton."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from triton_cachekit import __version__

from . import factories as f
from .conftest import KEY_A, make_entry


def test_version_and_help(cli: Any) -> None:
    assert cli("--version").stdout.strip() == f"triton-cachekit {__version__}"
    result = cli()
    assert result.returncode == 2 and "inspect" in result.stdout
    for command in (
        "inspect",
        "explain",
        "record",
        "stats",
        "prewarm",
        "verify-hit",
        "pack",
        "unpack",
        "relocate",
        "diff",
        "prune",
        "fingerprint",
    ):
        assert cli(command, "--help").returncode == 0


def test_inspect_table_json_and_strict(cli: Any, synthetic_cache: Path, tmp_path: Path) -> None:
    result = cli("inspect", synthetic_cache)
    assert result.returncode == 0, result.stderr
    assert "cuda:80" in result.stdout and "hip:gfx942" in result.stdout
    data = json.loads(cli("inspect", synthetic_cache, "--format", "json").stdout)
    assert data["totals"]["by_status"] == {"ok": 2}

    make_entry(tmp_path / "bad", group="none")
    assert cli("inspect", tmp_path / "bad").returncode == 0
    strict = cli("inspect", tmp_path / "bad", "--strict")
    assert (
        strict.returncode == 1
        and "group-missing" not in strict.stdout
        and "is missing" in strict.stdout
    )


def test_inspect_uses_triton_cache_dir_and_reports_missing(
    cli: Any, synthetic_cache: Path, tmp_path: Path
) -> None:
    result = cli("inspect", env={"TRITON_CACHE_DIR": str(synthetic_cache)})
    assert result.returncode == 0 and str(synthetic_cache) in result.stdout
    missing = cli("inspect", tmp_path / "nope")
    assert missing.returncode == 2 and "cache directory not found" in missing.stderr


def test_explain_stats_cli(cli: Any, tmp_path: Path) -> None:
    a = f.recording(tmp_path / "a.jsonl", [(f.compile_record(), False)])
    b = f.recording(tmp_path / "b.jsonl", [(f.compile_record(fn=f.fn(line=20)), False)])
    result = cli("explain", a, b)
    assert result.returncode == 0 and "[source.location.line]" in result.stdout
    assert cli("explain", a, b, "--fail-on-miss").returncode == 1
    assert json.loads(cli("explain", a, b, "--format", "json").stdout)["summary"] == {
        "miss-key-differs": 1
    }
    assert "No key collisions" in cli("explain", "--collisions", a, b).stdout
    assert cli("explain", a).returncode == 2
    assert cli("explain", a, tmp_path / "zzz.jsonl").returncode == 2
    stats = cli("stats", a, b)
    assert stats.returncode == 0 and "2 cache misses" in stats.stdout
    assert json.loads(cli("stats", a, "--format", "json").stdout)["misses"] == 1


def test_pack_unpack_relocate_diff_prune_cli(
    cli: Any, synthetic_cache: Path, tmp_path: Path
) -> None:
    archive = tmp_path / "cache.tar.gz"
    packed = cli("pack", synthetic_cache, "-o", archive)
    assert packed.returncode == 0, packed.stderr
    assert "Packed 2 entries" in packed.stdout and "Relocation:" in packed.stdout
    dest = tmp_path / "new" / "cache"
    unpacked = cli("unpack", archive, dest)
    assert unpacked.returncode == 0, unpacked.stderr
    assert (
        "checksums verified" in unpacked.stdout
        and "Rewrote group files of 1 entries" in unpacked.stdout
    )
    assert cli("inspect", dest, "--strict").returncode == 0
    assert "Nothing to do" in cli("relocate", dest).stdout

    same = cli("diff", synthetic_cache, dest, "--exit-code")
    assert same.returncode == 0 and "2 entries in both, 0 only in A" in same.stdout

    assert cli("prune", dest).returncode == 2
    dry = cli("prune", dest, "--target", "cuda")
    assert "Would remove 1 entries" in dry.stdout and (dest / KEY_A).exists()
    done = cli("prune", dest, "--target", "cuda", "--yes")
    assert "Removed 1 entries" in done.stdout and not (dest / KEY_A).exists()
    assert cli("diff", synthetic_cache, dest, "--exit-code").returncode == 1
    assert cli("unpack", tmp_path / "missing.tar", dest).returncode == 2
