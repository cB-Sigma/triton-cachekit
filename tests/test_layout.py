"""On-disk parsing with text-only entries (no Triton needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from triton_cachekit.inspect_report import filter_entries, render_json, render_table_report, totals
from triton_cachekit.layout import default_cache_dir, read_entry_dir, scan_cache

from .conftest import HASH_A, KEY_A, KEY_B, make_entry


def codes(entry: object) -> set[str]:
    return {p.code for p in entry.problems}  # type: ignore[attr-defined]


def test_scan_reads_both_group_encodings(synthetic_cache: Path) -> None:
    scan = scan_cache(synthetic_cache)
    by_key = {e.key: e for e in scan.entries}
    cuda, hip = by_key[KEY_A], by_key[KEY_B]
    assert cuda.kind == hip.kind == "kernel"
    assert (cuda.group_format, hip.group_format) == ("absolute", "relative")
    assert cuda.status == hip.status == "ok"
    assert cuda.target == "cuda:80" and hip.target == "hip:gfx942"
    assert cuda.key_hex == HASH_A
    assert list(cuda.stages) == ["source", "ttir", "ttgir", "llir", "ptx", "cubin"]
    assert list(hip.stages) == ["source", "ttir", "ttgir", "llir", "amdgcn", "hsaco"]
    assert cuda.triton_version == "3.9.0"
    assert (cuda.source_file, cuda.source_line) == ("/src/pkg/kernels.py", 11)
    assert cuda.summary_options()["num_warps"] == 4


def test_env_vars_are_separated_from_options(tmp_path: Path) -> None:
    make_entry(tmp_path, env={"TRITON_DISABLE_LINE_INFO": "true"})
    (entry,) = scan_cache(tmp_path).entries
    assert entry.env == {"TRITON_DISABLE_LINE_INFO": "true"}
    assert "TRITON_DISABLE_LINE_INFO" not in entry.options


def test_missing_group_file_is_incomplete(tmp_path: Path) -> None:
    make_entry(tmp_path, group="none")
    (entry,) = scan_cache(tmp_path).entries
    assert entry.status == "incomplete" and "group-missing" in codes(entry)
    assert not entry.loadable


def test_missing_binary_and_child(tmp_path: Path) -> None:
    directory = make_entry(tmp_path)
    (directory / "axpy.cubin").unlink()
    (entry,) = scan_cache(tmp_path).entries
    assert entry.status == "incomplete"
    assert {"binary-missing", "group-child-missing"} <= codes(entry)


def test_corrupt_metadata_and_group(tmp_path: Path) -> None:
    directory = make_entry(tmp_path)
    (directory / "axpy.json").write_text('{"hash": "28cb')
    (entry,) = scan_cache(tmp_path).entries
    assert entry.status == "corrupt" and "metadata-invalid" in codes(entry)

    (directory / "axpy.json").write_text(
        json.dumps({"hash": HASH_A, "target": {"backend": "cuda", "arch": 80}})
    )
    (directory / "__grp__axpy.json").write_text("{}")
    (entry,) = scan_cache(tmp_path).entries
    assert entry.status == "corrupt" and "group-invalid" in codes(entry)


def test_hash_mismatch_is_corrupt(tmp_path: Path) -> None:
    make_entry(tmp_path, KEY_B, HASH_A)
    (entry,) = scan_cache(tmp_path).entries
    assert entry.status == "corrupt" and "hash-mismatch" in codes(entry)


def test_stale_absolute_paths_are_reported(tmp_path: Path) -> None:
    make_entry(tmp_path / "cache", group_root=tmp_path / "gone")
    (entry,) = scan_cache(tmp_path / "cache").entries
    assert entry.status == "stale-paths" and not entry.loadable
    assert "relocate" in entry.problems[0].message


def test_foreign_paths_that_still_exist_are_a_warning(tmp_path: Path) -> None:
    make_entry(tmp_path / "original")
    make_entry(tmp_path / "copy", group_root=tmp_path / "original")
    (entry,) = scan_cache(tmp_path / "copy").entries
    assert entry.status == "ok"
    assert codes(entry) == {"group-foreign-paths"}


def test_interrupted_write_and_empty_dirs(tmp_path: Path) -> None:
    directory = make_entry(tmp_path)
    (directory / "tmp.pid_123_abcd").mkdir()
    (tmp_path / KEY_B / "tmp.pid_9_x").mkdir(parents=True)
    (tmp_path / "stray.txt").write_text("x")
    scan = scan_cache(tmp_path)
    by_key = {e.key: e for e in scan.entries}
    assert "interrupted-write" in codes(by_key[KEY_A]) and by_key[KEY_A].status == "ok"
    assert by_key[KEY_B].kind == "empty"
    assert scan.stray == ["stray.txt"]


def test_autotune_and_native_entries_are_classified(tmp_path: Path) -> None:
    (tmp_path / KEY_A).mkdir()
    (tmp_path / KEY_A / "matmul.autotune.json").write_text('{"key": [], "configs_timings": []}')
    (tmp_path / KEY_B).mkdir()
    (tmp_path / KEY_B / "__triton_launcher.cpython-313-x86_64-linux-gnu.so").write_bytes(b"\x7fELF")
    kinds = {e.key: (e.kind, e.name) for e in scan_cache(tmp_path).entries}
    assert kinds[KEY_A] == ("autotune", "matmul")
    assert kinds[KEY_B][0] == "native"


def test_metadata_without_triton_version_is_tolerated(tmp_path: Path) -> None:
    make_entry(tmp_path, triton_version=None)
    (entry,) = scan_cache(tmp_path).entries
    assert entry.triton_version is None and entry.status == "ok"
    assert "?" in render_table_report(scan_cache(tmp_path), [entry])


def test_scan_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scan_cache(tmp_path / "nope")
    assert read_entry_dir(tmp_path)[0].kind == "empty"


def test_default_cache_dir_follows_triton_knobs() -> None:
    assert default_cache_dir({"TRITON_CACHE_DIR": "/c"}) == Path("/c")
    assert default_cache_dir({"TRITON_HOME": "/h"}) == Path("/h/.triton/cache")


def test_reports_and_filters(synthetic_cache: Path) -> None:
    scan = scan_cache(synthetic_cache)
    assert [e.key for e in filter_entries(scan, target="hip")] == [KEY_B]
    assert [e.key for e in filter_entries(scan, target="cuda:80")] == [KEY_A]
    assert filter_entries(scan, target="cuda:90") == []
    assert len(filter_entries(scan, kernel="ax*")) == 2
    summary = totals(scan.entries)
    assert summary["by_target"]["cuda:80"]["entries"] == 1
    assert summary["by_kernel"]["axpy"]["entries"] == 2
    text = render_table_report(scan, scan.entries)
    assert "hip:gfx942" in text and "No incomplete or corrupt entries found." in text
    data = json.loads(render_json(scan, scan.entries))
    assert data["totals"]["entries"] == 2 and data["entries"][0]["stages"]
