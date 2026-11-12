from __future__ import annotations

import os
import time
from pathlib import Path

from triton_cachekit import cachetools as ct
from triton_cachekit.layout import scan_cache
from triton_cachekit.stats import compute_stats, render_text

from . import factories as f
from .conftest import HASH_A, HASH_B, KEY_A, KEY_B, make_entry

KEY_C = "MFRGGZDFMZTWQ2LKNNWG23TPOBYXE43UOV3HO6DZPIYTEMZUGU3A"


def age(directory: Path, days: float) -> None:
    stamp = time.time() - days * 86400
    for path in directory.iterdir():
        os.utime(path, (stamp, stamp))


def test_diff_reports_sides_content_and_hints(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    make_entry(a, KEY_A, HASH_A, "axpy")
    make_entry(b, KEY_A, HASH_A, "axpy")
    (b / KEY_A / "axpy.cubin").write_text("different binary")
    make_entry(a, KEY_B, HASH_B, "blend", source_path="/build/k.py")
    make_entry(
        b,
        KEY_C,
        "6" * 64,
        "blend",
        env={"TRITON_DISABLE_LINE_INFO": "true"},
        source_path="/srv/k.py",
    )
    diff = ct.diff_caches(scan_cache(a), scan_cache(b))
    assert diff.common == 1
    assert [e.key for e in diff.only_a] == [KEY_B] and [e.key for e in diff.only_b] == [KEY_C]
    assert diff.content_differs[0][2] == ["cubin: content differs"]
    assert "TRITON_DISABLE_LINE_INFO" in diff.hints[0] and "/srv/k.py:11" in diff.hints[0]
    text = ct.render_diff(diff, "a", "b")
    assert "1 entries in both, 1 only in A, 1 only in B, 1 in both with different content" in text
    assert ct.diff_caches(scan_cache(a), scan_cache(a)).to_dict()["only_a"] == []


def test_diff_admits_when_metadata_cannot_explain(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    make_entry(a, KEY_A, HASH_A, "blend")
    make_entry(b, KEY_B, HASH_B, "blend")
    diff = ct.diff_caches(scan_cache(a), scan_cache(b))
    assert "does not store" in diff.hints[0] and "explain" in diff.hints[0]


def test_prune_selectors_and_dry_run(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    old = make_entry(cache, KEY_A, HASH_A, "old")
    make_entry(cache, KEY_B, HASH_B, "amd", backend="hip", arch="gfx942", group="relative")
    make_entry(cache, KEY_C, "6" * 64, "broken", group="none")
    age(old, 40)

    plan = ct.plan_prune(scan_cache(cache), older_than=30 * 86400)
    assert [(e.name, r.split(" (")[0]) for e, r in plan.remove] == [("old", "older than cutoff")]
    assert "Dry run" in ct.render_prune(plan, applied=False)
    assert len(scan_cache(cache).entries) == 3  # planning deletes nothing

    plan = ct.plan_prune(scan_cache(cache), targets=["hip"], incomplete=True)
    assert sorted(e.name or "" for e, _ in plan.remove) == ["amd", "broken"]
    ct.apply_prune(plan)
    assert [e.name for e in scan_cache(cache).entries] == ["old"]
    assert not (cache / KEY_B).exists()


def test_prune_max_size_evicts_oldest_first(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    for index, (key, digest) in enumerate([(KEY_A, HASH_A), (KEY_B, HASH_B), (KEY_C, "6" * 64)]):
        age(make_entry(cache, key, digest, f"k{index}"), 10 - index)
    scan = scan_cache(cache)
    one = scan.entries[0].size
    plan = ct.plan_prune(scan, max_size=one + 10)
    assert sorted(e.name or "" for e, _ in plan.remove) == ["k0", "k1"]
    assert [e.name for e in plan.keep] == ["k2"]
    plan = ct.plan_prune(scan, triton_version_not="3.8.0", kernel="zzz*")
    assert len(plan.remove) == 3


def test_stats(tmp_path: Path) -> None:
    from triton_cachekit.recording import load_recording

    slow = f.compile_record(fn=f.fn("pkg.slow"))
    fast = f.compile_record(fn=f.fn("pkg.fast"))
    warm = f.compile_record(fn=f.fn("pkg.warm"))
    path = f.write_recording(
        tmp_path / "r.jsonl",
        [
            f.session(),
            slow,
            f.outcome(slow, hit=False, total=900_000),
            fast,
            f.outcome(fast, hit=False, total=100_000),
            warm,
            f.outcome(warm, hit=True, total=10_000),
        ],
    )
    stats = compute_stats([load_recording(path)])
    assert (stats.misses, stats.hits, stats.miss_us, stats.hit_us) == (2, 1, 1_000_000, 10_000)
    assert stats.kernels[0].kernel == "pkg.slow"
    assert stats.removable_us == 1_000_000 - 2 * 10_000
    assert abs(stats.removable_share - 980_000 / 1_010_000) < 1e-9
    text = render_text(stats)
    assert "would remove about 980.0 ms (97% of compile time)" in text
    assert stats.to_dict()["removable_is_upper_bound"] is False

    cold = compute_stats([load_recording(f.recording(tmp_path / "c.jsonl", [(slow, False)]))])
    assert cold.hit_cost_us is None and cold.removable_share == 1.0
    assert "upper bound" in render_text(cold)
