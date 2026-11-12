"""Integration tests against the installed Triton (fake driver, no GPU needed).

Each ``explain`` test changes exactly one thing between two recorded runs that
share a cache directory and asserts that the right key component is named.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from triton_cachekit import explain as ex
from triton_cachekit.layout import scan_cache
from triton_cachekit.recording import load_recording

from .conftest import BLEND_ARGS, KERNEL_SOURCE, _cli_env, compile_once

pytestmark = pytest.mark.triton

FP = {"TRITON_DEFAULT_FP_FUSION": "0"}


@pytest.fixture(scope="module")
def baseline(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("baseline")
    job, recording = compile_once(root / "app" / "kernels.py", root / "cache", root / "a.rec")
    assert job.result["specs"][0]["status"] == "compiled"
    return {
        "root": root,
        "cache": root / "cache",
        "recording": recording,
        "source": root / "app" / "kernels.py",
    }


def rerun(baseline: dict[str, Any], tmp_path: Path, **changes: Any) -> ex.ExplainReport:
    """Run again against the baseline's cache with ``changes`` applied, then explain."""
    source_file = changes.pop("source_file", baseline["source"])
    changes.setdefault("source", None if source_file == baseline["source"] else KERNEL_SOURCE)
    if changes["source"] is not None and source_file == baseline["source"]:
        # never edit the shared baseline file: write the edited copy elsewhere with the same name
        source_file = tmp_path / "edited" / "kernels.py"
    _, recording = compile_once(source_file, baseline["cache"], tmp_path / "b.rec", **changes)
    return ex.compare_recordings(baseline["recording"], recording, include_hits=True)


def only(report: ex.ExplainReport) -> ex.KernelReport:
    assert len(report.kernels) == 1, [k.kernel for k in report.kernels]
    return report.kernels[0]


def components(report: ex.ExplainReport) -> set[str]:
    return {finding.component for finding in only(report).findings}


def path_in_key(baseline: dict[str, Any]) -> bool:
    return "file" in baseline["recording"].session["location_in_key"]


# ------------------------------------------------------------------------ record
def test_recording_captures_every_key_component(baseline: dict[str, Any]) -> None:
    recording = baseline["recording"]
    session = recording.session
    assert session["triton"]["files_verified"] is True
    assert any(name.endswith("cache.py") for name in session["triton"]["files"])
    assert any("libtriton" in name for name in session["triton"]["files"])
    (item,) = recording.compiles
    assert item.kernel == "kernels.blend" and item.cache_hit is False
    comp = item.components
    assert set(comp) == {"triton", "source", "backend", "options", "env"}
    fn = comp["source"]["fn"]
    assert fn["file"].endswith("kernels.py") and fn["line"] == 12
    assert [d["name"] for d in fn["deps"]] == ["kernels.boost"]
    assert fn["constexpr_globals"] == [["GAIN", "constexpr[3]"]]
    assert comp["source"]["signature"]["a_ptr"] == "*fp32"
    assert comp["source"]["constants"][0]["name"] == "BLOCK"
    assert comp["backend"]["target"] == {"backend": "cuda", "arch": 80, "warp_size": 32}
    assert comp["env"] == {"TRITON_DEFAULT_FP_FUSION": "false"}
    assert comp["options"]["extern_libs"]["libdevice"]["sha256"]
    assert json.loads(item.specialization_data)["name"] == "kernels.blend"
    # the recorded hash is the directory Triton really wrote
    (entry,) = scan_cache(baseline["cache"]).entries
    assert entry.key == item.entry and entry.key_hex == item.hash
    assert set(item.outcome["artifacts"]) >= {"source", "ttir", "cubin", "json"}
    assert set(item.stage_us) >= {"frontend", "ttir", "ttgir", "llir", "ptx", "cubin"}


# ----------------------------------------------------------------------- explain
def test_identical_rerun_hits(baseline: dict[str, Any], tmp_path: Path) -> None:
    assert only(rerun(baseline, tmp_path)).status == "hit"


def test_explain_source_edit(baseline: dict[str, Any], tmp_path: Path) -> None:
    edited = KERNEL_SOURCE.replace("boost(a) * weight + b", "boost(a) * weight - b")
    report = rerun(baseline, tmp_path, source=edited)
    assert only(report).status == "miss-key-differs"
    expected = {"source.text"} | ({"source.location.file"} if path_in_key(baseline) else set())
    assert components(report) == expected
    assert any("weight - b" in line for line in only(report).findings[0].details)


def test_explain_edited_dependency(baseline: dict[str, Any], tmp_path: Path) -> None:
    edited = KERNEL_SOURCE.replace("return x * GAIN", "return x + GAIN")
    report = rerun(baseline, tmp_path, source=edited)
    assert "source.dependency.text" in components(report)
    assert "source.text" not in components(report)
    assert "kernels.blend -> kernels.boost" in only(report).findings[0].summary


def test_explain_constexpr_global(baseline: dict[str, Any], tmp_path: Path) -> None:
    edited = KERNEL_SOURCE.replace("GAIN = tl.constexpr(3)", "GAIN = tl.constexpr(4)")
    report = rerun(baseline, tmp_path, source=edited)
    assert {"source.globals", "source.dependency.globals"} <= components(report)


def test_explain_line_shift(baseline: dict[str, Any], tmp_path: Path) -> None:
    report = rerun(baseline, tmp_path, source="# a new first line\n" + KERNEL_SOURCE)
    assert {"source.location.line", "source.dependency.location.line"} <= components(report)
    assert "source.text" not in components(report)
    assert "line 12 in A and line 13 in B" in only(report).findings[0].summary


def test_explain_file_path_move(baseline: dict[str, Any], tmp_path: Path) -> None:
    """Same source at another path: a hit on 3.8/3.9, a miss where the path is hashed."""
    report = rerun(baseline, tmp_path, source_file=tmp_path / "moved" / "kernels.py")
    if path_in_key(baseline):
        assert only(report).status == "miss-key-differs"
        assert "source.location.file" in components(report)
    else:
        assert only(report).status == "hit"
        assert "not part of the key" in only(report).notes[0]


def test_explain_env_var(baseline: dict[str, Any], tmp_path: Path) -> None:
    report = rerun(baseline, tmp_path, env={"TRITON_DISABLE_LINE_INFO": "1"})
    assert components(report) == {"env"}
    assert only(report).findings[0].details == ["TRITON_DISABLE_LINE_INFO: unset in A, 'true' in B"]
    assert report.global_findings[0].component == "env"


def test_explain_option(baseline: dict[str, Any], tmp_path: Path) -> None:
    report = rerun(baseline, tmp_path, kwargs={"BLOCK": 64, "num_warps": 8})
    assert components(report) == {"options"}
    assert only(report).findings[0].details == ["num_warps: 4 (A) vs 8 (B)"]


def test_explain_option_default_from_environment(baseline: dict[str, Any], tmp_path: Path) -> None:
    report = rerun(baseline, tmp_path, env={"TRITON_DEFAULT_FP_FUSION": "1"})
    assert components(report) == {"env", "options"}


def test_explain_target(baseline: dict[str, Any], tmp_path: Path) -> None:
    report = rerun(baseline, tmp_path, target="cuda:90:32")
    assert "backend.target" in components(report)
    assert "cuda:80:32 (A) vs cuda:90:32 (B)" in only(report).findings[0].summary


def test_explain_specialisations(baseline: dict[str, Any], tmp_path: Path) -> None:
    report = rerun(baseline, tmp_path / "c", kwargs={"BLOCK": 128})
    assert components(report) == {"specialization.constants"}
    assert only(report).findings[0].details == ["BLOCK: 64 (A) vs 128 (B)"]

    half = ["tensor:float16", "tensor:float16", "tensor:float16", 1024, 0.25]
    report = rerun(baseline, tmp_path / "s", args=half)
    assert components(report) == {"specialization.signature"}

    unaligned = ["tensor:float32:unaligned", *BLEND_ARGS[1:]]
    report = rerun(baseline, tmp_path / "a", args=unaligned)
    assert components(report) == {"specialization.attrs"}


def test_amd_target_compiles_and_is_inspectable(tmp_path: Path) -> None:
    _, recording = compile_once(
        tmp_path / "k.py", tmp_path / "cache", tmp_path / "rec", target="hip:gfx942:64"
    )
    (entry,) = scan_cache(tmp_path / "cache").entries
    assert entry.target == "hip:gfx942" and entry.status == "ok"
    assert list(entry.stages)[-2:] == ["amdgcn", "hsaco"]
    assert recording.compiles[0].components["backend"]["class"] == "HIPBackend"


# --------------------------------------------------------------------- collisions
def test_same_kernel_at_two_paths_shares_a_key_with_different_debug_info(tmp_path: Path) -> None:
    recordings = []
    for side in "ab":
        _, recording = compile_once(
            tmp_path / side / "kernels.py", tmp_path / f"cache_{side}", tmp_path / f"{side}.rec"
        )
        recordings.append(recording)
    if "file" in recordings[0].session["location_in_key"]:
        assert recordings[0].compiles[0].hash != recordings[1].compiles[0].hash
        assert ex.find_collisions(recordings) == []
    else:
        (collision,) = ex.find_collisions(recordings)
        assert collision.severity == "debug-info"


def test_real_collision_is_detected(tmp_path: Path) -> None:
    """A constexpr read through a module attribute is not hashed (upstream #12429)."""
    kernel = textwrap.dedent(
        """\
        import settings
        import triton
        import triton.language as tl


        @triton.jit
        def times(x_ptr, out_ptr, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            tl.store(out_ptr + offs, tl.load(x_ptr + offs) * settings.FACTOR)
        """
    )
    recordings = []
    for side, factor in (("a", 2), ("b", 5)):
        (tmp_path / side).mkdir()
        (tmp_path / side / "settings.py").write_text(
            f"import triton.language as tl\nFACTOR = tl.constexpr({factor})\n"
        )
        _, recording = compile_once(
            tmp_path / side / "kern.py",
            tmp_path / f"cache_{side}",
            tmp_path / f"{side}.rec",
            name="times",
            args=["tensor:float32", "tensor:float32"],
            source=kernel,
        )
        recordings.append(recording)
    if recordings[0].compiles[0].hash != recordings[1].compiles[0].hash:
        pytest.skip("this Triton hashes module-attribute constexprs: #12429 does not reproduce")
    (collision,) = ex.find_collisions(recordings)
    assert collision.severity == "collision"


# ------------------------------------------------------------------------ inspect
def test_inspect_real_entries(demo: Any) -> None:
    import triton

    scan = scan_cache(demo.cache_dir)
    assert len(scan.entries) == 7  # 4 from the build, 3 recompiled by the "server"
    recorded = {
        c.entry: c for c in [*demo.build_recording.compiles, *demo.server_recording.compiles]
    }
    for entry in scan.entries:
        assert entry.kind == "kernel" and entry.status == "ok", entry.problems
        assert entry.triton_version == triton.__version__
        assert entry.target == "cuda:80" and entry.warp_size == 32
        assert list(entry.stages) == ["source", "ttir", "ttgir", "llir", "ptx", "cubin"]
        assert entry.stages["cubin"] > 0
        assert entry.key_hex == entry.metadata["hash"] == recorded[entry.key].hash
        assert entry.env == {"TRITON_DEFAULT_FP_FUSION": "false"}
        assert entry.source_file and entry.source_file.endswith("demo_kernels.py")
        assert entry.group_format in ("absolute", "relative")
        assert set(entry.group_children) == {
            f.name for f in entry.files if not f.name.startswith("__grp__")
        }
    lines = {e.name: e.source_line for e in scan.entries if "build" in (e.source_file or "")}
    assert lines["scale"] == 13 and lines["normalize"] == 21


def test_inspect_detects_damage_to_real_entries(demo: Any, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    shutil.copytree(demo.cache_dir, cache)
    from triton_cachekit.pack import relocate_cache

    relocate_cache(cache)
    entries = sorted(scan_cache(cache).entries, key=lambda e: e.key)
    assert {e.status for e in entries} == {"ok"}
    first, second, third, fourth = entries[:4]
    (first.path / f"__grp__{first.name}.json").unlink()
    (second.path / f"{second.name}.cubin").unlink()
    (third.path / f"{third.name}.json").write_text("{ truncated")
    (fourth.path / "tmp.pid_4242_deadbeef").mkdir()
    statuses = {e.key: e.status for e in scan_cache(cache).entries}
    assert statuses[first.key] == "incomplete"
    assert statuses[second.key] == "incomplete"
    assert statuses[third.key] == "corrupt"
    assert statuses[fourth.key] == "ok"
    assert sum(1 for s in statuses.values() if s == "ok") == 4


def test_explain_demo_names_three_different_causes(demo: Any) -> None:
    report = ex.compare_recordings(demo.build_recording, demo.server_recording, include_hits=True)
    by_kernel = {k.kernel: k for k in report.kernels}
    assert by_kernel["demo_kernels.scale"].status == "hit"
    assert [f.component for f in by_kernel["demo_kernels.normalize"].findings] == [
        "source.dependency.text"
    ]
    assert [f.component for f in by_kernel["demo_kernels.row_sum"].findings] == [
        "specialization.constants"
    ]
    assert [f.component for f in by_kernel["demo_kernels.saxpy"].findings] == [
        "source.location.line"
    ]
    assert len(report.misses) == 3


# ------------------------------------------------------------ prewarm / verify-hit
MANIFEST = """\
[prewarm]
target = "cuda:80"
env = { TRITON_DEFAULT_FP_FUSION = "0" }

[[kernel]]
file = "kernels.py"
name = "blend"

  [[kernel.spec]]
  args = ["tensor:float32", "tensor:float32", "tensor:float32", 1024, 0.25]
  kwargs = { BLOCK = 64 }

  [[kernel.spec]]
  args = ["tensor:float16", "tensor:float16", "tensor:float16", 1024, 0.25]
  kwargs = { BLOCK = 128, num_warps = 2 }
"""


@pytest.fixture(scope="module")
def prewarmed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("prewarm")
    (root / "kernels.py").write_text(KERNEL_SOURCE)
    (root / "manifest.toml").write_text(MANIFEST)
    result = run_cli(
        "prewarm",
        "--manifest",
        root / "manifest.toml",
        "--cache-dir",
        root / "cache",
        "-o",
        root / "out.json",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "prewarm: 2 compiled" in result.stdout
    assert "NOT VERIFIED ON A REAL GPU" in result.stdout
    return {"root": root, "cache": root / "cache", "manifest": root / "out.json"}


def run_cli(*args: Any, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "triton_cachekit", *map(str, args)],
        capture_output=True,
        text=True,
        env=_cli_env(env),
        timeout=600,
        check=False,
    )


def test_prewarm_writes_complete_entries_and_a_manifest(prewarmed: dict[str, Path]) -> None:
    entries = scan_cache(prewarmed["cache"]).entries
    assert len(entries) == 2 and {e.status for e in entries} == {"ok"}
    assert all(e.stages.get("cubin", 0) > 0 for e in entries)
    manifest = json.loads(prewarmed["manifest"].read_text())
    assert manifest["target"] == "cuda:80:32" and manifest["driver"] == "fake"
    assert manifest["verified_on_real_gpu"] is False and manifest["caveats"]
    assert manifest["summary"]["compiled"] == 2
    assert {e["entry"] for e in manifest["entries"]} == {e.key for e in entries}
    assert manifest["fingerprint"]["triton"]["files_verified"] is True
    assert load_recording(manifest["recording"]).compiles


def test_second_prewarm_in_a_fresh_process_hits(prewarmed: dict[str, Path], tmp_path: Path) -> None:
    root = prewarmed["root"]
    result = run_cli(
        "prewarm",
        "--manifest",
        root / "manifest.toml",
        "--cache-dir",
        root / "cache",
        "-o",
        tmp_path / "again.json",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "prewarm: 2 hit" in result.stdout
    again = json.loads((tmp_path / "again.json").read_text())
    first = json.loads(prewarmed["manifest"].read_text())
    assert [e["hash"] for e in again["entries"]] == [e["hash"] for e in first["entries"]]
    assert {e["status"] for e in again["entries"]} == {"hit"}


def test_verify_hit_simulation(prewarmed: dict[str, Path]) -> None:
    ok = run_cli(
        "verify-hit", "--manifest", prewarmed["manifest"], "--cache-dir", prewarmed["cache"],
        "--fake-driver", "--env", "TRITON_DEFAULT_FP_FUSION=0",
    )  # fmt: skip
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "SIMULATION" in ok.stdout and "verify-hit: 2 hit, 0 miss, 0 error" in ok.stdout


def test_verify_hit_explains_an_environment_mismatch(
    prewarmed: dict[str, Path], tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    shutil.copytree(prewarmed["cache"], cache)
    run_cli("relocate", cache)
    bad = run_cli(
        "verify-hit",
        "--manifest",
        prewarmed["manifest"],
        "--cache-dir",
        cache,
        "--fake-driver",
        "--format",
        "json",
    )
    assert bad.returncode == 1, bad.stdout + bad.stderr
    data = json.loads(bad.stdout)
    assert data["ok"] is False and data["simulated"] is True
    assert data["environment"][0]["component"] == "env"
    assert {row["status"] for row in data["kernels"]} == {"miss"}
    found = {f["component"] for k in data["explanation"]["kernels"] for f in k["findings"]}
    assert "env" in found


def test_verify_hit_without_a_gpu_refuses_to_claim_anything(prewarmed: dict[str, Path]) -> None:
    import triton

    try:
        triton.runtime.driver.active.get_current_target()
    except Exception:
        pass
    else:  # pragma: no cover - only on a GPU host
        pytest.skip("a real GPU driver is active")
    result = run_cli(
        "verify-hit", "--manifest", prewarmed["manifest"], "--cache-dir", prewarmed["cache"]
    )
    assert result.returncode == 2
    assert "no usable GPU driver" in result.stdout


def test_moved_cache_misses_until_relocated(prewarmed: dict[str, Path], tmp_path: Path) -> None:
    """Triton 3.8/3.9 store absolute paths in group files (upstream #11999)."""
    moved = tmp_path / "moved"
    shutil.copytree(prewarmed["cache"], moved)
    # make the original paths unreachable for the duration of the check
    hidden = prewarmed["cache"].with_name("cache.hidden")
    prewarmed["cache"].rename(hidden)
    try:
        verify = (
            "verify-hit",
            "--manifest",
            prewarmed["manifest"],
            "--cache-dir",
            moved,
            "--fake-driver",
            "--env",
            "TRITON_DEFAULT_FP_FUSION=0",
        )
        absolute = scan_cache(moved).entries[0].group_format == "absolute"
        if absolute:
            assert {e.status for e in scan_cache(moved).entries} == {"stale-paths"}
            fresh = tmp_path / "probe"
            shutil.copytree(moved, fresh)
            before = run_cli(*verify[:4], fresh, *verify[5:])
            assert before.returncode == 1 and "no loadable entry" in before.stdout
            relocated = run_cli("relocate", moved)
            assert "Rewrote group files of 2 entries" in relocated.stdout
        after = run_cli(*verify)
        assert after.returncode == 0, after.stdout + after.stderr
        assert "verify-hit: 2 hit" in after.stdout
    finally:
        hidden.rename(prewarmed["cache"])


def test_pack_unpack_then_hit(prewarmed: dict[str, Path], tmp_path: Path) -> None:
    archive = tmp_path / "cache.tar.gz"
    packed = run_cli("pack", prewarmed["cache"], "-o", archive)
    assert packed.returncode == 0 and "Packed 2 entries" in packed.stdout
    dest = tmp_path / "srv" / "cache"
    unpacked = run_cli("unpack", archive, dest)
    assert unpacked.returncode == 0, unpacked.stderr
    assert "matches the archive" in unpacked.stdout
    hidden = prewarmed["cache"].with_name("cache.hidden2")
    prewarmed["cache"].rename(hidden)
    try:
        result = run_cli(
            "verify-hit", "--manifest", prewarmed["manifest"], "--cache-dir", dest,
            "--fake-driver", "--env", "TRITON_DEFAULT_FP_FUSION=0",
        )  # fmt: skip
    finally:
        hidden.rename(prewarmed["cache"])
    assert result.returncode == 0, result.stdout + result.stderr


def test_prewarm_precheck_against_server_fingerprint(
    prewarmed: dict[str, Path], tmp_path: Path
) -> None:
    root = prewarmed["root"]
    fingerprint = run_cli("fingerprint", env={"TRITON_DEFAULT_FP_FUSION": "0"})
    assert fingerprint.returncode == 0
    data = json.loads(fingerprint.stdout)
    assert data["triton"]["files_verified"] is True
    assert data["env"]["cache_invalidating"] == {"TRITON_DEFAULT_FP_FUSION": "false"}
    (tmp_path / "server.json").write_text(json.dumps(data))
    common = ("prewarm", "--manifest", root / "manifest.toml", "--cache-dir", tmp_path / "cache")
    ok = run_cli(*common, "--server-fingerprint", tmp_path / "server.json")
    assert ok.returncode == 0 and "Triton files and environment match" in ok.stdout

    data["env"]["cache_invalidating"]["TRITON_DISABLE_LINE_INFO"] = "true"
    data["triton"]["key_digest"] = "0" * 64
    name = next(n for n in data["triton"]["files"] if "libtriton" in n)
    data["triton"]["files"][name] = "0" * 64
    (tmp_path / "other.json").write_text(json.dumps(data))
    blocked = run_cli(
        *common[:3],
        "--cache-dir",
        tmp_path / "never",
        "--server-fingerprint",
        tmp_path / "other.json",
    )
    assert blocked.returncode == 1
    assert "Nothing was compiled" in blocked.stdout and "libtriton" in blocked.stdout
    assert "TRITON_DISABLE_LINE_INFO" in blocked.stdout
    assert not (tmp_path / "never").exists()
    compare = run_cli(
        "fingerprint", "--compare", tmp_path / "other.json", env={"TRITON_DEFAULT_FP_FUSION": "0"}
    )
    assert compare.returncode == 1 and "[triton.files]" in compare.stdout


def test_prewarm_reports_errors_and_timeouts(tmp_path: Path) -> None:
    (tmp_path / "kernels.py").write_text(KERNEL_SOURCE)
    (tmp_path / "slow.py").write_text("import time\ntime.sleep(60)\n")
    (tmp_path / "m.toml").write_text(
        MANIFEST
        + '\n[[kernel]]\nfile = "kernels.py"\nname = "does_not_exist"\n  [[kernel.spec]]\n  args = []\n'
        + '\n[[kernel]]\nfile = "slow.py"\nname = "never"\n  [[kernel.spec]]\n  args = []\n'
    )
    result = run_cli(
        "prewarm",
        "--manifest",
        tmp_path / "m.toml",
        "--cache-dir",
        tmp_path / "cache",
        "--timeout",
        "8",
        "-j",
        "3",
    )
    assert result.returncode == 1
    assert "2 compiled" in result.stdout and "2 error" in result.stdout
    assert "does_not_exist" in result.stdout and "timed out after 8s" in result.stdout
    bad = run_cli("prewarm", "--manifest", tmp_path / "kernels.py", "--cache-dir", tmp_path / "c2")
    assert bad.returncode == 2 and "invalid TOML" in bad.stderr


# ------------------------------------------------- record CLI and replay
APP = """\
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triton
from triton_cachekit.fakedriver import PrewarmTensor, install_fake_driver

install_fake_driver("cuda:80")       # stands in for the GPU this test does not have
import kernels

t = PrewarmTensor("float32")
for block in (64, 128):
    kernels.blend.warmup(t, t, t, 1024, 0.25, BLOCK=block, grid=(1,))
print("app done")
"""


def test_record_command_and_prewarm_from_recording(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    (app / "kernels.py").write_text(KERNEL_SOURCE)
    (app / "serve.py").write_text(APP)
    recording_file = tmp_path / "run.jsonl"
    env = {"TRITON_CACHE_DIR": str(tmp_path / "live-cache"), **FP}
    recorded = run_cli(
        "record", "-o", recording_file, "--", sys.executable, app / "serve.py", env=env
    )
    assert recorded.returncode == 0, recorded.stdout + recorded.stderr
    assert "app done" in recorded.stdout and "recording written to" in recorded.stderr
    recording = load_recording(recording_file)
    assert [c.describe_specialization() for c in recording.compiles] == ["BLOCK=64", "BLOCK=128"]
    assert all(c.cache_hit is False and c.specialization_data for c in recording.compiles)
    assert recording.compiles[0].components["source"]["fn"]["deps"][0]["name"] == "kernels.boost"

    stats = run_cli("stats", recording_file)
    assert "2 cache misses, 0 hits" in stats.stdout and "kernels.blend" in stats.stdout

    # replay the recording into an empty cache on the "build machine"
    out = tmp_path / "out.json"
    replay = run_cli(
        "prewarm",
        "--from-recording",
        recording_file,
        "--cache-dir",
        tmp_path / "built",
        "-o",
        out,
        env=FP,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert "prewarm: 2 compiled" in replay.stdout
    built = {e.key for e in scan_cache(tmp_path / "built").entries}
    assert built == {c.entry for c in recording.compiles}  # same keys as the real run

    # and the application, pointed at the prewarmed cache, now hits
    second = tmp_path / "second.jsonl"
    env = {"TRITON_CACHE_DIR": str(tmp_path / "built"), **FP}
    assert (
        run_cli("record", "-o", second, "--", sys.executable, app / "serve.py", env=env).returncode
        == 0
    )
    assert [c.cache_hit for c in load_recording(second).compiles] == [True, True]
    explained = run_cli("explain", recording_file, second)
    assert "No cache misses in B" in explained.stdout


def test_record_no_source_and_import_api(tmp_path: Path) -> None:
    (tmp_path / "kernels.py").write_text(KERNEL_SOURCE)
    script = tmp_path / "s.py"
    script.write_text("import triton_cachekit.record\n" + APP)
    out = tmp_path / "r.jsonl"
    env = {
        "TRITON_CACHE_DIR": str(tmp_path / "c"),
        "TRITON_CACHEKIT_RECORD": str(out),
        "TRITON_CACHEKIT_RECORD_SOURCE": "0",
        **FP,
    }
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=_cli_env(env),
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    recording = load_recording(out)
    assert len(recording.compiles) == 2
    assert "src" not in recording.compiles[0].components["source"]["fn"]
    assert recording.session["record_source"] is False


def test_recorder_fails_loudly_on_unexpected_internals(tmp_path: Path) -> None:
    script = tmp_path / "s.py"
    script.write_text(
        textwrap.dedent(
            """\
            import triton.runtime.cache as cache
            import triton.compiler.compiler as compiler
            import triton.runtime.jit as jit

            def get_cache_key(src, backend, options):      # a future, different signature
                return "x"

            cache.get_cache_key = compiler.get_cache_key = jit.get_cache_key = get_cache_key
            del jit.DependenciesFinder._update_hash
            import triton_cachekit.record
            """
        )
    )
    env = {"TRITON_CACHEKIT_RECORD": str(tmp_path / "r.jsonl")}
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=_cli_env(env),
        timeout=300,
        check=False,
    )
    assert result.returncode != 0
    assert "UnsupportedTritonError" in result.stderr
    assert "get_cache_key has parameters ['src', 'backend', 'options']" in result.stderr
    assert "_update_hash" in result.stderr and "Nothing was recorded" in result.stderr
    assert not (tmp_path / "r.jsonl").exists()


def test_demo_flags(tmp_path: Path) -> None:
    result = run_cli("stats", "--demo", "--demo-dir", tmp_path / "demo")
    assert result.returncode == 0, result.stderr
    assert "3 cache misses, 1 hits" in result.stdout
    assert (tmp_path / "demo" / "cache").is_dir()


def test_prewarm_compiles_every_autotune_config(tmp_path: Path) -> None:
    source = KERNEL_SOURCE.replace(
        "@triton.jit\ndef blend(",
        "@triton.autotune(configs=[triton.Config({}, num_warps=2), triton.Config({}, num_warps=8)], key=['n'])\n"
        "@triton.jit\ndef blend(",
    )
    job, recording = compile_once(
        tmp_path / "k.py", tmp_path / "cache", tmp_path / "rec", source=source
    )
    assert len(job.result["specs"][0]["compilations"]) == 2
    warps = sorted(c.components["options"]["values"]["num_warps"] for c in recording.compiles)
    assert warps == [2, 8]
    assert len(scan_cache(tmp_path / "cache").entries) == 2


def test_example_manifest(tmp_path: Path) -> None:
    examples = Path(__file__).resolve().parent.parent / "examples"
    result = run_cli(
        "prewarm", "--manifest", examples / "prewarm.toml", "--cache-dir", tmp_path / "cache"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "prewarm: 3 compiled" in result.stdout
    entries = scan_cache(tmp_path / "cache").entries
    assert sorted(e.name or "" for e in entries) == ["rms_norm_rows", "rms_norm_rows", "silu"]
    assert (tmp_path / "prewarm-manifest.json").is_file()
