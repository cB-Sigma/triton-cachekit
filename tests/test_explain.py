"""Comparison logic on synthetic recordings: one controlled difference at a time."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from triton_cachekit import explain as ex
from triton_cachekit.recording import RecordingError, load_recording

from . import factories as f


def compare(
    tmp_path: Path,
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    sa: dict[str, Any] | None = None,
    sb: dict[str, Any] | None = None,
) -> ex.KernelReport:
    rec_a = load_recording(f.recording(tmp_path / "a.jsonl", [(a, False)], **(sa or {})))
    rec_b = load_recording(f.recording(tmp_path / "b.jsonl", [(b, False)], **(sb or {})))
    report = ex.compare_recordings(rec_a, rec_b)
    assert len(report.kernels) == 1
    return report.kernels[0]


def components(report: ex.KernelReport) -> list[str]:
    return [finding.component for finding in report.findings]


def test_identical_key_hit_is_not_reported(tmp_path: Path) -> None:
    a = f.compile_record()
    rec_a = load_recording(f.recording(tmp_path / "a.jsonl", [(a, False)]))
    rec_b = load_recording(f.recording(tmp_path / "b.jsonl", [(a, True)]))
    assert ex.compare_recordings(rec_a, rec_b).kernels == []
    (kernel,) = ex.compare_recordings(rec_a, rec_b, include_hits=True).kernels
    assert kernel.status == "hit"


def test_same_key_but_missed_points_at_cache_contents(tmp_path: Path) -> None:
    a = f.compile_record()
    b = f.compile_record(cache_dir="/elsewhere")
    assert a["hash"] == b["hash"]
    kernel = compare(tmp_path, a, b)
    assert kernel.status == "miss-same-key"
    assert components(kernel) == ["cache-contents"]
    assert "relocate" in kernel.findings[0].remedy and "/elsewhere" in kernel.findings[0].details[1]


def test_source_edit(tmp_path: Path) -> None:
    kernel = compare(
        tmp_path,
        f.compile_record(),
        f.compile_record(fn=f.fn(src="def blend(x):\n    return x + 1\n")),
    )
    assert kernel.status == "miss-key-differs"
    assert components(kernel) == ["source.text"]
    assert "+    return x + 1" in kernel.findings[0].details


def test_line_shift(tmp_path: Path) -> None:
    kernel = compare(tmp_path, f.compile_record(), f.compile_record(fn=f.fn(line=14, def_line=15)))
    assert components(kernel) == ["source.location.line"]
    assert "line 10 in A and line 14 in B" in kernel.findings[0].summary


def test_path_move_is_a_miss_only_when_the_path_is_in_the_key(tmp_path: Path) -> None:
    policy = ("file", "line", "def_line", "def_col")
    moved = f.fn(file="/other/site-packages/pkg/kernels.py")
    kernel = compare(
        tmp_path,
        f.compile_record(),
        f.compile_record(fn=moved),
        sa={"policy": policy},
        sb={"policy": policy},
    )
    assert components(kernel) == ["source.location.file"]
    assert "different path will not hit" in kernel.findings[0].remedy


def test_path_move_with_line_only_policy_is_just_a_note(tmp_path: Path) -> None:
    a = f.compile_record()
    b = f.compile_record()
    b["components"]["source"]["fn"]["file"] = "/other/kernels.py"  # not hashed in 3.9.0
    rec_a = load_recording(f.recording(tmp_path / "a.jsonl", [(a, False)]))
    rec_b = load_recording(f.recording(tmp_path / "b.jsonl", [(b, True)]))
    (kernel,) = ex.compare_recordings(rec_a, rec_b, include_hits=True).kernels
    assert kernel.status == "hit" and "not part of the key" in kernel.notes[0]


def test_dependency_edit_is_attributed_to_the_callee(tmp_path: Path) -> None:
    helper_a = f.fn("pkg.kernels.boost", src="def boost(x):\n    return x * 2\n", line=5)
    helper_b = f.fn("pkg.kernels.boost", src="def boost(x):\n    return x * 3\n", line=5)
    kernel = compare(
        tmp_path,
        f.compile_record(fn=f.fn(deps=[helper_a])),
        f.compile_record(fn=f.fn(deps=[helper_b])),
    )
    assert components(kernel) == ["source.dependency.text"]
    assert "pkg.kernels.blend -> pkg.kernels.boost" in kernel.findings[0].summary
    assert "caller" in kernel.findings[0].remedy


def test_unknown_dependencies_are_admitted(tmp_path: Path) -> None:
    a = f.fn(deps=None)
    b = f.fn(deps=None)
    b["cache_key"] = "f" * 64
    kernel = compare(tmp_path, f.compile_record(fn=a), f.compile_record(fn=b))
    assert components(kernel) == ["source.dependencies.unknown"]


def test_constexpr_global(tmp_path: Path) -> None:
    kernel = compare(
        tmp_path,
        f.compile_record(fn=f.fn(constexpr_globals=[["GAIN", "constexpr[3]"]])),
        f.compile_record(fn=f.fn(constexpr_globals=[["GAIN", "constexpr[4]"]])),
    )
    assert components(kernel) == ["source.globals"]
    assert kernel.findings[0].details == ["GAIN: constexpr[3] (A) vs constexpr[4] (B)"]


def test_env_var(tmp_path: Path) -> None:
    kernel = compare(
        tmp_path, f.compile_record(), f.compile_record(env={"TRITON_DISABLE_LINE_INFO": "true"})
    )
    assert components(kernel) == ["env"]
    assert kernel.findings[0].details == ["TRITON_DISABLE_LINE_INFO: unset in A, 'true' in B"]


def test_option(tmp_path: Path) -> None:
    kernel = compare(
        tmp_path,
        f.compile_record(),
        f.compile_record(
            options__values={"num_warps": 8, "num_stages": 3, "enable_fp_fusion": False}
        ),
    )
    assert components(kernel) == ["options"]
    assert kernel.findings[0].details == ["num_warps: 4 (A) vs 8 (B)"]


def test_extern_lib_content(tmp_path: Path) -> None:
    libs = {"libdevice": {"path": "/elsewhere/libdevice.10.bc", "sha256": "e" * 64}}
    kernel = compare(tmp_path, f.compile_record(), f.compile_record(options__extern_libs=libs))
    assert components(kernel) == ["options"]
    assert "extern_libs[libdevice]: file content differs" in kernel.findings[0].details[0]


def test_target(tmp_path: Path) -> None:
    backend = {
        "hash": f.PTXAS[:-2] + "90",
        "class": "CUDABackend",
        "target": {"backend": "cuda", "arch": 90, "warp_size": 32},
    }
    kernel = compare(tmp_path, f.compile_record(), f.compile_record(backend=backend))
    assert components(kernel) == ["backend.target"]
    assert "cuda:80:32 (A) vs cuda:90:32 (B)" in kernel.findings[0].summary


def test_toolchain(tmp_path: Path) -> None:
    backend = {
        "hash": f.PTXAS.replace("12.9", "13.0"),
        "class": "CUDABackend",
        "target": {"backend": "cuda", "arch": 80, "warp_size": 32},
    }
    kernel = compare(tmp_path, f.compile_record(), f.compile_record(backend=backend))
    assert components(kernel) == ["backend.toolchain"]
    assert any("13.0" in line for line in kernel.findings[0].details)


def test_triton_version_and_files(tmp_path: Path) -> None:
    b = f.compile_record(triton={"version": "3.10.0", "key_digest": "u" * 64})
    kernel = compare(tmp_path, f.compile_record(), b)
    assert components(kernel) == ["triton.version"]

    b = f.compile_record(triton={"version": "3.9.0", "key_digest": "u" * 64})
    other = f.session()["triton"] | {
        "files": {"runtime/cache.py": "a" * 64, "_C/libtriton.so": "c" * 64}
    }
    kernel = compare(tmp_path, f.compile_record(), b, sb={"triton": other})
    assert components(kernel) == ["triton.files"]
    assert kernel.findings[0].details == ["_C/libtriton.so: content differs"]


def test_specialisation_differences(tmp_path: Path) -> None:
    constants = [{"path": [2], "name": "BLOCK", "value": "128", "key": "128"}]
    kernel = compare(tmp_path, f.compile_record(), f.compile_record(source__constants=constants))
    assert components(kernel) == ["specialization.constants"]
    assert kernel.findings[0].details == ["BLOCK: 64 (A) vs 128 (B)"]

    signature = {"x": "*fp16", "n": "i32", "BLOCK": "constexpr"}
    kernel = compare(tmp_path, f.compile_record(), f.compile_record(source__signature=signature))
    assert components(kernel) == ["specialization.signature"]

    kernel = compare(tmp_path, f.compile_record(), f.compile_record(source__attrs="{}"))
    assert components(kernel) == ["specialization.attrs"]
    assert "divisible by 16" in kernel.findings[0].remedy


def test_best_candidate_is_the_closest_specialisation(tmp_path: Path) -> None:
    far = f.compile_record(
        source__constants=[{"path": [2], "name": "BLOCK", "value": "32", "key": "32"}],
        env={"X": "1"},
    )
    near = f.compile_record()
    b = f.compile_record(fn=f.fn(line=12))
    rec_a = load_recording(f.recording(tmp_path / "a.jsonl", [(far, False), (near, False)]))
    rec_b = load_recording(f.recording(tmp_path / "b.jsonl", [(b, False)]))
    (kernel,) = ex.compare_recordings(rec_a, rec_b).kernels
    assert kernel.matched_hash == near["hash"]
    assert components(kernel) == ["source.location.line"]


def test_new_kernel(tmp_path: Path) -> None:
    kernel = compare(tmp_path, f.compile_record(), f.compile_record(fn=f.fn("pkg.kernels.other")))
    assert kernel.status == "miss-new-kernel" and components(kernel) == ["coverage"]


def test_custom_pipeline(tmp_path: Path) -> None:
    kernel = compare(tmp_path, f.compile_record(), f.compile_record(custom_pipeline=True))
    assert components(kernel) == ["pipeline"]


def test_collisions(tmp_path: Path) -> None:
    rec = f.compile_record()

    def one(name: str, ir: str, no_loc: str) -> Path:
        return f.write_recording(
            tmp_path / name, [f.session(), rec, f.outcome(rec, hit=False, ir=ir, ir_no_loc=no_loc)]
        )

    real = [
        load_recording(one("a.jsonl", "1" * 64, "x" * 64)),
        load_recording(one("b.jsonl", "2" * 64, "y" * 64)),
    ]
    (collision,) = ex.find_collisions(real)
    assert collision.severity == "collision" and "#12429" in collision.summary

    debug = [
        load_recording(one("c.jsonl", "1" * 64, "x" * 64)),
        load_recording(one("d.jsonl", "2" * 64, "x" * 64)),
    ]
    (collision,) = ex.find_collisions(debug)
    assert collision.severity == "debug-info"

    same = [
        load_recording(one("e.jsonl", "1" * 64, "x" * 64)),
        load_recording(one("f.jsonl", "1" * 64, "x" * 64)),
    ]
    assert ex.find_collisions(same) == []
    assert "No key collisions" in ex.render_collisions([])


def test_rendering_and_json(tmp_path: Path) -> None:
    rec_a = load_recording(f.recording(tmp_path / "a.jsonl", [(f.compile_record(), False)]))
    b1 = f.compile_record(env={"TRITON_DISABLE_LINE_INFO": "true"})
    b2 = f.compile_record(fn=f.fn("pkg.kernels.other"), env={"TRITON_DISABLE_LINE_INFO": "true"})
    rec_b = load_recording(f.recording(tmp_path / "b.jsonl", [(b1, False), (b2, False)]))
    report = ex.compare_recordings(rec_a, rec_b)
    text = ex.render_text(report)
    assert "Environment differences that affect every kernel:" in text
    assert "2 compilation(s) in B missed the cache" in text
    assert "(see above)" in text and "MISS (not compiled in A)" in text
    data = json.loads(ex.render_json(report))
    assert data["summary"] == {"miss-key-differs": 1, "miss-new-kernel": 1}
    assert data["environment"][0]["component"] == "env"


def test_recording_loader_tolerates_truncation_and_rejects_other_files(tmp_path: Path) -> None:
    path = f.recording(tmp_path / "a.jsonl", [(f.compile_record(), False)])
    with open(path, "a") as handle:
        handle.write('{"type": "compile", "pid": 1, "ha')
    loaded = load_recording(path)
    assert len(loaded.compiles) == 1 and "unparsable" in loaded.warnings[0]

    other = tmp_path / "x.jsonl"
    other.write_text('{"hello": 1}\n')
    with pytest.raises(RecordingError):
        load_recording(other)
    f.write_recording(other, [f.session(schema=99)])
    with pytest.raises(RecordingError, match="schema"):
        load_recording(other)
    with pytest.raises(RecordingError):
        load_recording(tmp_path / "missing.jsonl")


def test_location_policy_detection() -> None:
    """Which location fields are hashed is read off the installed Triton's shape."""
    from types import SimpleNamespace

    from triton_cachekit.recorder import detect_location_policy

    class FinderMain:
        def __init__(self, name, globals, nonlocals, src, source_location) -> None:  # type: ignore[no-untyped-def]
            pass

    class FinderOld:
        def __init__(self, name, globals, nonlocals, src) -> None:  # type: ignore[no-untyped-def]
            pass

    class Callable39:
        @property
        def cache_key(self) -> str:
            return "x" + str(self.starting_line_number)  # type: ignore[attr-defined]

    class CallableOther:
        @property
        def cache_key(self) -> str:
            return "x"

    main = SimpleNamespace(DependenciesFinder=FinderMain, JITCallable=Callable39)
    assert detect_location_policy(main)[0] == ["file", "line", "def_line", "def_col"]  # type: ignore[arg-type]
    old = SimpleNamespace(DependenciesFinder=FinderOld, JITCallable=Callable39)
    assert detect_location_policy(old)[0] == ["line"]  # type: ignore[arg-type]
    other = SimpleNamespace(DependenciesFinder=FinderOld, JITCallable=CallableOther)
    assert detect_location_policy(other)[0] == []  # type: ignore[arg-type]
