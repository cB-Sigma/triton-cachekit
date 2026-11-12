"""Command line interface: ``triton-cachekit <command>``."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from . import __version__
from ._util import human_size, parse_age, parse_size, parse_target

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_USAGE = 2

_TRITON_MISSING = (
    "this command needs Triton, which is not importable in this Python environment "
    "({error}).\nInstall it next to triton-cachekit (pip install triton) and try again. "
    "The inspect, explain, stats, pack, unpack, relocate, diff and prune commands work "
    "without Triton."
)


class CliError(Exception):
    def __init__(self, message: str, code: int = EXIT_USAGE) -> None:
        super().__init__(message)
        self.code = code


def _require_triton() -> None:
    try:
        import triton  # noqa: F401
    except Exception as exc:
        raise CliError(_TRITON_MISSING.format(error=f"{type(exc).__name__}: {exc}")) from exc


def _cache_dir_arg(value: str | None) -> Path:
    from .layout import default_cache_dir

    path = Path(value) if value else default_cache_dir()
    if not path.is_dir():
        hint = "" if value else " (default location; pass a directory or set TRITON_CACHE_DIR)"
        raise CliError(f"cache directory not found: {path}{hint}")
    return path


@contextlib.contextmanager
def _demo_dir(keep: str | None) -> Iterator[Path]:
    if keep:
        path = Path(keep)
        if path.exists() and any(path.iterdir()):
            raise CliError(f"--demo-dir {path} must be empty or not exist")
        path.mkdir(parents=True, exist_ok=True)
        yield path
    else:
        with tempfile.TemporaryDirectory(prefix="triton-cachekit-demo-") as tmp:
            yield Path(tmp)


# ------------------------------------------------------------------- commands
def cmd_inspect(args: argparse.Namespace) -> int:
    from .inspect_report import filter_entries, render_json, render_table_report
    from .layout import scan_cache

    def run(cache_dir: Path) -> int:
        scan = scan_cache(cache_dir)
        entries = filter_entries(
            scan, kernel=args.kernel, target=args.target, only_problems=args.problems
        )
        if args.format == "json":
            print(render_json(scan, entries))
        else:
            print(
                render_table_report(scan, entries, wide=args.wide, show_location=args.locations),
                end="",
            )
        broken = any(e.kind == "kernel" and not e.loadable for e in entries)
        return EXIT_FINDINGS if args.strict and broken else EXIT_OK

    if args.demo:
        _require_triton()
        from .demo import run_demo

        with _demo_dir(args.demo_dir) as work:
            demo = run_demo(work, server=False)
            return run(demo.cache_dir)
    return run(_cache_dir_arg(args.cache_dir))


def cmd_explain(args: argparse.Namespace) -> int:
    from . import explain as ex
    from .recording import RecordingError, load_recording

    if args.demo:
        _require_triton()
        from .demo import run_demo

        with _demo_dir(args.demo_dir) as work:
            demo = run_demo(work)
            assert demo.server_recording is not None
            report = ex.compare_recordings(
                demo.build_recording, demo.server_recording, include_hits=args.all
            )
            print(
                ex.render_json(report) if args.format == "json" else ex.render_text(report), end=""
            )
            return EXIT_OK
    try:
        recordings = [load_recording(p) for p in args.recordings]
    except (RecordingError, OSError) as exc:
        raise CliError(str(exc)) from exc
    if args.collisions:
        if not recordings:
            raise CliError("explain --collisions needs at least one recording")
        collisions = ex.find_collisions(recordings)
        if args.format == "json":
            print(json.dumps([c.to_dict() for c in collisions], indent=2))
        else:
            print(ex.render_collisions(collisions), end="")
        return EXIT_FINDINGS if any(c.severity == "collision" for c in collisions) else EXIT_OK
    if len(recordings) != 2:
        raise CliError(
            "explain needs exactly two recordings: A (built the cache) and B (missed it)"
        )
    report = ex.compare_recordings(recordings[0], recordings[1], include_hits=args.all)
    print(ex.render_json(report) if args.format == "json" else ex.render_text(report), end="")
    return EXIT_FINDINGS if args.fail_on_miss and report.misses else EXIT_OK


def cmd_record(args: argparse.Namespace) -> int:
    from .recorder import ENV_OUTPUT, ENV_RECORD_SOURCE, resolve_output_path

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise CliError(
            "record: give the command to run after '--', e.g. record -o run.jsonl -- python app.py"
        )
    output = resolve_output_path(args.output)
    package_dir = Path(__file__).resolve().parent
    env = dict(os.environ)
    paths = [str(package_dir / "_autoload"), str(package_dir.parent)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env[ENV_OUTPUT] = output
    if args.no_source:
        env[ENV_RECORD_SOURCE] = "0"
    try:
        status = subprocess.call(command, env=env)
    except FileNotFoundError as exc:
        raise CliError(f"record: cannot run {command[0]!r}: {exc}") from exc
    print(f"triton-cachekit: recording written to {output}", file=sys.stderr)
    if "{pid}" not in output and not Path(output).exists():
        print(
            "triton-cachekit: no recording was produced - the command did not import Triton in a "
            "Python process that inherited PYTHONPATH.",
            file=sys.stderr,
        )
    return int(status)


def cmd_stats(args: argparse.Namespace) -> int:
    from . import stats as st
    from .recording import RecordingError, load_recording

    if args.demo:
        _require_triton()
        from .demo import run_demo

        with _demo_dir(args.demo_dir) as work:
            demo = run_demo(work)
            assert demo.server_recording is not None
            result = st.compute_stats([demo.server_recording])
            print(
                st.render_json(result)
                if args.format == "json"
                else st.render_text(result, top=args.top),
                end="",
            )
            return EXIT_OK
    if not args.recordings:
        raise CliError("stats needs at least one recording (or --demo)")
    try:
        recordings = [load_recording(p) for p in args.recordings]
    except (RecordingError, OSError) as exc:
        raise CliError(str(exc)) from exc
    result = st.compute_stats(recordings)
    print(
        st.render_json(result) if args.format == "json" else st.render_text(result, top=args.top),
        end="",
    )
    return EXIT_OK


def cmd_fingerprint(args: argparse.Namespace) -> int:
    from . import fingerprint as fp

    _require_triton()
    local = fp.collect()
    if args.compare:
        try:
            other = fp.load(args.compare)
        except (OSError, ValueError) as exc:
            raise CliError(str(exc)) from exc
        findings = fp.compare(local, other, a_name="this host", b_name=Path(args.compare).name)
        if not findings:
            print(
                "No difference in the checkable key inputs (Triton files, environment variables)."
            )
        for finding in findings:
            print(f"- [{finding.component}] {finding.summary}")
            for detail in finding.details:
                print(f"    {detail}")
            print(f"  why/fix: {finding.remedy}")
        caveat = fp.path_caveat(local)
        if caveat:
            print(f"note: {caveat}")
        return EXIT_FINDINGS if findings else EXIT_OK
    text = json.dumps(local, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"fingerprint written to {args.output}", file=sys.stderr)
    else:
        print(text)
    return EXIT_OK


def _parse_env(pairs: Sequence[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or not name:
            raise CliError(f"--env expects NAME=VALUE, got {pair!r}")
        env[name] = value
    return env


def cmd_prewarm(args: argparse.Namespace) -> int:
    from . import fingerprint as fp
    from . import prewarm as pw
    from .recording import RecordingError, load_recording

    _require_triton()
    extra_env = _parse_env(args.env)
    skipped: list[str] = []
    target = args.target
    with contextlib.ExitStack() as stack:
        if args.demo:
            from .demo import DEMO_TARGET, KERNELS_V1, _jobs

            work = stack.enter_context(_demo_dir(args.demo_dir))
            source = work / "build" / "demo_kernels.py"
            source.parent.mkdir(parents=True)
            source.write_text(KERNELS_V1, encoding="utf-8")
            jobs = _jobs(source, 128)
            target = DEMO_TARGET
            cache_dir = work / "cache"
            extra_env.setdefault("TRITON_DEFAULT_FP_FUSION", "0")
            output = work / "prewarm-manifest.json"
            record_dir = work / "prewarm.rec"
        else:
            if bool(args.manifest) == bool(args.from_recording):
                raise CliError(
                    "prewarm needs exactly one of --manifest or --from-recording (or --demo)"
                )
            if not args.cache_dir:
                raise CliError("prewarm needs --cache-dir (the directory to populate)")
            cache_dir = Path(args.cache_dir).resolve()
            if args.manifest:
                try:
                    manifest_target, manifest_env, jobs = pw.load_manifest(args.manifest)
                except (pw.ManifestError, OSError) as exc:
                    raise CliError(str(exc)) from exc
                target = target or manifest_target
                extra_env = {**manifest_env, **extra_env}
                if not target:
                    raise CliError(
                        "no target: pass --target or set [prewarm] target in the manifest"
                    )
            else:
                try:
                    recording = load_recording(args.from_recording)
                except (RecordingError, OSError) as exc:
                    raise CliError(str(exc)) from exc
                if target is None:
                    targets = sorted(
                        {
                            c.components.get("backend", {}).get("target", {}).get("backend", "?")
                            + ":"
                            + str(c.components.get("backend", {}).get("target", {}).get("arch"))
                            + ":"
                            + str(
                                c.components.get("backend", {}).get("target", {}).get("warp_size")
                            )
                            for c in recording.compiles
                        }
                    )
                    if len(targets) != 1:
                        raise CliError(
                            f"the recording contains {len(targets)} targets ({', '.join(targets)}): "
                            "choose one with --target"
                        )
                    target = targets[0]
                jobs, skipped = pw.jobs_from_recording(recording, target)
                if not jobs:
                    raise CliError(
                        "the recording contains no replayable JIT compilations for this target"
                    )
            output = (
                Path(args.output) if args.output else cache_dir.parent / "prewarm-manifest.json"
            )
            record_dir = (
                Path(args.recording_out) if args.recording_out else output.with_suffix(".rec")
            )
        assert target is not None
        try:
            parse_target(target)
        except ValueError as exc:
            raise CliError(str(exc)) from exc

        if args.server_fingerprint:
            try:
                server = fp.load(args.server_fingerprint)
            except (OSError, ValueError) as exc:
                raise CliError(str(exc)) from exc
            local = fp.collect()
            # compare with the environment the workers will really get
            local["env"]["cache_invalidating"] = {
                **local["env"]["cache_invalidating"],
                **{
                    k: _normalise_env(v) for k, v in extra_env.items() if k in _invalidating_names()
                },
            }
            findings = fp.compare(local, server, a_name="build", b_name="server")
            if findings:
                print(
                    "Pre-check against the server fingerprint FAILED - entries built now would miss:"
                )
                for finding in findings:
                    print(f"  - [{finding.component}] {finding.summary}")
                    for detail in finding.details[:20]:
                        print(f"      {detail}")
                    print(f"    why/fix: {finding.remedy}")
                if not args.force:
                    print("Nothing was compiled. Fix the differences or pass --force.")
                    return EXIT_FINDINGS
            else:
                print(
                    "Pre-check against the server fingerprint: Triton files and environment match."
                )
            caveat = fp.path_caveat(server)
            if caveat:
                print(f"note: {caveat}")

        print(
            f"Prewarming {sum(len(j.specs) for j in jobs)} specialisation(s) of {len(jobs)} kernel(s) "
            f"for {target} into {cache_dir}"
        )
        pw.run_jobs(
            jobs,
            target=target,
            fake_driver=True,
            cache_dir=cache_dir,
            record_dir=record_dir,
            timeout=args.timeout,
            extra_env=extra_env,
            parallel=args.jobs,
        )
        manifest = pw.build_output_manifest(
            jobs, target=target, cache_dir=cache_dir, recording=record_dir, extra_env=extra_env
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print()
        print(pw.render_results(jobs), end="")
        for reason in skipped:
            print(f"skipped: {reason}")
        print()
        print(f"Manifest:  {output}")
        print(f"Recording: {record_dir}")
        print()
        print("NOT VERIFIED ON A REAL GPU. These entries were compiled through a fake driver.")
        print("They are hit only if the GPU host computes the same keys; things that can differ:")
        for caveat in pw.CAVEATS[1:]:
            print(f"  * {caveat}")
        print(
            "Check it on the GPU host:  triton-cachekit verify-hit --manifest "
            f"{output.name} --cache-dir <cache>"
        )
        counts = manifest["summary"]
        return EXIT_FINDINGS if counts.get("error") else EXIT_OK


def _invalidating_names() -> set[str]:
    try:
        from triton._C.libtriton import get_cache_invalidating_env_vars
    except Exception:
        return set()
    # The binding only returns variables that are set; probe a superset instead.
    names = set(get_cache_invalidating_env_vars())
    return names | KNOWN_CACHE_INVALIDATING_ENV_VARS


def _normalise_env(value: str) -> str:
    """Mirror ``isEnvValueBool`` in Triton's GetEnv.h: on/true/1 and off/false/0."""
    lowered = value.strip().lower()
    if lowered in ("1", "true", "on"):
        return "true"
    if lowered in ("0", "false", "off"):
        return "false"
    return value


#: ``CACHE_INVALIDATING_ENV_VARS`` in include/triton/Tools/Sys/GetEnv.h (main @ 8a439de).
#: Used only to line up ``--env`` values with a server fingerprint before compiling.
KNOWN_CACHE_INVALIDATING_ENV_VARS: set[str] = {
    "AMDGCN_ENABLE_DUMP", "AMDGCN_USE_BUFFER_ATOMICS", "AMDGCN_USE_BUFFER_OPS", "DISABLE_LLVM_OPT",
    "DISABLE_MMA_V3", "DISABLE_MMA_V5", "DISABLE_PTXAS_OPT", "LLVM_IR_ENABLE_DUMP",
    "LLVM_ENABLE_TIMING", "LLVM_PASS_PLUGIN_PATH", "LLVM_EXTRACT_DI_LOCAL_VARIABLES",
    "MLIR_ENABLE_DIAGNOSTICS", "MLIR_ENABLE_DUMP", "MLIR_DUMP_PATH", "MLIR_ENABLE_TIMING",
    "MLIR_DISABLE_MULTITHREADING", "TRITON_DEFAULT_FP_FUSION", "TRITON_DISABLE_LINE_INFO",
    "TRITON_DUMP_MIR", "TRITON_SWAP_MIR", "TRITON_SWAP_MIR_ENABLE_MISCHED", "TRITON_ENABLE_LLVM_DEBUG",
    "TRITON_ALLOW_FP_FUSION", "TRITON_HIP_USE_ASYNC_COPY", "TRITON_HIP_USE_BLOCK_PINGPONG",
    "TRITON_HIP_USE_COEXEC_SCHEDULER", "TRITON_HIP_USE_EXPERT_SCHEDULING",
    "TRITON_HIP_USE_IN_THREAD_TRANSPOSE", "TRITON_LLVM_DEBUG_ONLY", "TRITON_ENABLE_ASAN",
    "TRITON_OVERRIDE_ARCH", "TRITON_USE_SLP_VECTORIZER", "USE_IR_LOC", "NVPTX_ENABLE_DUMP",
    "ALLOW_LHS_TMEM_LAYOUT_CONVERSION", "TRITON_F32_DEFAULT", "TRITON_ENABLE_EXPERIMENTAL_CONSAN",
    "TRITON_CONSAN_INIT_ALLOCATIONS", "TRITON_PLUGIN_PATHS", "TRITON_PLUGIN_VERSION_CHECK",
    "TRITON_PARTITION_SCHEDULING_ENABLE_DUMP_DOT", "TRITON_PARTITION_SCHEDULING_DUMP_DATA_ONLY",
    "TRITON_PARTITION_SCHEDULING_DUMP_LOOP_ONLY",
}  # fmt: skip


def cmd_verify_hit(args: argparse.Namespace) -> int:
    from . import explain as ex
    from . import fingerprint as fp
    from . import prewarm as pw
    from .recording import RecordingError, load_recording

    _require_triton()
    try:
        manifest = pw.load_output_manifest(args.manifest)
    except (pw.ManifestError, OSError, ValueError) as exc:
        raise CliError(str(exc)) from exc
    jobs = pw.jobs_from_output_manifest(manifest)
    cache_dir = _cache_dir_arg(args.cache_dir)
    with contextlib.ExitStack() as stack:
        if args.recording_out:
            record_dir = Path(args.recording_out)
        else:
            record_dir = Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="tck-verify-"))
            )
        pw.run_jobs(
            jobs,
            target=manifest.get("target"),
            fake_driver=args.fake_driver,
            cache_dir=cache_dir,
            record_dir=record_dir,
            timeout=args.timeout,
            extra_env=_parse_env(args.env),
            parallel=1,
        )
        local = next((j.result["fingerprint"] for j in jobs if j.result.get("fingerprint")), None)
        static = []
        if local and manifest.get("fingerprint"):
            static = fp.compare(
                manifest["fingerprint"], local, a_name="prewarm", b_name="this host"
            )
        actual_target = next((j.result.get("target") for j in jobs if j.result.get("target")), None)
        if actual_target and manifest.get("target") and actual_target != manifest["target"]:
            static.append(
                ex.Finding(
                    "backend.target",
                    f"the driver reports target {actual_target}, the cache was prewarmed for "
                    f"{manifest['target']}",
                )
            )
        fatal = next(
            (
                j.result["error"]
                for j in jobs
                if j.result.get("error") and not j.result.get("specs")
            ),
            None,
        )
        report = pw.VerifyReport(
            static_findings=static,
            jobs=jobs,
            expected={},
            rows=[] if fatal else pw.evaluate_verification(manifest, jobs),
            simulated=bool(args.fake_driver),
            target=actual_target,
            error=fatal,
        )
        explanation = None
        prewarm_recording = args.prewarm_recording or manifest.get("recording")
        if not fatal and any(r["status"] == "miss" for r in report.rows) and prewarm_recording:
            try:
                rec_a = load_recording(prewarm_recording)
                rec_b = load_recording(record_dir)
                explanation = ex.compare_recordings(rec_a, rec_b)
            except (RecordingError, OSError):
                explanation = None
        if args.format == "json":
            print(
                json.dumps(
                    {
                        "ok": report.ok,
                        "simulated": report.simulated,
                        "target": report.target,
                        "error": report.error,
                        "environment": [f.to_dict() for f in static],
                        "kernels": report.rows,
                        "explanation": explanation.to_dict() if explanation else None,
                    },
                    indent=2,
                )
            )
        else:
            print(pw.render_verification(report), end="")
            if explanation is not None:
                print()
                print("Why the misses happened (prewarm recording vs this host):")
                print(ex.render_text(explanation), end="")
            elif any(r["status"] == "miss" for r in report.rows):
                print(
                    "\nFor a component-level explanation pass the recording written by prewarm: "
                    "--prewarm-recording <dir>"
                )
        if report.error:
            return EXIT_USAGE
        return EXIT_OK if report.ok else EXIT_FINDINGS


def cmd_pack(args: argparse.Namespace) -> int:
    from . import pack as pk
    from .inspect_report import filter_entries
    from .layout import scan_cache

    cache_dir = _cache_dir_arg(args.cache_dir)
    scan = scan_cache(cache_dir)
    entries = filter_entries(scan, kernel=args.kernel, target=args.target)
    _, skipped = pk.select_entries(entries, include_incomplete=args.include_incomplete)
    fingerprint: dict[str, Any] | None = None
    try:
        from .fingerprint import collect

        fingerprint = collect()
    except Exception:
        fingerprint = None  # Triton not installed here: the manifest simply omits it
    output = Path(args.output)
    try:
        manifest = pk.pack(
            cache_dir,
            output,
            fmt=args.format,
            entries=entries,
            include_incomplete=args.include_incomplete,
            fingerprint=fingerprint,
        )
    except pk.PackError as exc:
        raise CliError(str(exc), EXIT_FINDINGS) from exc
    print(
        f"Packed {manifest['entry_count']} entries ({manifest['file_count']} files, "
        f"{human_size(manifest['total_size'])}) into {output} [{pk.detect_format(output, args.format)}]"
    )
    print(f"  Triton version(s): {', '.join(manifest['triton_versions']) or 'unknown'}")
    print(f"  Target(s):         {', '.join(manifest['targets']) or 'unknown'}")
    not_packed = [e for e in skipped if e.kind == "kernel"]
    if not_packed:
        print(
            f"  Skipped {len(not_packed)} incomplete/corrupt kernel entries (see 'inspect --problems')."
        )
    others = [e for e in skipped if e.kind in ("autotune", "native")]
    if others:
        print(f"  Skipped {len(others)} non-kernel entries (autotune results, launcher stubs).")
    print(f"  Relocation: {manifest['relocation']['note']}")
    print(f"  Portability: {manifest['portability']}")
    return EXIT_OK


def cmd_unpack(args: argparse.Namespace) -> int:
    from . import pack as pk

    archive = Path(args.archive)
    if not archive.exists():
        raise CliError(f"archive not found: {archive}")
    dest = Path(args.cache_dir)
    try:
        result = pk.unpack(archive, dest, relocate=not args.no_relocate, overwrite=args.overwrite)
    except (pk.PackError, OSError, ValueError) as exc:
        raise CliError(str(exc), EXIT_FINDINGS) from exc
    manifest = result["manifest"]
    print(
        f"Unpacked {manifest['entry_count']} entries into {dest}: {len(result['written'])} files written, "
        f"{len(result['skipped'])} already present, checksums verified."
    )
    if result["relocated"]:
        print(f"Rewrote group files of {len(result['relocated'])} entries for the new location.")
    elif manifest["relocation"]["group_files_absolute"] and args.no_relocate:
        print("warning: group files still hold the original absolute paths (--no-relocate).")
    try:
        import triton

        installed = str(triton.__version__)
    except Exception:
        installed = None
    if installed is None:
        print(
            "note: Triton is not installed here, so compatibility with the entries was not checked."
        )
    else:
        from .fingerprint import collect

        packed = manifest.get("packer_fingerprint") or {}
        local = collect()
        if manifest["triton_versions"] and installed not in manifest["triton_versions"]:
            print(
                f"warning: installed Triton is {installed}, the entries were written by "
                f"{', '.join(manifest['triton_versions'])}: they can never be hit here."
            )
        elif packed and packed.get("triton", {}).get("key_digest") != local["triton"]["key_digest"]:
            print(
                "warning: the Triton install here differs from the one that packed this archive "
                "(triton_key digest mismatch): the entries will not be hit unless they were "
                "written by an install identical to this one."
            )
        else:
            print(f"Installed Triton {installed} matches the archive.")
    return EXIT_OK


def cmd_relocate(args: argparse.Namespace) -> int:
    from . import pack as pk

    cache_dir = _cache_dir_arg(args.cache_dir)
    changed = pk.relocate_cache(cache_dir, dry_run=args.dry_run)
    verb = "Would rewrite" if args.dry_run else "Rewrote"
    print(f"{verb} group files of {len(changed)} entries to point at {cache_dir.resolve()}.")
    if not changed:
        print("Nothing to do: no group file with absolute paths pointing elsewhere.")
    return EXIT_OK


def cmd_diff(args: argparse.Namespace) -> int:
    from . import cachetools as ct
    from .layout import scan_cache

    a, b = _cache_dir_arg(args.a), _cache_dir_arg(args.b)
    diff = ct.diff_caches(scan_cache(a), scan_cache(b), compare_content=not args.no_content)
    if args.format == "json":
        print(json.dumps(diff.to_dict(), indent=2, default=str))
    else:
        print(ct.render_diff(diff, str(a), str(b)), end="")
    different = bool(diff.only_a or diff.only_b or diff.content_differs)
    return EXIT_FINDINGS if args.exit_code and different else EXIT_OK


def cmd_prune(args: argparse.Namespace) -> int:
    from . import cachetools as ct
    from .layout import scan_cache

    cache_dir = _cache_dir_arg(args.cache_dir)
    try:
        older = parse_age(args.older_than) if args.older_than else None
        max_size = parse_size(args.max_size) if args.max_size else None
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    if not any(
        [
            older,
            max_size is not None,
            args.target,
            args.kernel,
            args.incomplete,
            args.not_triton_version,
        ]
    ):
        raise CliError(
            "prune: nothing selected. Use --older-than, --max-size, --target, --kernel, "
            "--incomplete or --not-triton-version."
        )
    plan = ct.plan_prune(
        scan_cache(cache_dir),
        older_than=older,
        max_size=max_size,
        targets=args.target,
        kernel=args.kernel,
        incomplete=args.incomplete,
        triton_version_not=args.not_triton_version,
    )
    if args.yes:
        ct.apply_prune(plan)
    print(ct.render_prune(plan, applied=args.yes), end="")
    return EXIT_OK


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triton-cachekit",
        description=(
            "Inspect, explain, prewarm and ship Triton kernel compilation caches. "
            "Independent project; not affiliated with the Triton maintainers."
        ),
    )
    parser.add_argument("--version", action="version", version=f"triton-cachekit {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    def demo_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--demo",
            action="store_true",
            help="run on a built-in scenario (compiles real kernels; needs Triton, no GPU)",
        )
        p.add_argument(
            "--demo-dir",
            metavar="DIR",
            help="keep the demo files in DIR instead of a temporary directory",
        )

    p = sub.add_parser("inspect", help="list the entries of a cache directory")
    p.add_argument(
        "cache_dir",
        nargs="?",
        help="cache directory (default: $TRITON_CACHE_DIR or ~/.triton/cache)",
    )
    p.add_argument("--format", choices=("table", "json"), default="table")
    p.add_argument("--kernel", metavar="GLOB", help="only kernels whose name matches")
    p.add_argument(
        "--target", metavar="BACKEND[:ARCH]", help="only entries for this target, e.g. cuda:80"
    )
    p.add_argument("--problems", action="store_true", help="only entries with problems")
    p.add_argument("--wide", action="store_true", help="do not truncate keys and options")
    p.add_argument(
        "--locations", action="store_true", help="also show where each kernel was defined"
    )
    p.add_argument(
        "--strict", action="store_true", help="exit 1 if any kernel entry is not loadable"
    )
    demo_flags(p)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("explain", help="explain cache misses by comparing two recordings")
    p.add_argument(
        "recordings", nargs="*", metavar="RECORDING", help="A (built the cache) and B (missed it)"
    )
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.add_argument("--all", action="store_true", help="also list compilations that hit")
    p.add_argument(
        "--collisions", action="store_true", help="only look for different kernels sharing a key"
    )
    p.add_argument("--fail-on-miss", action="store_true", help="exit 1 if B had any miss")
    demo_flags(p)
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("record", help="run a command with the cache-key recorder enabled")
    p.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="recording to write (default: ./triton-cachekit-record.jsonl)",
    )
    p.add_argument("--no-source", action="store_true", help="do not store kernel source text")
    p.add_argument("command", nargs=argparse.REMAINDER, help="-- command [args...]")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("stats", help="compile time by kernel and what a warm cache would save")
    p.add_argument("recordings", nargs="*", metavar="RECORDING")
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.add_argument("--top", type=int, default=20, metavar="N")
    demo_flags(p)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser(
        "fingerprint", help="describe this host's Triton install and key-relevant environment"
    )
    p.add_argument("-o", "--output", metavar="FILE")
    p.add_argument("--compare", metavar="FILE", help="compare this host with a saved fingerprint")
    p.set_defaults(func=cmd_fingerprint)

    p = sub.add_parser(
        "prewarm",
        help="populate a cache for a target GPU on a machine without one (unverified on real GPUs)",
    )
    p.add_argument("--manifest", metavar="TOML", help="kernels and specialisations to compile")
    p.add_argument(
        "--from-recording", metavar="RECORDING", help="replay the compilations of a recorded run"
    )
    p.add_argument(
        "--target", metavar="BACKEND:ARCH[:WARP]", help="e.g. cuda:80, cuda:90, hip:gfx942:64"
    )
    p.add_argument("--cache-dir", metavar="DIR", help="cache directory to populate")
    p.add_argument(
        "-o",
        "--output",
        metavar="JSON",
        help="manifest of what was written (default: next to the cache)",
    )
    p.add_argument(
        "--recording-out", metavar="DIR", help="where to store the key recording of this run"
    )
    p.add_argument(
        "--timeout", type=float, default=600.0, metavar="SECONDS", help="per kernel (default 600)"
    )
    p.add_argument(
        "-j", "--jobs", type=int, default=1, metavar="N", help="kernels compiled in parallel"
    )
    p.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="environment for the compile workers (repeatable)",
    )
    p.add_argument(
        "--server-fingerprint",
        metavar="JSON",
        help="fingerprint taken on the GPU host; checked before compiling",
    )
    p.add_argument(
        "--force", action="store_true", help="compile even if the server fingerprint differs"
    )
    demo_flags(p)
    p.set_defaults(func=cmd_prewarm)

    p = sub.add_parser(
        "verify-hit", help="on the GPU host: check that prewarmed entries are really hit"
    )
    p.add_argument("--manifest", required=True, metavar="JSON", help="manifest written by prewarm")
    p.add_argument(
        "--cache-dir",
        metavar="DIR",
        help="cache directory (default: $TRITON_CACHE_DIR or ~/.triton/cache)",
    )
    p.add_argument(
        "--prewarm-recording", metavar="DIR", help="recording written by prewarm, to explain misses"
    )
    p.add_argument(
        "--recording-out", metavar="DIR", help="keep the recording of this verification run"
    )
    p.add_argument("--timeout", type=float, default=600.0, metavar="SECONDS")
    p.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument(
        "--fake-driver",
        action="store_true",
        help="simulate without a GPU (tests the tool, verifies nothing)",
    )
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.set_defaults(func=cmd_verify_hit)

    p = sub.add_parser(
        "pack", help="archive a cache (tar, tar.gz or OCI image layout) with manifest and checksums"
    )
    p.add_argument("cache_dir", nargs="?")
    p.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="PATH",
        help="*.tar, *.tar.gz, or a directory for an OCI layout",
    )
    p.add_argument(
        "--format", choices=("tar", "tar.gz", "oci"), help="default: from the output name"
    )
    p.add_argument("--kernel", metavar="GLOB")
    p.add_argument("--target", metavar="BACKEND[:ARCH]")
    p.add_argument("--include-incomplete", action="store_true")
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("unpack", help="extract an archive into a cache directory and fix its paths")
    p.add_argument("archive")
    p.add_argument("cache_dir")
    p.add_argument("--no-relocate", action="store_true", help="leave group files untouched")
    p.add_argument("--overwrite", action="store_true", help="replace files that already exist")
    p.set_defaults(func=cmd_unpack)

    p = sub.add_parser(
        "relocate", help="fix a cache directory that was copied or mounted at a new path"
    )
    p.add_argument("cache_dir", nargs="?")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_relocate)

    p = sub.add_parser("diff", help="compare two cache directories")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.add_argument("--no-content", action="store_true", help="compare keys only, do not hash files")
    p.add_argument("--exit-code", action="store_true", help="exit 1 if the caches differ")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser(
        "prune", help="delete entries by age, size, target or state (dry run unless --yes)"
    )
    p.add_argument("cache_dir", nargs="?")
    p.add_argument("--older-than", metavar="AGE", help="e.g. 30d, 12h")
    p.add_argument(
        "--max-size", metavar="SIZE", help="evict oldest entries until the cache fits, e.g. 2G"
    )
    p.add_argument(
        "--target",
        action="append",
        metavar="BACKEND[:ARCH]",
        help="remove entries for this target (repeatable)",
    )
    p.add_argument("--kernel", metavar="GLOB", help="remove entries of matching kernels")
    p.add_argument("--incomplete", action="store_true", help="remove entries Triton cannot load")
    p.add_argument(
        "--not-triton-version",
        metavar="VERSION",
        help="remove entries written by any other Triton version",
    )
    p.add_argument("--yes", action="store_true", help="actually delete")
    p.set_defaults(func=cmd_prune)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_USAGE
    try:
        return int(args.func(args))
    except CliError as exc:
        print(f"triton-cachekit: error: {exc}", file=sys.stderr)
        return exc.code
    except BrokenPipeError:
        return EXIT_OK
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
