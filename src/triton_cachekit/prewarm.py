"""``prewarm`` and ``verify-hit``: fill a cache without a GPU, then test it with one.

What is and is not established (see README, "What is and is not proven"):

* On a GPU-less machine the fake driver makes Triton run its complete pipeline
  and write complete cache entries, and a second process on the same machine
  hits them. This is covered by the integration tests.
* Whether the real GPU host computes the same keys is **not** verified by this
  project. ``verify-hit`` exists so that it can be checked on real hardware.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from ._util import human_duration_us, parse_target, render_table
from ._worker import RESULT_MARKER
from .explain import Finding
from .recording import Recording

MANIFEST_SCHEMA = 1

CAVEATS: tuple[str, ...] = (
    "Not verified on a real GPU: these entries were compiled through a fake driver on a "
    "machine without the target GPU. Run 'triton-cachekit verify-hit' on the GPU host.",
    "The server must run the byte-identical Triton install (same wheel, same Python version): "
    "triton_key() hashes the installed files.",
    "The server must have the same cache-invalidating environment variables set, and the same "
    "option-defaulting variables (TRITON_DEFAULT_FP_FUSION, TRITON_DEBUG, PTXAS_OPTIONS, ...).",
    "The kernel source must be byte-identical and start at the same line; on Triton builds "
    "that hash the file path (main after PR #11832) it must also be at the same absolute path.",
    "The specialisations must match what the application really passes: dtypes, constexpr "
    "values, integers equal to 1, and whether pointers/integers are divisible by 16. "
    "Prewarming from a recording of a real run avoids guessing these.",
    "The real driver must report the target you named (CUDA: capability major*10+minor, "
    "warp size 32; AMD: the gfx arch and the device's warp size).",
    "Triton 3.8.0/3.9.0 store absolute paths in the group files: mount the cache at the path "
    "it was built at, or run 'triton-cachekit relocate' / use 'unpack' on the server.",
)


class ManifestError(ValueError):
    pass


@dataclass
class Job:
    kernel: dict[str, Any]
    specs: list[dict[str, Any]]
    result: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        file = self.kernel.get("file")
        where = self.kernel.get("module") or (Path(str(file)).name if file else "?")
        return f"{where}:{self.kernel['name']}"


# ------------------------------------------------------------------ job sources
def load_manifest(path: str | Path) -> tuple[str | None, dict[str, str], list[Job]]:
    """Parse a prewarm manifest (TOML). Returns ``(target, env, jobs)``."""
    manifest_path = Path(path)
    try:
        with open(manifest_path, "rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"{manifest_path}: invalid TOML: {exc}") from exc
    settings = data.get("prewarm", {})
    if not isinstance(settings, dict):
        raise ManifestError(f"{manifest_path}: [prewarm] must be a table")
    target = settings.get("target")
    env = {str(k): str(v) for k, v in dict(settings.get("env", {})).items()}
    kernels = data.get("kernel")
    if not isinstance(kernels, list) or not kernels:
        raise ManifestError(f"{manifest_path}: expected at least one [[kernel]] table")
    jobs: list[Job] = []
    for index, item in enumerate(kernels):
        where = f"{manifest_path}: kernel #{index + 1}"
        if not isinstance(item, dict) or "name" not in item:
            raise ManifestError(f"{where}: 'name' is required")
        if ("file" in item) == ("module" in item):
            raise ManifestError(f"{where}: give exactly one of 'file' or 'module'")
        kernel: dict[str, Any] = {"name": str(item["name"])}
        if "file" in item:
            kernel["file"] = str((manifest_path.parent / str(item["file"])).resolve())
        else:
            kernel["module"] = str(item["module"])
        specs = item.get("spec")
        if not isinstance(specs, list) or not specs:
            raise ManifestError(f"{where}: expected at least one [[kernel.spec]] table")
        parsed = []
        for spec in specs:
            unknown = set(spec) - {"args", "kwargs"}
            if unknown:
                raise ManifestError(f"{where}: unknown spec keys {sorted(unknown)}")
            parsed.append(
                {"args": list(spec.get("args", [])), "kwargs": dict(spec.get("kwargs", {}))}
            )
        jobs.append(Job(kernel=kernel, specs=parsed))
    return (str(target) if target else None), env, jobs


def jobs_from_recording(
    recording: Recording, target: str | None = None
) -> tuple[list[Job], list[str]]:
    """Group the JIT compilations of a recording into replayable jobs.

    Returns the jobs and a list of human-readable reasons for skipped compilations.
    """
    grouped: dict[tuple[str, str, str], Job] = {}
    skipped: list[str] = []
    seen: set[str] = set()
    for item in recording.compiles:
        if item.hash in seen:
            continue
        seen.add(item.hash)
        if target and not _target_matches(item.target, target):
            skipped.append(f"{item.kernel}: recorded for {item.target}, not {target}")
            continue
        data = item.specialization_data
        if data is None:
            skipped.append(f"{item.kernel}: no specialisation data (not compiled through the JIT)")
            continue
        fn = item.components.get("source", {}).get("fn", {})
        qualified = item.kernel
        module, _, name = qualified.rpartition(".")
        if "<locals>" in qualified:
            skipped.append(f"{qualified}: defined inside a function, cannot be imported by name")
            continue
        file = str(fn.get("file") or "")
        key = (module, file, name)
        job = grouped.get(key)
        if job is None:
            job = Job(kernel={"module": module, "file": file or None, "name": name}, specs=[])
            grouped[key] = job
        job.specs.append({"specialization_data": data, "expected_hash": item.hash})
    return list(grouped.values()), skipped


def _target_matches(recorded: str, wanted: str) -> bool:
    backend, arch, _ = parse_target(wanted)
    return recorded == f"{backend}:{arch}"


# ------------------------------------------------------------------- execution
def _worker_env(cache_dir: Path | None, extra_env: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update(extra_env)
    if cache_dir is not None:
        env["TRITON_CACHE_DIR"] = str(cache_dir)
    # Make sure the worker imports this very package, installed or not.
    package_parent = str(Path(__file__).resolve().parent.parent)
    existing = env.get("PYTHONPATH", "")
    if package_parent not in existing.split(os.pathsep):
        env["PYTHONPATH"] = package_parent + (os.pathsep + existing if existing else "")
    env.pop("TRITON_CACHEKIT_RECORD", None)
    return env


def run_job(
    job: Job,
    *,
    target: str | None,
    fake_driver: bool,
    cache_dir: Path | None,
    record_to: Path | None,
    timeout: float,
    extra_env: dict[str, str],
    python: str | None = None,
) -> dict[str, Any]:
    payload = {
        "kernel": job.kernel,
        "specs": job.specs,
        "target": target,
        "fake_driver": fake_driver,
        "record_to": str(record_to) if record_to else None,
    }
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [python or sys.executable, "-m", "triton_cachekit._worker"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_worker_env(cache_dir, extra_env),
            check=False,
        )
    except subprocess.TimeoutExpired:
        result: dict[str, Any] = {
            "specs": [],
            "error": f"timed out after {timeout:.0f}s (worker killed)",
            "timed_out": True,
        }
    else:
        result = {"specs": [], "error": None}
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith(RESULT_MARKER):
                result = json.loads(line[len(RESULT_MARKER) :])
                break
        else:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
            result["error"] = f"worker exited with status {proc.returncode} without a result" + (
                ":\n" + "\n".join(tail) if tail else ""
            )
    result["wall_seconds"] = round(time.perf_counter() - started, 3)
    job.result = result
    return result


def run_jobs(
    jobs: list[Job],
    *,
    target: str | None,
    fake_driver: bool,
    cache_dir: Path | None,
    record_dir: Path | None,
    timeout: float,
    extra_env: dict[str, str],
    parallel: int = 1,
    python: str | None = None,
) -> None:
    def run(indexed: tuple[int, Job]) -> None:
        index, job = indexed
        record_to = record_dir / f"job-{index:04d}.jsonl" if record_dir else None
        run_job(
            job,
            target=target,
            fake_driver=fake_driver,
            cache_dir=cache_dir,
            record_to=record_to,
            timeout=timeout,
            extra_env=extra_env,
            python=python,
        )

    if record_dir is not None:
        record_dir.mkdir(parents=True, exist_ok=True)
        for stale in record_dir.glob("job-*.jsonl"):
            stale.unlink()
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        list(pool.map(run, enumerate(jobs)))


# --------------------------------------------------------------------- results
def summarise(jobs: list[Job]) -> dict[str, int]:
    counts = {"compiled": 0, "hit": 0, "duplicate": 0, "partial": 0, "error": 0}
    for job in jobs:
        specs = job.result.get("specs", [])
        if job.result.get("error"):
            counts["error"] += max(1, len(job.specs) - len(specs))
        for spec in specs:
            counts[spec.get("status", "error")] = counts.get(spec.get("status", "error"), 0) + 1
    return counts


def build_output_manifest(
    jobs: list[Job],
    *,
    target: str | None,
    cache_dir: Path | None,
    recording: Path | None,
    extra_env: dict[str, str],
    kind: str = "prewarm",
) -> dict[str, Any]:
    fingerprint = next((j.result["fingerprint"] for j in jobs if j.result.get("fingerprint")), None)
    actual_target = next((j.result["target"] for j in jobs if j.result.get("target")), target)
    entries = []
    for job in jobs:
        for spec, outcome in zip(job.specs, job.result.get("specs", []), strict=False):
            for compilation in outcome.get("compilations", []):
                from ._util import base32_key

                entries.append(
                    {
                        "kernel": job.label,
                        "hash": compilation["hash"],
                        "entry": base32_key(compilation["hash"]),
                        "status": "hit" if compilation["cache_hit"] else "compiled",
                        "compile_us": compilation["us"],
                        "spec": spec,
                    }
                )
    return {
        "schema": MANIFEST_SCHEMA,
        "kind": kind,
        "tool_version": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": actual_target,
        "driver": next((j.result.get("driver") for j in jobs if j.result.get("driver")), None),
        "cache_dir": str(cache_dir) if cache_dir else None,
        "recording": str(recording) if recording else None,
        "env": extra_env,
        "fingerprint": fingerprint,
        "summary": summarise(jobs),
        "entries": entries,
        "jobs": [{"kernel": j.kernel, "specs": j.specs, "result": _slim(j.result)} for j in jobs],
        "verified_on_real_gpu": False,
        "caveats": list(CAVEATS) if kind == "prewarm" else [],
    }


def _slim(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "fingerprint"}


def load_output_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or data.get("schema") != MANIFEST_SCHEMA or "jobs" not in data:
        raise ManifestError(f"{path}: not a triton-cachekit prewarm manifest (schema 1)")
    return data


def jobs_from_output_manifest(data: dict[str, Any]) -> list[Job]:
    jobs = []
    for item in data["jobs"]:
        jobs.append(Job(kernel=dict(item["kernel"]), specs=[dict(s) for s in item["specs"]]))
    return jobs


def render_results(jobs: list[Job], *, verb: str = "prewarm") -> str:
    rows = []
    errors: list[str] = []
    for job in jobs:
        if job.result.get("error"):
            errors.append(f"{job.label}: {job.result['error']}")
        for spec, outcome in zip(job.specs, job.result.get("specs", []), strict=False):
            compilations = outcome.get("compilations", [])
            entry = "-"
            if compilations:
                from ._util import base32_key

                entry = base32_key(compilations[0]["hash"])[:10]
                if len(compilations) > 1:
                    entry += f" (+{len(compilations) - 1})"
            took = sum(c["us"] for c in compilations)
            rows.append(
                [
                    job.label if len(job.label) < 48 else "..." + job.label[-45:],
                    describe_spec(spec),
                    outcome.get("status", "error"),
                    entry,
                    human_duration_us(took) if compilations else "-",
                ]
            )
            if outcome.get("error"):
                errors.append(f"{job.label} [{describe_spec(spec)}]: {outcome['error']}")
    out = [
        render_table(
            ["KERNEL", "SPECIALISATION", "RESULT", "ENTRY", "COMPILE"], rows, right_align=(4,)
        )
    ]
    counts = summarise(jobs)
    out.append("")
    tally = ", ".join(f"{count} {name}" for name, count in counts.items() if count)
    out.append(f"{verb}: {tally or 'nothing to do'}")
    if errors:
        out.append("")
        out.append("Errors:")
        out.extend(f"  - {line}" for line in errors)
    return "\n".join(out) + "\n"


def describe_spec(spec: dict[str, Any]) -> str:
    if "specialization_data" in spec:
        try:
            data = json.loads(spec["specialization_data"])
        except ValueError:
            return "(recorded)"
        sig = data.get("signature", {})
        consts = dict(
            zip(
                [tuple(k) for k in data.get("constant_keys", [])],
                data.get("constant_vals", []),
                strict=False,
            )
        )
        names = list(sig)
        parts = []
        for index, name in enumerate(names):
            if sig[name] == "constexpr":
                parts.append(f"{name}={consts.get((index,), '?')}")
            else:
                parts.append(str(sig[name]))
        text = ", ".join(parts)
    else:
        parts = [str(a) for a in spec.get("args", [])]
        parts += [f"{k}={v}" for k, v in spec.get("kwargs", {}).items()]
        text = ", ".join(parts)
    return text if len(text) <= 60 else text[:57] + "..."


# ------------------------------------------------------------------ verify-hit
@dataclass
class VerifyReport:
    static_findings: list[Finding]
    jobs: list[Job]
    expected: dict[str, str]  # hash -> kernel label
    rows: list[dict[str, Any]] = field(default_factory=list)
    simulated: bool = False
    target: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return (
            self.error is None and bool(self.rows) and all(r["status"] == "hit" for r in self.rows)
        )


def evaluate_verification(manifest: dict[str, Any], jobs: list[Job]) -> list[dict[str, Any]]:
    """Compare what the host compiled with what the prewarm run wrote."""
    expected_by_job: list[list[list[str]]] = []
    for item in manifest["jobs"]:
        per_spec = []
        for outcome in item.get("result", {}).get("specs", []):
            per_spec.append([c["hash"] for c in outcome.get("compilations", [])])
        expected_by_job.append(per_spec)
    rows: list[dict[str, Any]] = []
    for job_index, job in enumerate(jobs):
        outcomes = job.result.get("specs", [])
        for spec_index, spec in enumerate(job.specs):
            expected = []
            if job_index < len(expected_by_job) and spec_index < len(expected_by_job[job_index]):
                expected = expected_by_job[job_index][spec_index]
            outcome = outcomes[spec_index] if spec_index < len(outcomes) else None
            row: dict[str, Any] = {
                "kernel": job.label,
                "spec": describe_spec(spec),
                "expected": expected,
                "actual": [],
                "status": "error",
                "reason": "",
            }
            if outcome is None:
                row["reason"] = job.result.get("error") or "worker produced no result"
            elif outcome.get("status") == "error":
                row["reason"] = outcome.get("error", "error")
            else:
                compilations = outcome.get("compilations", [])
                row["actual"] = [c["hash"] for c in compilations]
                if not compilations:
                    row["status"] = "duplicate"
                    row["reason"] = "same specialisation as an earlier row"
                elif all(c["cache_hit"] for c in compilations):
                    row["status"] = "hit"
                elif set(row["actual"]) == set(expected):
                    row["status"] = "miss"
                    row["reason"] = (
                        "same key as prewarm, but no loadable entry in this cache directory"
                    )
                else:
                    row["status"] = "miss"
                    row["reason"] = "this host computes a different key than the prewarm run"
            rows.append(row)
    return rows


def render_verification(report: VerifyReport) -> str:
    out: list[str] = []
    if report.simulated:
        out.append(
            "SIMULATION: --fake-driver was used. This exercises the tool, it does NOT verify "
            "anything about a real GPU."
        )
        out.append("")
    if report.target:
        out.append(f"Driver target on this host: {report.target}")
    if report.static_findings:
        out.append("Environment differs from the prewarm run:")
        for finding in report.static_findings:
            out.append(f"  - [{finding.component}] {finding.summary}")
            out.extend(f"      {d}" for d in finding.details[:12])
        out.append("")
    if report.error:
        out.append(f"error: {report.error}")
        return "\n".join(out) + "\n"
    table = [
        [
            r["kernel"] if len(r["kernel"]) < 48 else "..." + r["kernel"][-45:],
            r["spec"],
            r["status"].upper(),
            r["reason"],
        ]
        for r in report.rows
    ]
    out.append(render_table(["KERNEL", "SPECIALISATION", "RESULT", "WHY"], table))
    hits = sum(1 for r in report.rows if r["status"] == "hit")
    misses = sum(1 for r in report.rows if r["status"] == "miss")
    errors = sum(1 for r in report.rows if r["status"] == "error")
    out.append("")
    out.append(f"verify-hit: {hits} hit, {misses} miss, {errors} error")
    return "\n".join(out) + "\n"
