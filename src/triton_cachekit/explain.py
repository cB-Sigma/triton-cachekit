"""Compare two recordings and explain cache misses component by component.

Every explanation here corresponds to a statement that was checked in Triton's
source (see ``docs/cache-key.md`` for file references). When the recorder cannot
see why a hash changed, the report says so instead of guessing.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from typing import Any

from ._util import human_duration_us
from .recording import Compile, Recording

# Order matters: the first group that differs is usually the root cause.
COMPONENT_ORDER = ("triton", "backend", "env", "options", "source", "pipeline")


@dataclass
class Finding:
    component: str  # e.g. "triton.version", "source.location.line"
    summary: str
    details: list[str] = field(default_factory=list)
    remedy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "summary": self.summary,
            "details": self.details,
            "remedy": self.remedy,
        }


@dataclass
class KernelReport:
    kernel: str
    specialization: str
    target: str
    status: str  # hit | miss-same-key | miss-key-differs | miss-new-kernel | unknown
    hash: str
    entry: str
    compile_us: int | None
    matched_hash: str | None = None
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "specialization": self.specialization,
            "target": self.target,
            "status": self.status,
            "hash": self.hash,
            "entry": self.entry,
            "compile_us": self.compile_us,
            "compared_with_hash": self.matched_hash,
            "findings": [f.to_dict() for f in self.findings],
            "notes": self.notes,
        }


@dataclass
class Collision:
    hash: str
    entry: str
    severity: str  # "collision" | "debug-info" | "shared"
    summary: str
    members: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash,
            "entry": self.entry,
            "severity": self.severity,
            "summary": self.summary,
            "members": self.members,
        }


@dataclass
class ExplainReport:
    label_a: str
    label_b: str
    kernels: list[KernelReport]
    collisions: list[Collision]
    global_findings: list[Finding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def misses(self) -> list[KernelReport]:
        return [k for k in self.kernels if k.status.startswith("miss")]

    def to_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for item in self.kernels:
            counts[item.status] = counts.get(item.status, 0) + 1
        return {
            "a": self.label_a,
            "b": self.label_b,
            "summary": counts,
            "environment": [f.to_dict() for f in self.global_findings],
            "kernels": [k.to_dict() for k in self.kernels],
            "collisions": [c.to_dict() for c in self.collisions],
            "warnings": self.warnings,
        }


# ----------------------------------------------------------------------------
# component comparison
# ----------------------------------------------------------------------------
def _location_policy(session: dict[str, Any] | None) -> list[str]:
    if not session:
        return ["line"]
    policy = session.get("location_in_key")
    return list(policy) if isinstance(policy, list) else ["line"]


def _triton_findings(
    a: Compile, b: Compile, sess_a: dict[str, Any] | None, sess_b: dict[str, Any] | None
) -> list[Finding]:
    ta = a.components.get("triton", {})
    tb = b.components.get("triton", {})
    if ta.get("key_digest") == tb.get("key_digest"):
        return []
    remedy = (
        "The key starts with triton_key(): the Triton version followed by the sha256 of "
        "runtime/cache.py, every module under triton/compiler, triton/backends and "
        "triton/language, and the libtriton shared library. Entries are only shared between "
        "byte-identical Triton installs: use the same wheel (same version, same Python ABI, "
        "same platform) on both sides and do not patch it in place."
    )
    if ta.get("version") != tb.get("version"):
        return [
            Finding(
                "triton.version",
                f"Triton version differs: {ta.get('version')} (A) vs {tb.get('version')} (B)",
                remedy=remedy,
            )
        ]
    files_a = (sess_a or {}).get("triton", {})
    files_b = (sess_b or {}).get("triton", {})
    details: list[str] = []
    if files_a.get("files_verified") and files_b.get("files_verified"):
        fa: dict[str, str] = files_a.get("files", {})
        fb: dict[str, str] = files_b.get("files", {})
        for name in sorted(set(fa) | set(fb)):
            if name not in fb:
                details.append(f"{name}: only in A")
            elif name not in fa:
                details.append(f"{name}: only in B")
            elif fa[name] != fb[name]:
                details.append(f"{name}: content differs")
    else:
        details.append("per-file hashes are not available for both recordings")
    return [
        Finding(
            "triton.files",
            f"Triton {ta.get('version')} is installed on both sides but its files differ",
            details=details,
            remedy=remedy,
        )
    ]


def _backend_findings(a: Compile, b: Compile) -> list[Finding]:
    ba = a.components.get("backend", {})
    bb = b.components.get("backend", {})
    if ba.get("hash") == bb.get("hash"):
        return []
    target_a, target_b = ba.get("target"), bb.get("target")
    if target_a != target_b:
        return [
            Finding(
                "backend.target",
                f"compilation target differs: {_fmt_target(target_a)} (A) vs {_fmt_target(target_b)} (B)",
                remedy=(
                    "The backend hash contains the target architecture (CUDA compute capability "
                    "such as 80/90, or the AMD gfx name). A cache built for one GPU generation "
                    "is never used on another; prewarm once per target you deploy to."
                ),
            )
        ]
    details = list(
        difflib.unified_diff(
            str(ba.get("hash")).splitlines(),
            str(bb.get("hash")).splitlines(),
            "A",
            "B",
            lineterm="",
        )
    )
    return [
        Finding(
            "backend.toolchain",
            "backend toolchain differs for the same target",
            details=details,
            remedy=(
                "For CUDA the backend hash is the output of 'ptxas --version' plus the "
                "architecture; for AMD it is the target plus the AMD code generator revision. "
                "Use the same Triton wheel and do not point TRITON_PTXAS_PATH (or "
                "TRITON_PTXAS_BLACKWELL_PATH) at a different ptxas on one side only."
            ),
        )
    ]


def _fmt_target(target: Any) -> str:
    if isinstance(target, dict):
        return f"{target.get('backend')}:{target.get('arch')}:{target.get('warp_size')}"
    return str(target)


def _env_findings(a: Compile, b: Compile) -> list[Finding]:
    ea: dict[str, str] = a.components.get("env", {})
    eb: dict[str, str] = b.components.get("env", {})
    if ea == eb:
        return []
    details = []
    for name in sorted(set(ea) | set(eb)):
        if name not in eb:
            details.append(f"{name}: {ea[name]!r} in A, unset in B")
        elif name not in ea:
            details.append(f"{name}: unset in A, {eb[name]!r} in B")
        elif ea[name] != eb[name]:
            details.append(f"{name}: {ea[name]!r} in A, {eb[name]!r} in B")
    return [
        Finding(
            "env",
            "cache-invalidating environment variables differ",
            details=details,
            remedy=(
                "Triton puts every *set* variable from its cache-invalidating list into the key "
                "(sorted name/value pairs; boolean spellings are normalised to true/false). "
                "An unset variable and one explicitly set to its default are different keys "
                "(upstream issue #11817). Export exactly the same set on both sides."
            ),
        )
    ]


def _options_findings(a: Compile, b: Compile) -> list[Finding]:
    oa = a.components.get("options", {})
    ob = b.components.get("options", {})
    if oa.get("hash") == ob.get("hash"):
        return []
    va: dict[str, Any] = oa.get("values", {})
    vb: dict[str, Any] = ob.get("values", {})
    details = []
    for name in sorted(set(va) | set(vb)):
        if name == "extern_libs":
            continue
        if va.get(name) != vb.get(name):
            details.append(f"{name}: {va.get(name)!r} (A) vs {vb.get(name)!r} (B)")
    la: dict[str, Any] = oa.get("extern_libs", {})
    lb: dict[str, Any] = ob.get("extern_libs", {})
    for name in sorted(set(la) | set(lb)):
        sha_a = (la.get(name) or {}).get("sha256")
        sha_b = (lb.get(name) or {}).get("sha256")
        if sha_a != sha_b:
            details.append(
                f"extern_libs[{name}]: file content differs "
                f"({(la.get(name) or {}).get('path')} vs {(lb.get(name) or {}).get('path')})"
            )
    if not details:
        details.append("the options hash differs but no recorded field does")
    return [
        Finding(
            "options",
            "compiler options differ",
            details=details,
            remedy=(
                "Every field of the backend options object is hashed (extern libraries by "
                "file content, not by path). Launch the kernel with the same num_warps / "
                "num_stages / ... on both sides. Some defaults are read from the environment "
                "when the options are built (in Triton 3.9.0: enable_fp_fusion from "
                "TRITON_DEFAULT_FP_FUSION, debug from TRITON_DEBUG, ptx_options from "
                "PTXAS_OPTIONS, arch from TRITON_OVERRIDE_ARCH, libdevice from "
                "TRITON_LIBDEVICE_PATH)."
            ),
        )
    ]


def _source_diff(src_a: str | None, src_b: str | None) -> list[str]:
    if src_a is None or src_b is None:
        return ["(source text was not recorded; re-record without TRITON_CACHEKIT_RECORD_SOURCE=0)"]
    diff = list(
        difflib.unified_diff(src_a.splitlines(), src_b.splitlines(), "A", "B", lineterm="", n=1)
    )
    if len(diff) > 40:
        diff = [*diff[:40], f"... ({len(diff) - 40} more diff lines)"]
    return diff


def _fn_findings(
    fa: dict[str, Any], fb: dict[str, Any], policy: list[str], prefix: str, depth: int = 0
) -> list[Finding]:
    """Explain why two recorded function hashes differ."""
    findings: list[Finding] = []
    who = fb.get("name") or fa.get("name") or "?"
    label = f"{prefix}{who}"

    if fa.get("src_sha256") != fb.get("src_sha256"):
        findings.append(
            Finding(
                "source.text",
                f"source code of {label} differs",
                details=_source_diff(fa.get("src"), fb.get("src")),
                remedy=(
                    "The function's source text (from its 'def' line on, comments and "
                    "whitespace included) is hashed. Any edit, including reformatting, "
                    "gives a new key: rebuild the cache from the code you deploy."
                ),
            )
        )
    if "line" in policy and fa.get("line") != fb.get("line"):
        findings.append(
            Finding(
                "source.location.line",
                f"{label} starts at line {fa.get('line')} in A and line {fb.get('line')} in B",
                remedy=(
                    "The line number where the function starts (its first decorator) is part "
                    "of the hash, so inserting or removing lines above a kernel changes its "
                    "key even though the kernel is untouched - also with "
                    "TRITON_DISABLE_LINE_INFO=1 (upstream issue #11817). Build the cache from "
                    "exactly the file that is deployed."
                ),
            )
        )
    if "file" in policy and fa.get("file") != fb.get("file"):
        findings.append(
            Finding(
                "source.location.file",
                f"{label} is defined in a file at a different path",
                details=[f"A: {fa.get('file')}", f"B: {fb.get('file')}"],
                remedy=(
                    "In this Triton the path of the source file is part of the key: the same "
                    "kernel installed at a different path will not hit (upstream issue "
                    "#12367). Install the package at the same absolute path on the build "
                    "machine and on the server (same venv location, same site-packages)."
                ),
            )
        )
    for part, title in (("def_line", "'def' line"), ("def_col", "'def' column")):
        if part in policy and fa.get(part) != fb.get(part) and fa.get("line") == fb.get("line"):
            findings.append(
                Finding(
                    f"source.location.{part}",
                    f"{label}: {title} differs ({fa.get(part)} vs {fb.get(part)})",
                    remedy="The position of the 'def' statement is part of the key in this Triton.",
                )
            )
    if fa.get("constexpr_globals") != fb.get("constexpr_globals"):
        ga = dict(fa.get("constexpr_globals") or [])
        gb = dict(fb.get("constexpr_globals") or [])
        details = [
            f"{name}: {ga.get(name, '<absent>')} (A) vs {gb.get(name, '<absent>')} (B)"
            for name in sorted(set(ga) | set(gb))
            if ga.get(name) != gb.get(name)
        ]
        findings.append(
            Finding(
                "source.globals",
                f"tl.constexpr globals read by {label} differ",
                details=details,
                remedy=(
                    "Global tl.constexpr values a kernel reads by name are hashed. Make sure "
                    "configuration that feeds such globals is identical on both sides."
                ),
            )
        )
    if bool(fa.get("noinline")) != bool(fb.get("noinline")) and depth > 0:
        findings.append(
            Finding(
                "source.noinline",
                f"{label}: 'noinline' differs",
                remedy="The noinline flag of every called @triton.jit function is hashed.",
            )
        )

    deps_a, deps_b = fa.get("deps"), fb.get("deps")
    if deps_a is None or deps_b is None:
        if not findings and fa.get("cache_key") != fb.get("cache_key"):
            findings.append(
                Finding(
                    "source.dependencies.unknown",
                    f"the hash of {label} differs but its dependencies were not recorded",
                    remedy=(
                        "The recorder was enabled after this function had already been hashed. "
                        "Enable it before the kernels are first used (import "
                        "triton_cachekit.record at start-up or use 'triton-cachekit record --')."
                    ),
                )
            )
        return findings
    by_name_a = {d.get("name"): d for d in deps_a}
    by_name_b = {d.get("name"): d for d in deps_b}
    for name in sorted(set(by_name_a) | set(by_name_b), key=str):
        if name not in by_name_b:
            findings.append(
                Finding(
                    "source.dependency",
                    f"{label} calls {name} in A but not in B",
                    remedy="Called @triton.jit functions are part of the caller's hash.",
                )
            )
        elif name not in by_name_a:
            findings.append(
                Finding(
                    "source.dependency",
                    f"{label} calls {name} in B but not in A",
                    remedy="Called @triton.jit functions are part of the caller's hash.",
                )
            )
        elif by_name_a[name].get("cache_key") != by_name_b[name].get("cache_key") and depth < 8:
            nested = _fn_findings(
                by_name_a[name], by_name_b[name], policy, f"{label} -> ", depth + 1
            )
            for item in nested:
                item.component = item.component.replace("source.", "source.dependency.", 1)
                if "caller" not in item.remedy:
                    item.remedy += (
                        " The hash of every @triton.jit function a kernel calls is folded "
                        "into the caller's hash, so this changes the caller's key too."
                    )
            findings.extend(nested)
    if not findings and fa.get("cache_key") != fb.get("cache_key"):
        findings.append(
            Finding(
                "source.unknown",
                f"the hash of {label} differs but no recorded input does",
                remedy=(
                    "Something feeds the function hash that the recorder does not capture "
                    "(for example the annotations of a @tl.aggregate class). Please open an "
                    "issue with both recordings."
                ),
            )
        )
    return findings


def _source_findings(a: Compile, b: Compile, policy: list[str]) -> list[Finding]:
    sa = a.components.get("source", {})
    sb = b.components.get("source", {})
    if sa.get("hash") == sb.get("hash"):
        return []
    if sa.get("kind") != "ast" or sb.get("kind") != "ast":
        return [
            Finding(
                "source.ir",
                "the IR file given to triton.compile differs",
                details=[f"A: {sa.get('path')}", f"B: {sb.get('path')}"],
                remedy="For IR sources the key hashes the file extension and the file content.",
            )
        ]
    findings: list[Finding] = []
    fa, fb = sa.get("fn", {}), sb.get("fn", {})
    if fa.get("cache_key") != fb.get("cache_key"):
        findings.extend(_fn_findings(fa, fb, policy, ""))
    sig_a: dict[str, str] = sa.get("signature", {})
    sig_b: dict[str, str] = sb.get("signature", {})
    if sig_a != sig_b:
        details = [
            f"{name}: {sig_a.get(name, '<absent>')} (A) vs {sig_b.get(name, '<absent>')} (B)"
            for name in list(dict.fromkeys([*sig_a, *sig_b]))
            if sig_a.get(name) != sig_b.get(name)
        ]
        findings.append(
            Finding(
                "specialization.signature",
                "argument types differ",
                details=details,
                remedy=(
                    "Each distinct combination of argument types is a separate compilation "
                    "(tensor dtype, i32 vs i64 integers, and an integer argument equal to 1 "
                    "is turned into a constexpr). A did not compile this combination: add it "
                    "to the prewarm manifest or record a run that exercises it."
                ),
            )
        )
    const_a = {tuple(c.get("path", [])): c for c in sa.get("constants", [])}
    const_b = {tuple(c.get("path", [])): c for c in sb.get("constants", [])}
    if {k: v.get("key") for k, v in const_a.items()} != {
        k: v.get("key") for k, v in const_b.items()
    }:
        details = []
        for path in sorted(set(const_a) | set(const_b)):
            ca, cb = const_a.get(path), const_b.get(path)
            if (ca or {}).get("key") != (cb or {}).get("key"):
                label = (cb or ca or {}).get("name") or f"arg{list(path)}"
                details.append(
                    f"{label}: {(ca or {}).get('value', '<not constexpr>')} (A) vs "
                    f"{(cb or {}).get('value', '<not constexpr>')} (B)"
                )
        findings.append(
            Finding(
                "specialization.constants",
                "constexpr argument values differ",
                details=details,
                remedy=(
                    "Every distinct tl.constexpr value is a separate compilation. A did not "
                    "compile this value: add the specialisation to the prewarm manifest or "
                    "record a run that exercises it."
                ),
            )
        )
    if sa.get("attrs") != sb.get("attrs"):
        findings.append(
            Finding(
                "specialization.attrs",
                "argument specialisation hints differ",
                details=[f"A: {sa.get('attrs')}", f"B: {sb.get('attrs')}"],
                remedy=(
                    "Triton specialises on whether each pointer and each integer argument is "
                    "divisible by 16 ('tt.divisibility'). The same kernel called with an "
                    "unaligned tensor view or an integer that is not a multiple of 16 is a "
                    "different key. Prewarm both variants, or mark the argument with "
                    "do_not_specialize / do_not_specialize_on_alignment in @triton.jit."
                ),
            )
        )
    if not findings:
        findings.append(
            Finding(
                "source.unknown",
                "the source hash differs but no recorded input does",
                remedy="Please open an issue with both recordings.",
            )
        )
    return findings


def compare_compiles(
    a: Compile,
    b: Compile,
    sess_a: dict[str, Any] | None = None,
    sess_b: dict[str, Any] | None = None,
) -> list[Finding]:
    """All key components that differ between two compilations, root causes first."""
    sess_a = sess_a or a.session
    sess_b = sess_b or b.session
    findings: list[Finding] = []
    findings.extend(_triton_findings(a, b, sess_a, sess_b))
    findings.extend(_backend_findings(a, b))
    findings.extend(_env_findings(a, b))
    findings.extend(_options_findings(a, b))
    findings.extend(_source_findings(a, b, _location_policy(sess_b)))
    if bool(a.raw.get("custom_pipeline")) != bool(b.raw.get("custom_pipeline")):
        findings.append(
            Finding(
                "pipeline",
                "a custom compilation pipeline hook is active on one side only",
                remedy=(
                    "knobs.runtime.add_stages_inspection_hook appends its own key to the cache "
                    "key. Install the same hook on both sides or on neither."
                ),
            )
        )
    if not findings and a.hash != b.hash:
        findings.append(
            Finding(
                "unknown",
                "the keys differ but every recorded component is equal",
                remedy="Please open an issue with both recordings.",
            )
        )
    return findings


# ----------------------------------------------------------------------------
# recording comparison
# ----------------------------------------------------------------------------
def _candidates(compile_b: Compile, recording_a: Recording) -> list[Compile]:
    same_name = [c for c in recording_a.compiles if c.kernel == compile_b.kernel]
    if same_name:
        return same_name
    # same function name in a module that was renamed or loaded under another name
    return [c for c in recording_a.compiles if c.name == compile_b.name]


def _path_note(a: Compile, b: Compile, policy: list[str]) -> str | None:
    fa = a.components.get("source", {}).get("fn", {})
    fb = b.components.get("source", {}).get("fn", {})
    if (
        fa.get("file")
        and fb.get("file")
        and fa.get("file") != fb.get("file")
        and "file" not in policy
    ):
        return (
            f"the source file path differs ({fa.get('file')} vs {fb.get('file')}), which is not "
            "part of the key in this Triton version (3.8.0/3.9.0 hash only the start line; "
            "main hashes the path since PR #11832)"
        )
    return None


def compare_recordings(a: Recording, b: Recording, *, include_hits: bool = False) -> ExplainReport:
    """Explain, for every compilation in ``b``, how it relates to the cache built by ``a``."""
    hashes_a = a.by_hash()
    reports: list[KernelReport] = []
    seen: set[str] = set()
    for item in b.compiles:
        if item.hash in seen:
            continue
        seen.add(item.hash)
        report = KernelReport(
            kernel=item.kernel,
            specialization=item.describe_specialization(),
            target=item.target,
            status="unknown",
            hash=item.hash,
            entry=item.entry,
            compile_us=item.total_us,
        )
        policy = _location_policy(item.session)
        if item.hash in hashes_a:
            first = hashes_a[item.hash][0]
            report.matched_hash = item.hash
            note = _path_note(first, item, policy)
            if note:
                report.notes.append(note)
            if item.cache_hit:
                report.status = "hit"
            elif item.cache_hit is None:
                report.status = "unknown"
                report.notes.append("same key as in A, but B recorded no outcome for it")
            else:
                report.status = "miss-same-key"
                report.findings.append(_same_key_miss(first, item))
        elif item.cache_hit:
            report.status = "hit"
            report.notes.append("hit an entry that A did not produce (already in B's cache)")
        else:
            candidates = _candidates(item, a)
            if not candidates:
                report.status = "miss-new-kernel" if item.cache_hit is False else "unknown"
                report.findings.append(
                    Finding(
                        "coverage",
                        "A never compiled a kernel with this name",
                        remedy=(
                            "The run that built the cache did not reach this kernel. Extend "
                            "the warm-up workload or add the kernel to the prewarm manifest."
                        ),
                    )
                )
            else:
                scored = [
                    (
                        compare_compiles(
                            c, item, c.session or a.session, item.session or b.session
                        ),
                        c,
                    )
                    for c in candidates
                ]
                scored.sort(key=lambda pair: (len(pair[0]), _rank(pair[0])))
                findings, best = scored[0]
                report.status = "miss-key-differs" if item.cache_hit is False else "unknown"
                report.findings = findings
                report.matched_hash = best.hash
                if item.cache_hit is None:
                    report.notes.append("B recorded no outcome for this compilation")
        if report.status != "hit" or include_hits:
            reports.append(report)

    global_findings: list[Finding] = []
    if a.compiles and b.compiles:
        probe = compare_compiles(a.compiles[0], b.compiles[0], a.session, b.session)
        global_findings = [f for f in probe if f.component.startswith(("triton.", "env"))]
    return ExplainReport(
        label_a=a.label,
        label_b=b.label,
        kernels=reports,
        collisions=[c for c in find_collisions([a, b]) if c.severity != "shared"],
        global_findings=global_findings,
        warnings=[*a.warnings, *b.warnings],
    )


def _rank(findings: list[Finding]) -> int:
    order = {"triton": 0, "backend": 1, "env": 2, "options": 3, "source": 4, "specialization": 5}
    return min((order.get(f.component.split(".")[0], 9) for f in findings), default=9)


def _same_key_miss(a: Compile, b: Compile) -> Finding:
    details = [f"A used cache directory {a.cache_dir}", f"B used cache directory {b.cache_dir}"]
    return Finding(
        "cache-contents",
        "the key is identical to A's, but B's cache directory had no loadable entry for it",
        details=details,
        remedy=(
            f"Check that the entry {b.entry} from A was shipped into B's cache directory and "
            "that TRITON_CACHE_DIR points at it ('triton-cachekit inspect DIR'). If the "
            "directory was copied or mounted at another path, Triton 3.8.0/3.9.0 reject it "
            "because the group file stores absolute paths: run 'triton-cachekit relocate DIR' "
            "(or use 'triton-cachekit unpack', which does it). TRITON_ALWAYS_COMPILE=1 also "
            "forces a miss."
        ),
    )


def find_collisions(recordings: list[Recording]) -> list[Collision]:
    """Different kernels (or different generated code) that share one cache key."""
    groups: dict[str, list[tuple[Recording, Compile]]] = {}
    for recording in recordings:
        for item in recording.compiles:
            groups.setdefault(item.hash, []).append((recording, item))
    collisions: list[Collision] = []
    for digest, members in groups.items():
        if len(members) < 2:
            continue
        identities = {}
        fresh_ir: set[str] = set()
        fresh_ir_no_loc: set[str] = set()
        for recording, item in members:
            fn = item.components.get("source", {}).get("fn", {})
            identities[(item.kernel, fn.get("file"), fn.get("line"))] = recording.label
            if item.cache_hit is False and item.outcome is not None:
                artifacts = item.outcome.get("artifacts", {})
                ir = artifacts.get("source") or artifacts.get("ttir") or {}
                if ir.get("sha256"):
                    fresh_ir.add(str(ir["sha256"]))
                if ir.get("sha256_no_loc"):
                    fresh_ir_no_loc.add(str(ir["sha256_no_loc"]))
        described = [
            f"{kernel} ({file}:{line}) in {label}"
            for (kernel, file, line), label in identities.items()
        ]
        entry = members[0][1].entry
        if len(fresh_ir_no_loc) > 1:
            collisions.append(
                Collision(
                    digest,
                    entry,
                    "collision",
                    "same cache key, but the freshly generated IR differs: whichever version "
                    "is compiled first is silently reused for the other (cf. upstream issue "
                    "#12429: values a kernel reads that are not hashed)",
                    described,
                )
            )
        elif len(fresh_ir) > 1:
            collisions.append(
                Collision(
                    digest,
                    entry,
                    "debug-info",
                    "same cache key and same code, but the embedded debug locations differ "
                    "(the source path is in the IR but not in the key in this Triton; "
                    "upstream issue #11817)",
                    described,
                )
            )
        elif len({(kernel, file) for kernel, file, _ in identities}) > 1:
            collisions.append(
                Collision(
                    digest,
                    entry,
                    "shared",
                    "kernels with different names or files share one key (identical source "
                    "and specialisation; the name and path are not hashed in this Triton)",
                    described,
                )
            )
    return collisions


# ----------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------
_STATUS_TEXT = {
    "hit": "HIT",
    "miss-same-key": "MISS (key matches A, entry not found)",
    "miss-key-differs": "MISS (key differs from A)",
    "miss-new-kernel": "MISS (not compiled in A)",
    "unknown": "UNKNOWN",
}


def render_text(report: ExplainReport, *, max_detail_lines: int = 24) -> str:
    out: list[str] = []
    total = len(report.kernels)
    misses = report.misses
    out.append(f"A: {report.label_a}")
    out.append(f"B: {report.label_b}")
    out.append("")
    if report.global_findings:
        out.append("Environment differences that affect every kernel:")
        for finding in report.global_findings:
            out.extend(_render_finding(finding, "  ", max_detail_lines))
        out.append("")
    if not misses:
        out.append("No cache misses in B that need explaining.")
    else:
        wasted = sum(k.compile_us or 0 for k in misses)
        out.append(
            f"{len(misses)} compilation(s) in B missed the cache "
            f"({human_duration_us(wasted)} spent compiling)."
        )
    out.append("")
    explained: set[tuple[str, str]] = set()
    for kernel in report.kernels:
        spec = f" [{kernel.specialization}]" if kernel.specialization else ""
        took = f", {human_duration_us(kernel.compile_us)}" if kernel.compile_us is not None else ""
        out.append(f"{kernel.kernel}{spec}  ({kernel.target}{took})")
        out.append(f"  {_STATUS_TEXT.get(kernel.status, kernel.status)}  entry {kernel.entry[:10]}")
        for finding in kernel.findings:
            if report.global_findings and finding.component.startswith(("triton.", "env")):
                out.append(f"  - [{finding.component}] {finding.summary} (see above)")
                continue
            signature = (finding.component, finding.remedy)
            out.extend(
                _render_finding(
                    finding, "  ", max_detail_lines, with_remedy=signature not in explained
                )
            )
            explained.add(signature)
        for note in kernel.notes:
            out.append(f"  note: {note}")
        out.append("")
    if total == 0 and not misses:
        out.append("")
    if report.collisions:
        out.append("Key collisions:")
        for collision in report.collisions:
            out.append(
                f"  [{collision.severity}] entry {collision.entry[:10]}: {collision.summary}"
            )
            for member in collision.members:
                out.append(f"      {member}")
        out.append("")
    for warning in report.warnings:
        out.append(f"warning: {warning}")
    return "\n".join(out).rstrip() + "\n"


def _render_finding(
    finding: Finding, indent: str, max_detail_lines: int, *, with_remedy: bool = True
) -> list[str]:
    import textwrap

    lines = [f"{indent}- [{finding.component}] {finding.summary}"]
    details = finding.details
    if len(details) > max_detail_lines:
        details = [
            *details[:max_detail_lines],
            f"... ({len(finding.details) - max_detail_lines} more)",
        ]
    lines.extend(f"{indent}    {line}" for line in details)
    if finding.remedy and not with_remedy:
        lines.append(f"{indent}  why/fix: as explained above for [{finding.component}]")
    elif finding.remedy:
        wrapped = textwrap.wrap(
            finding.remedy,
            width=92,
            initial_indent=f"{indent}  why/fix: ",
            subsequent_indent=f"{indent}    ",
        )
        lines.extend(wrapped)
    return lines


def render_json(report: ExplainReport) -> str:
    return json.dumps(report.to_dict(), indent=2)


def render_collisions(collisions: list[Collision]) -> str:
    if not collisions:
        return "No key collisions found.\n"
    out = ["Key collisions:"]
    for collision in collisions:
        out.append(f"  [{collision.severity}] entry {collision.entry[:10]}: {collision.summary}")
        out.extend(f"      {member}" for member in collision.members)
    return "\n".join(out) + "\n"
