"""Environment fingerprints: everything checkable about a host that feeds the cache key.

A fingerprint needs Triton to be installed but no GPU, so it can be taken on the
inference host at image build time and compared on the build machine *before*
any kernel is compiled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .explain import Finding

_TRITON_REMEDY = (
    "triton_key() - the Triton version plus the sha256 of its compiler, backend and language "
    "sources and of libtriton - is the first component of every cache key. Entries built with "
    "one install can never be hit by another. Use the same wheel on both hosts."
)


def collect() -> dict[str, Any]:
    from .recorder import collect_fingerprint

    return collect_fingerprint()


def load(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "triton" not in data:
        # also accept a prewarm manifest or a recording session line
        if isinstance(data, dict) and isinstance(data.get("fingerprint"), dict):
            return dict(data["fingerprint"])
        raise ValueError(f"{path}: not a triton-cachekit fingerprint")
    return data


def compare(
    a: dict[str, Any], b: dict[str, Any], *, a_name: str = "A", b_name: str = "B"
) -> list[Finding]:
    """Differences between two fingerprints that change cache keys."""
    findings: list[Finding] = []
    ta, tb = a.get("triton", {}), b.get("triton", {})
    if ta.get("version") != tb.get("version"):
        findings.append(
            Finding(
                "triton.version",
                f"Triton version differs: {ta.get('version')} ({a_name}) vs {tb.get('version')} ({b_name})",
                remedy=_TRITON_REMEDY,
            )
        )
    elif ta.get("key_digest") != tb.get("key_digest"):
        details = []
        fa: dict[str, str] = ta.get("files", {}) or {}
        fb: dict[str, str] = tb.get("files", {}) or {}
        if ta.get("files_verified") and tb.get("files_verified"):
            for name in sorted(set(fa) | set(fb)):
                if fa.get(name) != fb.get(name):
                    state = (
                        "content differs"
                        if name in fa and name in fb
                        else "present on one side only"
                    )
                    details.append(f"{name}: {state}")
        if a.get("implementation") != b.get("implementation"):
            details.append(
                f"Python differs ({a.get('implementation')} vs {b.get('implementation')}): "
                "each Python version has its own Triton wheel with its own libtriton build"
            )
        findings.append(
            Finding(
                "triton.files",
                f"Triton {ta.get('version')} on both sides, but the installed files differ",
                details=details,
                remedy=_TRITON_REMEDY,
            )
        )
    ea: dict[str, str] = (a.get("env", {}) or {}).get("cache_invalidating", {}) or {}
    eb: dict[str, str] = (b.get("env", {}) or {}).get("cache_invalidating", {}) or {}
    if ea != eb:
        details = [
            f"{name}: {ea.get(name, '<unset>')!s} ({a_name}) vs {eb.get(name, '<unset>')!s} ({b_name})"
            for name in sorted(set(ea) | set(eb))
            if ea.get(name) != eb.get(name)
        ]
        findings.append(
            Finding(
                "env",
                "cache-invalidating environment variables differ",
                details=details,
                remedy=(
                    "Every set variable from Triton's cache-invalidating list is part of the key; "
                    "unset and explicitly-set-to-default are different keys. Pass the server's "
                    "values to prewarm with --env NAME=VALUE (or export them)."
                ),
            )
        )
    oa: dict[str, str] = (a.get("env", {}) or {}).get("triton", {}) or {}
    ob: dict[str, str] = (b.get("env", {}) or {}).get("triton", {}) or {}
    option_vars = (
        "TRITON_DEBUG",
        "PTXAS_OPTIONS",
        "TRITON_LIBDEVICE_PATH",
        "TRITON_PTXAS_PATH",
        "TRITON_PTXAS_BLACKWELL_PATH",
        "TRITON_FORCE_DISABLE_FP_FUSION",
        "TRITON_INSTRUMENTATION_MODE",
        "TRITON_FPSAN_HOMOMORPHIC_CASTS",
    )
    details = [
        f"{name}: {oa.get(name, '<unset>')} ({a_name}) vs {ob.get(name, '<unset>')} ({b_name})"
        for name in option_vars
        if oa.get(name) != ob.get(name)
    ]
    if details:
        findings.append(
            Finding(
                "options.env",
                "environment variables that change compiler options or the toolchain differ",
                details=details,
                remedy=(
                    "These are not in Triton's cache-invalidating list but reach the key through "
                    "the options hash (debug, ptx_options, libdevice content, fp fusion, "
                    "instrumentation) or the backend hash (which ptxas runs)."
                ),
            )
        )
    return findings


def path_caveat(fingerprint: dict[str, Any]) -> str | None:
    policy = fingerprint.get("location_in_key") or []
    if "file" in policy:
        return (
            "This Triton hashes the absolute path of each kernel's source file "
            f"(detected by: {fingerprint.get('location_in_key_detected_by')}). The kernels must be "
            "installed at the same absolute path on the server (upstream issue #12367)."
        )
    return None
