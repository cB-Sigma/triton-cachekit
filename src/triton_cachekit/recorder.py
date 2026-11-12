"""Record every component that feeds Triton's cache key, per compilation.

The recorder does not re-implement the key. It wraps the functions Triton itself
uses and writes down what flowed through them:

* ``triton.runtime.cache.get_cache_key(src, backend, backend_options, env_vars)``
  is the single place where the on-disk key string is assembled. We wrap the name
  in every Triton module that imported it and log each argument's contribution
  (``triton_key()``, ``src.hash()``, ``backend.hash()``, ``backend_options.hash()``,
  the environment variables) next to the finished key.
* ``JITCallable.cache_key`` and ``DependenciesFinder._update_hash`` are wrapped to
  learn which other ``@triton.jit`` functions a kernel's hash depends on.
* ``JITFunction._do_compile`` is wrapped to capture the specialisation data that
  ``JITFunction.preload`` can replay (used by ``triton-cachekit prewarm``).
* ``knobs.compilation.listener`` reports hit/miss and stage timings.

If the installed Triton does not look the way these hooks expect, installation
raises :class:`UnsupportedTritonError` instead of silently recording nonsense.

Enable with ``import triton_cachekit.record`` or by running the program under
``triton-cachekit record -- <command>``.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import os
import platform
import re
import sys
import threading
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from . import __version__

SCHEMA_VERSION = 1
ENV_OUTPUT = "TRITON_CACHEKIT_RECORD"
ENV_RECORD_SOURCE = "TRITON_CACHEKIT_RECORD_SOURCE"
DEFAULT_OUTPUT = "triton-cachekit-record.jsonl"
#: Triton releases this recorder has been run against.
VERIFIED_TRITON_VERSIONS: tuple[str, ...] = ("3.8.0", "3.9.0")

_EXPECTED_GET_CACHE_KEY_PARAMS = ["src", "backend", "backend_options", "env_vars"]
_EXPECTED_DO_COMPILE_PARAMS = [
    "self",
    "key",
    "signature",
    "device",
    "constexprs",
    "options",
    "attrs",
    "warmup",
]
_EXPECTED_SERIALIZE_PARAMS = ["name", "signature", "constants", "attrs", "options", "key", "target"]
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_LOC_DEF_LINE = re.compile(r"^#loc\d* = .*$", re.MULTILINE)
_LOC_INLINE = re.compile(r"\s*\bloc\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)")


class UnsupportedTritonError(RuntimeError):
    """The installed Triton's internals differ from what the recorder hooks expect."""


def ir_digest_without_locations(text: str) -> str:
    """Digest of MLIR text with debug locations removed.

    Two compilations of the same kernel from files at different paths differ only
    in ``loc(...)`` attributes; this digest is equal for them.
    """
    stripped = _LOC_DEF_LINE.sub("", text)
    stripped = _LOC_INLINE.sub("", stripped)
    stripped = "\n".join(line.rstrip() for line in stripped.splitlines() if line.strip())
    return hashlib.sha256(stripped.encode("utf-8")).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _jsonable(value: Any, depth: int = 0) -> Any:
    """Best-effort conversion to something ``json.dumps`` accepts."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if depth > 6:
        return repr(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonable(v, depth + 1) for v in value]
    if hasattr(value, "__dict__") and not callable(value):
        return {str(k): _jsonable(v, depth + 1) for k, v in vars(value).items()}
    return repr(value)


def _params(func: Any) -> list[str] | None:
    try:
        return list(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return None


def triton_file_hashes(triton_module: ModuleType) -> tuple[dict[str, str], bool]:
    """Per-file hashes of the files ``triton_key()`` covers, for diagnostics.

    ``triton_key()`` only exposes one concatenated string. To be able to say
    *which* file differs between two installs we hash the same set of files
    ourselves, then check that joining our hashes reproduces Triton's own value.
    The boolean is that check; when it is ``False`` the mapping must not be
    trusted to match what the key covers.
    """
    import pkgutil
    import sysconfig

    from triton.runtime import cache as tcache

    root = os.path.dirname(os.path.abspath(triton_module.__file__ or ""))
    ordered: list[tuple[str, str]] = []

    def add(path: str) -> None:
        digest = _sha256_file(path) or "unreadable"
        ordered.append((os.path.relpath(path, root), digest))

    try:
        add(os.path.abspath(tcache.__file__))
        for sub, prefix in (("compiler", "triton.compiler."), ("backends", "triton.backends.")):
            for lib in pkgutil.walk_packages([os.path.join(root, sub)], prefix=prefix):
                spec = lib.module_finder.find_spec(lib.name)  # type: ignore[call-arg]
                if spec is not None and spec.origin:
                    add(spec.origin)
        ext = str(sysconfig.get_config_var("EXT_SUFFIX")).split(".")[-1]
        add(os.path.join(root, "_C", f"libtriton.{ext}"))
        for lib in pkgutil.walk_packages(
            [os.path.join(root, "language")], prefix="triton.language."
        ):
            spec = lib.module_finder.find_spec(lib.name)  # type: ignore[call-arg]
            if spec is not None and spec.origin:
                add(spec.origin)
        ours = f"{triton_module.__version__}" + "-".join(d for _, d in ordered)
        verified = ours == tcache.triton_key()
    except Exception:
        return dict(ordered), False
    return dict(ordered), bool(verified)


def detect_location_policy(jit_module: ModuleType) -> tuple[list[str], str]:
    """Which parts of a kernel's source location feed its hash in this Triton.

    Triton 3.8.0/3.9.0 append only ``starting_line_number`` to the hash input.
    ``main`` after PR #11832 also hashes the file name and the ``def`` line/column
    (passed to ``DependenciesFinder`` as ``source_location``).
    """
    finder_params = _params(jit_module.DependenciesFinder.__init__) or []
    if "source_location" in finder_params:
        return ["file", "line", "def_line", "def_col"], "DependenciesFinder(source_location=...)"
    try:
        prop = jit_module.JITCallable.__dict__["cache_key"]
        source = inspect.getsource(getattr(prop, "fget", prop))
    except (KeyError, OSError, TypeError):
        return [], "unknown"
    if "starting_line_number" in source:
        return ["line"], "JITCallable.cache_key appends starting_line_number"
    return [], "no location found in JITCallable.cache_key"


def collect_fingerprint() -> dict[str, Any]:
    """Describe the installed Triton and the environment as far as the key cares.

    Needs Triton to be importable but no GPU.
    """
    import triton
    from triton._C.libtriton import get_cache_invalidating_env_vars
    from triton.runtime import cache as tcache
    from triton.runtime import jit as tjit

    files, verified = triton_file_hashes(triton)
    policy, how = detect_location_policy(tjit)
    key = tcache.triton_key()
    return {
        "schema": SCHEMA_VERSION,
        "tool_version": __version__,
        "triton": {
            "version": str(triton.__version__),
            "path": os.path.dirname(os.path.abspath(triton.__file__ or "")),
            "key_digest": _sha256_text(key),
            "files": files,
            "files_verified": verified,
        },
        "location_in_key": policy,
        "location_in_key_detected_by": how,
        "env": {
            "cache_invalidating": dict(get_cache_invalidating_env_vars()),
            "triton": {
                k: v
                for k, v in sorted(os.environ.items())
                if (k.startswith("TRITON_") or k == "PTXAS_OPTIONS")
                and not k.startswith("TRITON_CACHEKIT")
            },
        },
        "cache_dir": _cache_dir(),
        "python": platform.python_version(),
        "implementation": sys.implementation.cache_tag,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _cache_dir() -> str | None:
    try:
        from triton import knobs

        return str(knobs.cache.dir)
    except Exception:
        return None


class Recorder:
    """Writes one JSON object per line; see ``docs/recording-format.md``."""

    def __init__(self, path: str | os.PathLike[str], *, record_source: bool = True) -> None:
        self.path = str(path)
        self.record_source = record_source
        self._lock = threading.RLock()
        self._local = threading.local()
        self._seq = 0
        self._session_pid: int | None = None
        self._fingerprint: dict[str, Any] | None = None
        self._deps: dict[int, tuple[Any, list[Any]]] = {}
        self._seen: dict[str, int] = {}
        self._installed = False
        self._originals: list[tuple[Any, str, Any]] = []
        self._previous_listener: Any = None
        self._listener: Any = None

    # ------------------------------------------------------------------ install
    def install(self) -> None:
        if self._installed:
            return
        try:
            import triton
            from triton import knobs
            from triton.compiler import compiler as tcompiler
            from triton.runtime import cache as tcache
            from triton.runtime import jit as tjit
        except Exception as exc:  # pragma: no cover - depends on the environment
            raise UnsupportedTritonError(f"cannot import Triton: {exc!r}") from exc

        problems = self._check_internals(tcache, tcompiler, tjit, knobs)
        if problems:
            raise UnsupportedTritonError(
                "triton-cachekit cannot hook Triton "
                f"{getattr(triton, '__version__', '?')} at {os.path.dirname(triton.__file__ or '')}: "
                "its internals differ from what the recorder expects.\n  - "
                + "\n  - ".join(problems)
                + f"\nVerified Triton versions: {', '.join(VERIFIED_TRITON_VERSIONS)}. "
                "Nothing was recorded. Please report this at "
                "https://github.com/cb-Sigma/triton-cachekit/issues."
            )

        self._fingerprint = collect_fingerprint()
        original = tcache.get_cache_key
        wrapped = self._wrap_get_cache_key(original)
        for module in (tcache, tcompiler, tjit):
            if getattr(module, "get_cache_key", None) is original:
                self._patch(module, "get_cache_key", wrapped)
        self._patch_dependency_tracking(tjit)
        self._patch_do_compile(tjit)

        self._previous_listener = knobs.compilation.listener
        self._listener = self._make_listener(self._previous_listener)
        knobs.compilation.listener = self._listener
        self._installed = True
        self._write_session()

    def uninstall(self) -> None:
        """Undo :meth:`install` (used by tests and by in-process prewarm workers)."""
        if not self._installed:
            return
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)
        self._originals.clear()
        with contextlib.suppress(Exception):
            from triton import knobs

            if knobs.compilation.listener is self._listener:
                knobs.compilation.listener = self._previous_listener
        self._installed = False

    def _patch(self, owner: Any, name: str, replacement: Any) -> None:
        self._originals.append((owner, name, owner.__dict__[name]))
        setattr(owner, name, replacement)

    @staticmethod
    def _check_internals(tcache: Any, tcompiler: Any, tjit: Any, knobs: Any) -> list[str]:
        problems: list[str] = []
        get_cache_key = getattr(tcache, "get_cache_key", None)
        if get_cache_key is None:
            problems.append("triton.runtime.cache.get_cache_key does not exist")
        elif _params(get_cache_key) != _EXPECTED_GET_CACHE_KEY_PARAMS:
            problems.append(
                "triton.runtime.cache.get_cache_key has parameters "
                f"{_params(get_cache_key)}, expected {_EXPECTED_GET_CACHE_KEY_PARAMS}"
            )
        if not callable(getattr(tcache, "triton_key", None)):
            problems.append("triton.runtime.cache.triton_key does not exist")
        if (
            get_cache_key is not None
            and getattr(tcompiler, "get_cache_key", None) is not get_cache_key
        ):
            problems.append(
                "triton.compiler.compiler does not use triton.runtime.cache.get_cache_key "
                "(the key is assembled somewhere else)"
            )
        compile_fn = getattr(tcompiler, "compile", None)
        try:
            compile_src = inspect.getsource(compile_fn) if compile_fn is not None else ""
        except (OSError, TypeError):
            compile_src = ""
        if "get_cache_key(" not in compile_src or "get_cache_manager(" not in compile_src:
            problems.append(
                "triton.compiler.compiler.compile no longer calls get_cache_key()/get_cache_manager()"
            )
        if not hasattr(getattr(knobs, "compilation", None), "listener"):
            problems.append("triton.knobs.compilation.listener does not exist")
        for name in ("src", "hash"):
            if not hasattr(getattr(tcompiler, "ASTSource", None), "hash"):
                problems.append(f"triton.compiler.compiler.ASTSource.{name} is missing")
                break
        jit_callable = getattr(tjit, "JITCallable", None)
        if not isinstance(getattr(jit_callable, "__dict__", {}).get("cache_key"), property):
            problems.append("triton.runtime.jit.JITCallable.cache_key is not a property")
        finder = getattr(tjit, "DependenciesFinder", None)
        if _params(getattr(finder, "_update_hash", None)) != ["self", "func"]:
            problems.append(
                "triton.runtime.jit.DependenciesFinder._update_hash(self, func) is missing"
            )
        jit_function = getattr(tjit, "JITFunction", None)
        if _params(getattr(jit_function, "_do_compile", None)) != _EXPECTED_DO_COMPILE_PARAMS:
            problems.append(
                "triton.runtime.jit.JITFunction._do_compile has parameters "
                f"{_params(getattr(jit_function, '_do_compile', None))}, "
                f"expected {_EXPECTED_DO_COMPILE_PARAMS}"
            )
        serialize = getattr(tjit, "serialize_specialization_data", None)
        if _params(serialize) != _EXPECTED_SERIALIZE_PARAMS:
            problems.append(
                "triton.runtime.jit.serialize_specialization_data has parameters "
                f"{_params(serialize)}, expected {_EXPECTED_SERIALIZE_PARAMS}"
            )
        return problems

    # ------------------------------------------------------------------- hooks
    def _wrap_get_cache_key(self, original: Callable[..., str]) -> Callable[..., str]:
        recorder = self

        @functools.wraps(original)
        def get_cache_key(src: Any, backend: Any, backend_options: Any, env_vars: Any) -> str:
            key = original(src, backend, backend_options, env_vars)
            try:
                recorder._on_key(src, backend, backend_options, env_vars, key)
            except Exception as exc:  # never break a compilation
                recorder._internal_error("get_cache_key", exc)
            return key

        return get_cache_key

    def _patch_dependency_tracking(self, tjit: Any) -> None:
        recorder = self
        original_prop: property = tjit.JITCallable.__dict__["cache_key"]
        original_fget = original_prop.fget
        assert original_fget is not None

        def cache_key(self: Any) -> Any:
            if getattr(self, "hash", None) is not None:
                return original_fget(self)
            stack = recorder._stack()
            recorder._deps[id(self)] = (self, [])
            stack.append(self)
            try:
                return original_fget(self)
            finally:
                stack.pop()

        self._patch(tjit.JITCallable, "cache_key", property(cache_key, doc=original_prop.__doc__))

        original_update = tjit.DependenciesFinder._update_hash

        @functools.wraps(original_update)
        def _update_hash(self: Any, func: Any) -> Any:
            stack = recorder._stack()
            if stack:
                owner = recorder._deps.get(id(stack[-1]))
                if owner is not None and all(func is not d for d in owner[1]):
                    owner[1].append(func)
            return original_update(self, func)

        self._patch(tjit.DependenciesFinder, "_update_hash", _update_hash)

    def _patch_do_compile(self, tjit: Any) -> None:
        recorder = self
        original = tjit.JITFunction._do_compile
        serialize = tjit.serialize_specialization_data
        get_full_name = tjit.get_full_name

        @functools.wraps(original)
        def _do_compile(
            self: Any,
            key: Any,
            signature: Any,
            device: Any,
            constexprs: Any,
            options: Any,
            attrs: Any,
            warmup: Any,
        ) -> Any:
            context: dict[str, Any] = {"warmup": bool(warmup), "specialization_data": None}
            try:
                target = self.device_caches[device][2]
                context["specialization_data"] = serialize(
                    get_full_name(self.fn), signature, constexprs, attrs, options, key, target
                )
            except Exception as exc:
                context["specialization_error"] = f"{type(exc).__name__}: {exc}"
            previous = getattr(recorder._local, "jit", None)
            recorder._local.jit = context
            try:
                return original(self, key, signature, device, constexprs, options, attrs, warmup)
            finally:
                recorder._local.jit = previous

        self._patch(tjit.JITFunction, "_do_compile", _do_compile)

    def _make_listener(self, previous: Any) -> Callable[..., None]:
        recorder = self

        def listener(
            *, src: Any, metadata: Any, metadata_group: Any, times: Any, cache_hit: bool
        ) -> None:
            try:
                recorder._on_outcome(src, metadata, metadata_group, times, cache_hit)
            except Exception as exc:
                recorder._internal_error("compilation listener", exc)
            if previous is not None:
                previous(
                    src=src,
                    metadata=metadata,
                    metadata_group=metadata_group,
                    times=times,
                    cache_hit=cache_hit,
                )

        return listener

    def _stack(self) -> list[Any]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    # --------------------------------------------------------------- recording
    def _fn_info(self, fn: Any, seen: set[int]) -> dict[str, Any]:
        info: dict[str, Any] = {
            "name": getattr(fn, "_fn_name", None) or getattr(fn, "__name__", repr(fn)),
            "cache_key": fn.cache_key,
            "file": getattr(fn, "file_name", None),
            "line": getattr(fn, "starting_line_number", None),
            "def_line": getattr(fn, "def_file_line_number", None),
            "def_col": getattr(fn, "def_file_col_number", None),
            "noinline": bool(getattr(fn, "noinline", False)),
        }
        src_text = getattr(fn, "src", None)
        if isinstance(src_text, str):
            info["src_sha256"] = _sha256_text(src_text)
            if self.record_source:
                info["src"] = src_text
        constexpr_globals = []
        for (name, _), (value, _) in dict(getattr(fn, "used_global_vals", {})).items():
            if type(value).__name__ == "constexpr":
                constexpr_globals.append([str(name), repr(value)])
        info["constexpr_globals"] = sorted(constexpr_globals)
        known = self._deps.get(id(fn))
        if known is None or known[0] is not fn:
            # The hash was computed before the recorder was installed.
            info["deps"] = None
        elif id(fn) in seen:
            info["deps"] = []
        else:
            seen = seen | {id(fn)}
            info["deps"] = [self._fn_info(dep, seen) for dep in known[1]]
        return info

    def _source_component(self, src: Any) -> dict[str, Any]:
        fn = getattr(src, "fn", None)
        if fn is None or not hasattr(src, "signature"):
            return {
                "kind": "ir",
                "hash": src.hash(),
                "path": str(getattr(src, "path", "")),
                "ext": getattr(src, "ext", None),
            }
        constants = []
        arg_names = list(getattr(fn, "arg_names", []))
        for path, value in sorted(dict(getattr(src, "constants", {})).items()):
            name = None
            if isinstance(path, tuple) and len(path) == 1 and path[0] < len(arg_names):
                name = arg_names[path[0]]
            constants.append(
                {
                    "path": list(path) if isinstance(path, tuple) else [path],
                    "name": name,
                    "value": repr(value),
                    # mirrors ASTSource.hash: cache_key for JIT callables, str() otherwise
                    "key": value.cache_key if hasattr(value, "cache_key") else str(value),
                }
            )
        return {
            "kind": "ast",
            "hash": src.hash(),
            "fn": self._fn_info(fn, set()),
            "signature": {str(k): str(v) for k, v in dict(src.signature).items()},
            "constants": constants,
            "attrs": str(getattr(src, "attrs", {})),
        }

    @staticmethod
    def _options_component(backend_options: Any) -> dict[str, Any]:
        values = dict(vars(backend_options))
        libs: dict[str, Any] = {}
        raw_libs = values.get("extern_libs") or ()
        with contextlib.suppress(Exception):
            for name, path in dict(raw_libs).items():
                libs[str(name)] = {"path": str(path), "sha256": _sha256_file(str(path))}
        return {
            "hash": backend_options.hash(),
            "values": _jsonable(values),
            "extern_libs": libs,
        }

    def _on_key(
        self, src: Any, backend: Any, backend_options: Any, env_vars: Any, key: str
    ) -> None:
        from triton import knobs

        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        with self._lock:
            pending = self._seen.get(digest, 0)
            if pending > 0:
                # async compilation asks for the key twice for the same build
                return
            self._seen[digest] = pending + 1
        assert self._fingerprint is not None
        target = getattr(backend, "target", None)
        fn = getattr(src, "fn", None)
        jit = getattr(self._local, "jit", None) or {}
        record: dict[str, Any] = {
            "type": "compile",
            "kernel": getattr(fn, "_fn_name", None) or str(getattr(src, "name", "?")),
            "name": str(getattr(src, "name", "?")),
            "hash": digest,
            "entry": _base32(digest),
            "cache_dir": _cache_dir(),
            "components": {
                "triton": {
                    "version": self._fingerprint["triton"]["version"],
                    "key_digest": self._fingerprint["triton"]["key_digest"],
                },
                "source": self._source_component(src),
                "backend": {
                    "hash": str(backend.hash()),
                    "class": type(backend).__name__,
                    "target": _jsonable(target),
                },
                "options": self._options_component(backend_options),
                "env": {str(k): str(v) for k, v in dict(env_vars).items()},
            },
            "custom_pipeline": knobs.runtime.add_stages_inspection_hook is not None,
            "via_jit": bool(jit),
            "warmup": jit.get("warmup"),
            "specialization_data": jit.get("specialization_data"),
        }
        if "specialization_error" in jit:
            record["specialization_error"] = jit["specialization_error"]
        self._write(record)

    def _on_outcome(
        self, src: Any, metadata: Any, metadata_group: Any, times: Any, cache_hit: bool
    ) -> None:
        meta = dict(metadata)
        digest = str(meta.get("hash"))
        with self._lock:
            known = digest in self._seen
            if known:
                self._seen[digest] = 0
        if not known:
            from triton import knobs

            if knobs.runtime.add_stages_inspection_hook is None:
                message = (
                    f"Triton reported a compilation for cache hash {digest[:16]}... that does not "
                    "match any key seen by the recorder: the key derivation in this Triton "
                    "differs from what triton-cachekit expects. Do not trust this recording."
                )
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                self._write({"type": "warning", "message": message})
        artifacts: dict[str, Any] = {}
        for name, path in dict(metadata_group).items():
            ext = str(name).rpartition(".")[2]
            if str(name).startswith("__grp__"):
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            item: dict[str, Any] = {"size": size}
            if size <= _MAX_ARTIFACT_BYTES:
                item["sha256"] = _sha256_file(str(path))
                if ext in ("source", "ttir"):
                    with contextlib.suppress(OSError, UnicodeDecodeError):
                        item["sha256_no_loc"] = ir_digest_without_locations(
                            Path(path).read_text(encoding="utf-8")
                        )
            artifacts[ext] = item
        record = {
            "type": "outcome",
            "hash": digest,
            "kernel": getattr(getattr(src, "fn", None), "_fn_name", None)
            or str(getattr(src, "name", "?")),
            "cache_hit": bool(cache_hit),
            "times_us": {
                "total": int(times.total),
                "ir_initialization": int(times.ir_initialization),
                "lowering": {str(stage): int(us) for stage, us in times.lowering_stages},
                "store_results": int(times.store_results),
            },
            "artifacts": artifacts,
        }
        self._write(record)

    def _internal_error(self, where: str, exc: BaseException) -> None:
        message = f"triton-cachekit recorder failed in {where}: {type(exc).__name__}: {exc}"
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        with contextlib.suppress(Exception):
            self._write({"type": "warning", "message": message})

    # ------------------------------------------------------------------- output
    def _write_session(self) -> None:
        assert self._fingerprint is not None
        record = dict(self._fingerprint)
        record.update(
            {
                "type": "session",
                "argv": list(sys.argv),
                "cwd": os.getcwd(),
                "hostname": platform.node(),
                "record_source": self.record_source,
                "verified_triton_versions": list(VERIFIED_TRITON_VERSIONS),
                "triton_version_verified": record["triton"]["version"] in VERIFIED_TRITON_VERSIONS,
            }
        )
        self._session_pid = os.getpid()
        self._emit(record)

    def _write(self, record: dict[str, Any]) -> None:
        with self._lock:
            if self._session_pid != os.getpid():
                # forked child: give it its own session header
                parent = self._session_pid
                self._write_session()
                self._emit({"type": "note", "message": f"forked from pid {parent}"})
            self._emit(record)

    def _emit(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            record = {"pid": os.getpid(), "seq": self._seq, "time": time.time(), **record}
            line = json.dumps(record, default=repr, sort_keys=False) + "\n"
            path = self.path.replace("{pid}", str(os.getpid()))
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)


def _base32(hex_digest: str) -> str:
    import base64

    return base64.b32encode(bytes.fromhex(hex_digest)).decode("ascii").rstrip("=")


# --------------------------------------------------------------------- module API
_active: Recorder | None = None


def resolve_output_path(path: str | os.PathLike[str] | None = None) -> str:
    chosen = str(path) if path is not None else os.environ.get(ENV_OUTPUT) or DEFAULT_OUTPUT
    if chosen.endswith(os.sep) or os.path.isdir(chosen):
        chosen = os.path.join(chosen, "record-{pid}.jsonl")
    return os.path.abspath(chosen)


def install(
    path: str | os.PathLike[str] | None = None, *, record_source: bool | None = None
) -> Recorder:
    """Install the recorder in this process (idempotent) and return it."""
    global _active
    if _active is not None:
        return _active
    if record_source is None:
        record_source = os.environ.get(ENV_RECORD_SOURCE, "1").lower() not in ("0", "false", "no")
    recorder = Recorder(resolve_output_path(path), record_source=record_source)
    recorder.install()
    _active = recorder
    return recorder


def uninstall() -> None:
    global _active
    if _active is not None:
        _active.uninstall()
        _active = None


def active() -> Recorder | None:
    return _active


class _WrappedLoader(importlib.abc.Loader):
    def __init__(self, inner: Any, callback: Callable[[], None]) -> None:
        self._inner = inner
        self._callback = callback

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        result: ModuleType | None = self._inner.create_module(spec)
        return result

    def exec_module(self, module: ModuleType) -> None:
        self._inner.exec_module(module)
        self._callback()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _TritonImportWatcher(importlib.abc.MetaPathFinder):
    """Installs the recorder right after ``import triton`` completes."""

    def __init__(self, callback: Callable[[], None]) -> None:
        self._callback = callback

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != "triton":
            return None
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(self)
        spec = importlib.util.find_spec("triton")
        if spec is None or spec.loader is None:
            return None
        spec.loader = _WrappedLoader(spec.loader, self._callback)
        return spec


def install_when_triton_is_imported() -> None:
    """Defer :func:`install` until the host program imports Triton.

    Used by ``triton-cachekit record -- <command>`` so that processes which never
    touch Triton (shell wrappers, launchers) are left alone.
    """
    if "triton" in sys.modules and hasattr(sys.modules["triton"], "compile"):
        install()
        return
    if any(isinstance(f, _TritonImportWatcher) for f in sys.meta_path):
        return

    def callback() -> None:
        install()

    sys.meta_path.insert(0, _TritonImportWatcher(callback))
