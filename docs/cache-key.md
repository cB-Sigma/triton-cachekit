# How Triton's compilation cache key is built

This is a reading of the Triton source, not official documentation. Line numbers refer to the
**3.9.0 wheel** (`pip install triton==3.9.0`, files under `site-packages/triton/`, which correspond
to `python/triton/` in the repository). Differences in 3.8.0 and in `main` at commit `8a439de`
are called out. If you find a mistake, please open an issue.

Triton has two caches. Only the second one is on disk.

1. **In-process**: `JITFunction.device_caches[device]` maps
   `str(specialization) + str(options)` to a compiled kernel (`runtime/jit.py`,
   `compute_cache_key`, line 611). It lives as long as the process.
2. **On disk**: `compiler/compiler.py`, `compile()` (line 226) builds a key string, hashes it and
   looks in `TRITON_CACHE_DIR`. Everything below is about this one.

## The key string

`runtime/cache.py`, `get_cache_key` (line 319):

```python
key = f"{triton_key()}-{src.hash()}-{backend.hash()}-{backend_options.hash()}-{str(sorted(env_vars.items()))}"
```

`compile()` then does (lines 246-251):

```python
key = get_cache_key(src, backend, options, env_vars=env_vars)
if knobs.runtime.add_stages_inspection_hook is not None:  # custom pipelines add their own key
    key += inspect_stages_key
hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
fn_cache_manager = get_cache_manager(hash)  # directory = _base32(hash)
```

`_base32` (`runtime/cache.py` line 253) is `base64.b32encode(bytes.fromhex(hash))` with the `=`
padding stripped: 52 characters for a SHA-256.

### 1. `triton_key()` - the installation

`runtime/cache.py`, line 284. The string is the Triton version followed by SHA-256 digests,
joined with `-`, of:

- `runtime/cache.py` itself,
- every module found by `pkgutil.walk_packages` under `compiler/` and `backends/`,
- the `_C/libtriton.<ext>` shared library,
- every module under `language/`.

Consequences: only a byte-identical install produces the same keys. A different release, a wheel
for another Python minor version, or an edited file changes every key. Not covered: `runtime/jit.py`
and the rest of `runtime/`, the bundled `ptxas` binary (see `backend.hash()`), `libdevice` (see
options).

The function is `lru_cache`d, so files are hashed once per process.

### 2. `src.hash()` - the kernel and its specialisation

`compiler/compiler.py`, `ASTSource.hash` (line 71):

```python
sorted_sig = [v for k, v in sorted(self.signature.items())]
get_key = lambda x: x.cache_key if hasattr(x, "cache_key") else str(x)
constants_key = "-".join([get_key(v) for k, v in sorted(self.constants.items())])
key = f"{self.fn.cache_key}-{str(self.attrs)}-{sorted_sig}-{constants_key}"
```

- `signature`: one type string per parameter (`*fp32`, `i32`, `constexpr`, ...).
- `constants`: the values of constexpr parameters. `str(x)` is used, so `1` and `1.0` are distinct
  here (the in-process cache is where they can be confused, issue #12239).
- `attrs`: per-argument attributes, in practice `tt.divisibility = 16`.
- `fn.cache_key`: the function hash, below.

How arguments become a signature and attributes is decided in C++
(`python/src/specialize.cc`): tensors by `dtype` and `data_ptr() % 16`; Python integers become
`i32`/`i64`/`u64`, get the divisibility attribute when `value % 16 == 0`, and are turned into a
constexpr when the value is exactly `1`; floats and bools are not value-specialised. Parameters
listed in `do_not_specialize` / `do_not_specialize_on_alignment` opt out.

For IR sources (`triton.compile("file.ttir")`), `IRSource.hash` (line 114) hashes the file
extension and content instead.

### 3. The function hash - `JITCallable.cache_key`

`runtime/jit.py`, line 523. A SHA-256 over:

- the source text from the `def` line on (`self.src`, decorators removed, dedented);
- for every `@triton.jit` function referenced in the body, that function's own `cache_key` plus its
  `noinline` flag (`DependenciesFinder._update_hash`, line 98), recursively;
- for `@tl.aggregate` classes, their annotations and hashed attributes;
- **source location**, which differs by version:

  | Version | What is hashed | Where |
  | --- | --- | --- |
  | 3.8.0, 3.9.0 | `starting_line_number` only (the line of the first decorator) | `self.hash = dependencies_finder.ret + str(self.starting_line_number)` (line 535; line 528 in 3.8.0) |
  | `main` @ 8a439de (after PR #11832) | `(file_name, starting_line_number, def_file_line_number, def_file_col_number)` | passed to `DependenciesFinder(source_location=...)` and hashed in its constructor |

- the global `tl.constexpr` values the function (or anything it calls) reads by name, sorted
  (`constexpr_globals`).

Not hashed, and therefore sources of collisions (issue #12429): values reached through attribute
access on a module or object, default arguments, closures over non-constexpr values. The test suite
reproduces the module-attribute case on 3.8.0 and 3.9.0.

`TRITON_DISABLE_LINE_INFO=1` does not remove the line number (or, on `main`, the path) from the
hash (issues #11817, #12367).

### 4. `backend.hash()`

- CUDA, `backends/nvidia/compiler.py` line 675: `f'{get_ptxas_version(arch)}-{self.target.arch}'`.
  `get_ptxas_version` (line 49) is the full output of `ptxas --version` for the `ptxas` Triton
  would use for that architecture (the bundled one, `ptxas-blackwell` for newer architectures, or
  whatever `TRITON_PTXAS_PATH` / `TRITON_PTXAS_BLACKWELL_PATH` point to).
- AMD, `backends/amd/compiler.py` line 843: `f'{self.target}-{get_amd_codegen_revision()}'`.

### 5. `backend_options.hash()`

`CUDAOptions.hash` (`backends/nvidia/compiler.py` line 167) and `HIPOptions.hash`
(`backends/amd/compiler.py` line 326): every dataclass field as `name-value`, sorted by name, with
`extern_libs` replaced by `(name, sha256(file content))`. So library *paths* do not matter, their
content does.

Options are built by `parse_options` (line 216 / 356). Several fields take their default from the
environment at that moment, which makes those variables part of the key even though they are not
in the list of section 6:

| Field | Source |
| --- | --- |
| `arch` | `TRITON_OVERRIDE_ARCH` or `sm<capability>` |
| `enable_fp_fusion` | `knobs.language.fp_fusion_enabled(...)`: `TRITON_DEFAULT_FP_FUSION`, `TRITON_FORCE_DISABLE_FP_FUSION` (3.9.0) |
| `debug` | `TRITON_DEBUG` (via `JITFunction.run`) |
| `ptx_options` | `PTXAS_OPTIONS` |
| `extern_libs["libdevice"]` | `TRITON_LIBDEVICE_PATH` (content hashed) |
| `instrumentation_mode`, `fpsan_homomorphic_casts` | `TRITON_INSTRUMENTATION_MODE`, `TRITON_FPSAN_HOMOMORPHIC_CASTS` |

### 6. Environment variables

`get_cache_invalidating_env_vars()` is implemented in C++ (`python/src/ir.cc`). It walks
`CACHE_INVALIDATING_ENV_VARS` (`include/triton/Tools/Sys/GetEnv.h`) and returns each variable that
is set to a non-empty value; values that parse as booleans are normalised to `"true"`/`"false"`.

Consequences: `VAR=1` and `VAR=true` are the same key, but an unset variable and one set to its
default value are not (issue #11817). The list is compiled into `libtriton`, so it can differ per
release; this document read it at `main` @ 8a439de, where it has 43 entries
(`TRITON_DEFAULT_FP_FUSION`, `TRITON_DISABLE_LINE_INFO`, `TRITON_OVERRIDE_ARCH`, `DISABLE_LLVM_OPT`,
`DISABLE_PTXAS_OPT`, `MLIR_ENABLE_DUMP`, `USE_IR_LOC`, the `AMDGCN_*` and `TRITON_HIP_*` switches,
...). The recorder asks the installed Triton rather than relying on that list.

## What is in an entry

Written by `compile()` through `FileCacheManager` (`runtime/cache.py`):

```text
<TRITON_CACHE_DIR>/<base32 key>/
    <kernel>.source  .ttir  .ttgir  .llir  .ptx  .cubin      (NVIDIA)
    <kernel>.source  .ttir  .ttgir  .llir  .amdgcn  .hsaco   (AMD)
    <kernel>.json                 metadata
    __grp__<kernel>.json          {"child_paths": ...}
```

- `<kernel>` is the function's `__name__`, truncated to 150 characters.
- `TRITON_STORE_BINARY_ONLY=1` keeps only the binary and the JSON files.
- Files are written to `tmp.pid_<pid>_<uuid>/` and renamed into place.
- The metadata JSON (built from line 282 on) holds `hash`, `target`, every option, every set
  cache-invalidating variable under its own name, `triton_version`, and fields added by the
  backend (`name`, `shared`, `global_scratch_size`, ...). It does **not** record the kernel's
  source, file, specialisation or the Triton file hashes: from an entry alone you cannot tell
  what was hashed. That is why `triton-cachekit explain` needs recordings.

### When is an entry a hit?

`compile()` line 265: `metadata_group = fn_cache_manager.get_group(f"{name}.json") or {}`; it is a
hit when the group contains the metadata file (and `TRITON_ALWAYS_COMPILE` is not set).
`FileCacheManager.get_group` (line 73) returns `None`, a miss, when the group file is missing, is
not valid JSON, has no `child_paths`, or any listed child does not exist.

| Version | `child_paths` | Effect of moving the cache directory |
| --- | --- | --- |
| 3.8.0, 3.9.0 | `{"name": "/absolute/path"}` | Every entry misses once the original path is gone (issue #11999); if the original still exists, files are loaded from *there* |
| `main` @ 8a439de | `["name", ...]`, joined with the entry directory | Position-independent |

`triton-cachekit relocate` rewrites the absolute form for the directory's current location.

## Remote cache

`TRITON_CACHE_MANAGER=triton.runtime.cache:RemoteCacheManager` with
`TRITON_REMOTE_CACHE_BACKEND=triton.runtime.cache:RedisRemoteCacheBackend` stores each file under
`TRITON_REDIS_KEY_FORMAT` (default `triton:{key}:{filename}`) in Redis (`TRITON_REDIS_HOST`,
`TRITON_REDIS_PORT`) and materialises files into the local cache directory on access. The key is
computed exactly as above; group files hold a sorted list of names. triton-cachekit does not talk
to Redis.

## Other things stored in the cache directory

- Autotuner results: `<fn>.autotune.json`, keyed by `triton_key()`, the backend hash, the function hash,
  the environment variables, the tuning key and the candidate configs (`runtime/autotuner.py`, `check_disk_cache`), only when
  `TRITON_CACHE_AUTOTUNING` is on.
- Compiled launcher/driver stubs (`*.so`), written through `runtime/build.py` on first launch on
  a GPU host.

`inspect` lists both kinds without interpreting them.
