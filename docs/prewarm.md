# Prewarming a cache without a GPU

> **Unverified on real GPUs.** Everything on this page was exercised on a CPU-only machine. That
> the resulting cache is hit by a process using a real GPU driver has not been tested by this
> project. Use `verify-hit` (below) on your hardware and please report the result.

## How it works

Before compiling, Triton asks the active driver for the current device, stream and target
(`JITFunction.run` and `create_binder` in `runtime/jit.py`). `prewarm` installs a driver object
whose answers are constants (`triton_cachekit.fakedriver`), with the target you name, and then
uses Triton's own entry points:

- manifest specs call `kernel.warmup(*args, grid=(1,), **kwargs)`, the documented way to compile
  without launching;
- recording specs call `kernel.preload(specialization_data)` with the data Triton serialised
  during the recorded run.

From there the normal pipeline runs: frontend, MLIR passes, LLVM, then the bundled `ptxas` (NVIDIA)
or the AMD code generator, all on the CPU. Nothing is launched.

Each kernel is compiled in its own subprocess (`python -m triton_cachekit._worker`), so a compiler
crash or hang costs one job; `--timeout` (default 600 s per kernel) kills stuck jobs and `-j N`
runs N at a time.

## Targets

`<backend>:<arch>[:<warp-size>]`, the same convention as Triton's `tools/compile.py`:

| Example | Meaning |
| --- | --- |
| `cuda:80` | NVIDIA compute capability 8.0. The real driver computes `major * 10 + minor` and always reports warp size 32, which is the default here. |
| `cuda:89`, `cuda:90`, `cuda:100`, `cuda:120` | other NVIDIA generations |
| `hip:gfx942:64` | AMD; the warp size is read from the device by the real driver, so it must be given explicitly |

The example manifest was compiled for `cuda` 75, 80, 89, 90, 100, 120 and `hip` gfx942, gfx950,
gfx1100 and gfx1250 with the 3.9.0 wheel on a CPU-only machine (complete entries, binaries not
executed); the automated tests cover `cuda:80`, `cuda:90` and `hip:gfx942:64`. Some kernel features are architecture-specific and will fail to
compile for older targets exactly as they would on that GPU.

## Manifest

```toml
[prewarm]
target = "cuda:80"                         # or pass --target
env = { TRITON_DEFAULT_FP_FUSION = "0" }   # environment for the compile workers

[[kernel]]
file = "kernels.py"       # path relative to the manifest, or: module = "package.module"
name = "rms_norm_rows"    # attribute in that module (a @triton.jit or @triton.autotune object)

  [[kernel.spec]]
  args = ["tensor:float16", "tensor:float16", "tensor:float16", 4096]
  kwargs = { BLOCK = 4096, num_warps = 4 }
```

Arguments:

| Manifest value | Passed to `warmup` as |
| --- | --- |
| `"tensor:<dtype>"` | a tensor stand-in with that dtype and a 16-byte-aligned pointer. dtype names are torch's: `float16`, `bfloat16`, `float32`, `int32`, `int64`, `bool`, ... |
| `"tensor:<dtype>:unaligned"` | the same with a pointer that is not divisible by 16 (tensor views with odd offsets) |
| `{ tensor = "<dtype>", aligned = false }` | table form of the above |
| numbers, booleans, strings | themselves |
| lists | tuples |
| `{ value = ... }` | the value, for strings that would otherwise be read as `tensor:` |

Use realistic integers: Triton specialises on an integer being exactly `1` and on it being
divisible by 16. `kwargs` holds constexpr arguments and compile options (`num_warps`,
`num_stages`, ...).

Loading a kernel imports its module, which runs the module's top-level code. With `file = ...` the
file's directory is put on `sys.path`, like Triton's own AOT tool does.

Autotuned kernels: every configuration is compiled (`Autotuner.warmup`).

## From a recording

Guessing specialisations is error-prone. If you can run the real workload once (on a GPU machine,
or anywhere the application runs), record it and replay:

```console
gpu$   triton-cachekit record -o run.jsonl -- python serve.py
build$ triton-cachekit prewarm --from-recording run.jsonl --cache-dir ./cache
```

The replay uses the recorded target. Kernels are located by their recorded module name (falling
back to the recorded file path), so the same code must be importable on the build machine. Kernels
defined inside functions cannot be replayed and are listed as skipped. The test suite checks that
a replay reproduces exactly the keys of the recorded run when the code and environment are the same
(with the recorded run itself using the fake driver).

## Output

- the cache directory;
- `prewarm-manifest.json`: target, driver (`"fake"`), a fingerprint of the Triton install and
  environment, one row per entry written (`kernel`, `hash`, `entry`, `status`, compile time), the
  jobs (so `verify-hit` can re-run them), `"verified_on_real_gpu": false` and the list of caveats;
- `prewarm-manifest.rec/`: a key recording of the prewarm run, for `explain`.

Exit status 1 if any specialisation failed to compile or timed out.

## Checking what can be checked before compiling

```console
server$ triton-cachekit fingerprint -o server.json
build$  triton-cachekit prewarm ... --server-fingerprint server.json
```

The fingerprint needs Triton but no GPU. `prewarm` compares the Triton version, the per-file
hashes behind `triton_key()`, the cache-invalidating environment variables (after applying
`--env`), and the variables that change option defaults or the toolchain. On a mismatch nothing is
compiled (override with `--force`). If the server's Triton hashes source paths, a note reminds you
that kernels must live at the same absolute path.

What the fingerprint cannot check: the target the real driver will report, the specialisations the
application will request, and the kernel source on the server.

## Why a prewarmed entry can still miss

| Component | Can differ because |
| --- | --- |
| Triton install | another version, another Python version's wheel, a patched file |
| Target | the driver reports another capability / arch / warp size than you named |
| `ptxas` | `TRITON_PTXAS_PATH` points somewhere else on the server |
| Environment | a cache-invalidating variable set on one side only; `TRITON_DEBUG`, `PTXAS_OPTIONS`, ... |
| Source | not the same bytes, or shifted by a line; on `main`, a different install path |
| Specialisation | other dtypes or constexpr values; integers that are 1 or not multiples of 16; unaligned tensor views |
| Group file | 3.8.0/3.9.0: cache mounted at another path without `relocate`/`unpack` |

## `verify-hit`: testing it on the GPU host

```console
gpu$ triton-cachekit unpack cache.tar.gz "$TRITON_CACHE_DIR"
gpu$ triton-cachekit verify-hit --manifest prewarm-manifest.json --prewarm-recording prewarm-manifest.rec
```

It re-runs the manifest's jobs in subprocesses with the real driver, in warm-up mode, and prints
one row per specialisation:

- `HIT`: Triton found and loaded the prewarmed entry.
- `MISS`, "this host computes a different key": followed by an `explain` report against the prewarm
  recording naming the component.
- `MISS`, "same key, but no loadable entry": the entry is not in this cache directory, or its
  group file points elsewhere (`inspect` will show `stale-paths`).

Exit status: 0 all hit, 1 any miss, 2 could not run (for example no GPU driver). Environment
variables from the manifest are deliberately *not* applied: the point is to test the host as it
is. Misses are compiled and written to the cache.

This still uses stand-in tensors, so it validates the key machinery, not that your application
requests the same specialisations. For that, record the real application on the GPU host and run
`triton-cachekit explain prewarm-manifest.rec server.jsonl`.
