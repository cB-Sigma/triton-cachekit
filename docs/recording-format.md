# Recording format

A recording is a JSON Lines file (one JSON object per line), written by
`triton_cachekit.recorder`. Several processes may append to the same file; every record carries
`pid`, a per-process sequence number `seq` and a wall-clock `time`. With `{pid}` in the file name,
or a directory as the output, each process gets its own file. `explain`, `stats` and
`prewarm --from-recording` accept a file or a directory of `*.jsonl` files.

The current schema version is `1`. Readers reject other versions instead of guessing.

## `session` - once per process, when the recorder is installed

| Field | Meaning |
| --- | --- |
| `schema`, `tool_version` | format and triton-cachekit version |
| `triton.version`, `triton.path` | installed Triton |
| `triton.key_digest` | SHA-256 of Triton's own `triton_key()` string |
| `triton.files` | relative path -> SHA-256 for every file `triton_key()` covers |
| `triton.files_verified` | `true` when joining those digests reproduces `triton_key()` exactly; if `false` the list must not be used to name differing files |
| `location_in_key` | which source-location fields this Triton hashes: `["line"]` (3.8.0/3.9.0) or `["file", "line", "def_line", "def_col"]` (`main`) |
| `location_in_key_detected_by` | how that was determined |
| `env.cache_invalidating` | what `get_cache_invalidating_env_vars()` returned |
| `env.triton` | all `TRITON_*` variables and `PTXAS_OPTIONS` |
| `cache_dir`, `argv`, `cwd`, `hostname`, `python`, `implementation`, `platform`, `machine` | context |
| `record_source` | whether kernel source text is included |
| `triton_version_verified` | whether this Triton version is one the recorder was tested with |

## `compile` - when Triton builds a key (before it looks in the cache)

| Field | Meaning |
| --- | --- |
| `kernel`, `name` | qualified name (`module.qualname`) and the name used for file names |
| `hash` | SHA-256 of the key string = the `hash` in the entry's metadata |
| `entry` | the cache directory name (base32 of `hash`) |
| `cache_dir` | `knobs.cache.dir` at that moment |
| `components.triton` | `version`, `key_digest` |
| `components.source` | `kind` (`ast`/`ir`), `hash` (`src.hash()`), `signature`, `constants` (`path`, `name`, `value`, `key`), `attrs`, and `fn` |
| `components.source.fn` | `name`, `cache_key`, `file`, `line`, `def_line`, `def_col`, `noinline`, `src_sha256`, `src` (optional), `constexpr_globals`, and `deps`: the same structure for every `@triton.jit` function it references. `deps: null` means the function was hashed before the recorder was installed |
| `components.backend` | `hash` (`backend.hash()`), `class`, `target` |
| `components.options` | `hash`, `values` (all option fields), `extern_libs` (`path`, `sha256`) |
| `components.env` | the environment variables passed into the key |
| `custom_pipeline` | a stage-inspection hook is installed (it appends to the key) |
| `via_jit`, `warmup` | compiled through `JITFunction` (vs a direct `triton.compile`), and whether in warm-up mode |
| `specialization_data` | the JSON string `JITFunction.preload()` accepts; used to replay the compilation |

## `outcome` - from Triton's compilation listener

| Field | Meaning |
| --- | --- |
| `hash` | joins the outcome to its `compile` record (same `pid`) |
| `cache_hit` | as reported by Triton |
| `times_us` | `total`, `ir_initialization`, `lowering` (per stage), `store_results`, in microseconds |
| `artifacts` | per file extension: `size`, `sha256`; for `source`/`ttir` also `sha256_no_loc`, the digest with MLIR debug locations stripped (used to tell real collisions from path-only differences) |

A `compile` record without an `outcome` means the compilation raised, or another tool replaced
the listener.

## `warning`, `note`

Free-text records. A `warning` saying that a compilation hash matched no recorded key means the
installed Triton derives keys differently from what the recorder assumes; do not trust that
recording.

## Privacy

Recordings contain absolute paths, the host name, the command line, `TRITON_*` environment
variables and, unless `--no-source` / `TRITON_CACHEKIT_RECORD_SOURCE=0` is used, kernel source
text. They never contain tensor data or compiled binaries.
