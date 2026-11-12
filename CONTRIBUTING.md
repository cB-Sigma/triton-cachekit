# Contributing

Thanks for looking. The most valuable contributions right now, in order:

1. **`verify-hit` results from real GPUs.** The central open question of this project is whether a
   cache built by `prewarm` on a CPU machine is hit on a GPU host. Run
   `examples/verify_on_gpu.sh` and open a "verify-hit result" issue with the output, hit or miss.
2. **Cache directories `inspect` gets wrong**, especially from Triton versions other than 3.8.0
   and 3.9.0. The metadata JSON and a directory listing of one entry are enough; no binaries.
3. **Recordings where `explain` says "unknown"** or names the wrong component.
4. Bug fixes and documentation.

## Development setup

```console
$ git clone https://github.com/cb-Sigma/triton-cachekit && cd triton-cachekit
$ python -m venv .venv && . .venv/bin/activate
$ pip install -e '.[dev]'
$ pip install triton==3.9.0        # optional: enables the integration tests (no GPU needed)
```

## Checks

All four must pass; CI runs the same commands.

```console
$ ruff check .
$ ruff format --check .
$ mypy
$ pytest
```

- Without Triton installed, the tests marked `triton` are skipped and the rest runs in a few
  seconds. With Triton, the whole suite takes about two minutes: the integration tests compile
  real kernels through the fake driver, each in a fresh subprocess.
- `pytest -m "not triton"` runs only the fast tests.
- `python scripts/update_readme.py` regenerates the output blocks in `README.md` (needs Triton).
  `tests/test_readme.py` fails if they drift from what the tool prints.

## Ground rules for changes

- **Only claim what is verified.** Every explanation `explain` prints must correspond to something
  you checked in Triton's source for a stated version; add the reference to `docs/cache-key.md`.
  If something was not run on a real GPU, the code and the docs must say so.
- **Do not re-implement Triton's key.** The recorder observes Triton's own functions. If a new
  Triton version changes them, extend the checks in `Recorder._check_internals` so that older
  assumptions fail loudly, then add support.
- **`inspect`, `explain`, `stats`, `pack`, `unpack`, `relocate`, `diff` and `prune` must not
  import Triton** and must keep working on a machine without it.
- No runtime dependencies: the package is standard library only.
- Test fixtures are generated in the tests. Do not commit cache entries or other binaries.
- Example kernels must be your own work, not copied from Triton or other projects.

## Supporting a new Triton version

1. Install it in a fresh environment and run the full suite.
2. If `UnsupportedTritonError` is raised, read the listed differences and the new Triton source.
3. Update the hooks, `VERIFIED_TRITON_VERSIONS` in `recorder.py`, `docs/cache-key.md`, the
   version table in the README and the CI matrix.

## Commit and pull request style

Small, focused pull requests with a description of what was verified and how. Add a line to
`CHANGELOG.md` under "Unreleased" for user-visible changes.

By contributing you agree that your contributions are licensed under the Apache License 2.0.
