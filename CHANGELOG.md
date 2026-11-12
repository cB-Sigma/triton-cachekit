# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-12

First public version. Developed and tested against the Triton 3.8.0 and 3.9.0 wheels on a
CPU-only Linux machine; nothing has been run on a real GPU yet.

### Added

- `inspect`: Triton-free parser for the on-disk cache (absolute and relative group-file
  encodings), table and JSON output, totals by kernel / target / Triton version, detection of
  incomplete, corrupt and mis-relocated entries.
- `record`: recorder that hooks Triton's own key computation and logs each key component,
  callee hashes, specialisation data, hit/miss and stage timings to JSONL. Refuses to install
  (`UnsupportedTritonError`) when Triton's internals do not match.
- `explain`: component-level comparison of two recordings with plain-language remedies, and
  `--collisions` to find different kernels (or different generated IR) sharing one key.
- `stats`: compile time by kernel and stage and the share a warm cache would remove.
- `prewarm`: GPU-less cache population through a stand-in driver, from a TOML manifest or from a
  recording, in isolated subprocesses with timeouts; `fingerprint` and `--server-fingerprint`
  pre-checks.
- `verify-hit`: per-kernel hit/miss report for a prewarmed cache, to be run on the GPU host.
- `pack` / `unpack` / `relocate`: tar, tar.gz and OCI image layout archives with manifest and
  checksums; rewriting of absolute group-file paths.
- `diff` and `prune`.
- `--demo` for `inspect`, `explain`, `stats` and `prewarm`.

### Known limitations

- A prewarmed cache being hit on a real GPU host is unverified.
- Not run against a Triton `main` build (source-path key component, relative group files).
- Linux only.

[Unreleased]: https://github.com/cb-Sigma/triton-cachekit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cb-Sigma/triton-cachekit/releases/tag/v0.1.0
