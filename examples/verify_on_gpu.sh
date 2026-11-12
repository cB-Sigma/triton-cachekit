#!/usr/bin/env bash
# Run this ON THE GPU HOST to test whether a prewarmed cache is really hit.
# Nobody has run this against real hardware for this project yet: if you do,
# please report the outcome (hit or miss, GPU, Triton version) in an issue.
#
# Usage: verify_on_gpu.sh <cache.tar.gz> <prewarm-manifest.json> [prewarm.rec dir]
set -euo pipefail

archive="$1"
manifest="$2"
recording="${3:-}"
cache="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"

triton-cachekit unpack "$archive" "$cache"
triton-cachekit inspect "$cache" --strict

args=(--manifest "$manifest" --cache-dir "$cache" --recording-out ./verify.rec)
if [[ -n "$recording" ]]; then
    args+=(--prewarm-recording "$recording")
fi
# Exit status: 0 = every specialisation hit, 1 = at least one miss (with reasons), 2 = could not run.
triton-cachekit verify-hit "${args[@]}"
