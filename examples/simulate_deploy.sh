#!/usr/bin/env bash
# Walk through the whole flow on one machine without a GPU:
#   prewarm -> inspect -> pack -> unpack somewhere else -> verify-hit (simulated)
#
# The last step uses --fake-driver, so it only shows that the tooling fits
# together. On a real deployment run verify-hit on the GPU host WITHOUT that flag.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

triton-cachekit prewarm --manifest "$here/prewarm.toml" --cache-dir "$work/build/cache" \
    -o "$work/build/prewarm-manifest.json"
triton-cachekit inspect "$work/build/cache"
triton-cachekit pack "$work/build/cache" -o "$work/cache.tar.gz"

# --- "on the server" -------------------------------------------------------
triton-cachekit unpack "$work/cache.tar.gz" "$work/server/cache"
mv "$work/build/cache" "$work/build/cache.gone"   # the build path does not exist on a server
triton-cachekit verify-hit --manifest "$work/build/prewarm-manifest.json" \
    --cache-dir "$work/server/cache" --env TRITON_DEFAULT_FP_FUSION=0 --fake-driver
