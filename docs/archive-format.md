# Archive format (`pack` / `unpack`)

## Contents

Whether written as `.tar`, `.tar.gz` or as the single layer of an OCI image layout, the payload is
the same tar stream:

```text
triton-cachekit.manifest.json
cache/<KEY>/<file>
cache/<KEY>/<file>
...
```

Files are stored byte for byte, including group files with their original absolute paths. Members
are sorted, owned by uid/gid 0 with mode 0644, and the gzip header carries no timestamp.

Only complete kernel entries are packed by default (`--include-incomplete` to override). Autotune
results and launcher stubs are skipped: the former need a GPU to be meaningful, the latter are
host-specific shared objects.

## Manifest

```jsonc
{
  "schema": 1,
  "tool_version": "0.1.0",
  "created": "2026-11-12T20:00:00Z",
  "source_cache_dir": "/build/cache",
  "triton_versions": ["3.9.0"],
  "targets": ["cuda:80"],
  "entry_count": 3, "file_count": 24, "total_size": 196301,
  "relocation": { "group_files_absolute": true, "note": "..." },
  "portability": "...",
  "packer_fingerprint": { ... },      // `fingerprint` of the packing host, or null without Triton
  "entries": [
    {
      "key": "W6BWNRTVCN...", "kind": "kernel", "kernel": "rms_norm_rows",
      "target": "cuda:80", "triton_version": "3.9.0", "env": {},
      "group_format": "absolute",
      "source_file": "/build/app/kernels.py", "source_line": 17,
      "files": [ { "path": "W6BWNRTVCN.../rms_norm_rows.cubin", "size": 9944, "sha256": "..." } ]
    }
  ]
}
```

## Unpacking

`unpack ARCHIVE DIR`:

1. reads the manifest and refuses unknown schema versions;
2. accepts only regular-file members named `cache/<key>/<file>` that are listed in the manifest
   (no absolute paths, no `..`, no links), and verifies each SHA-256;
3. writes files atomically; existing files are kept unless `--overwrite`;
4. rewrites `__grp__*.json` files that use absolute paths so they point into `DIR`
   (`--no-relocate` to skip);
5. if Triton is installed, warns when its version or `triton_key()` digest differs from the
   archive's, because such entries can never be hit there.

The checksums protect against corruption in transit. They are **not** a signature: the manifest is
part of the archive. Sign archives with external tooling if they cross a trust boundary.

## OCI image layout

`pack --format oci -o DIR` (or any output path that is not `*.tar`/`*.tar.gz`) writes:

```text
DIR/oci-layout                 {"imageLayoutVersion": "1.0.0"}
DIR/index.json                 one manifest descriptor, tagged "latest"
DIR/blobs/sha256/<digest>      image manifest, config, and one tar+gzip layer
```

- manifest `artifactType`: `application/vnd.triton-cachekit.cache.v1`
- layer media type: `application/vnd.oci.image.layer.v1.tar+gzip`
- config media type: `application/vnd.oci.image.config.v1+json` (with the layer's `diff_id`), so
  the result is also a valid single-layer image whose root contains `cache/` and the manifest.

The layout can be copied to a registry with standard tools, for example
`skopeo copy oci:DIR:latest docker://registry.example/cache:tag` or
`oras cp --from-oci-layout DIR:latest registry.example/cache:tag`. The test suite validates the
layout's structure and digests; pushing to and pulling from a registry has not been tested.

`unpack` reads the same layout directory back.

If the cache is extracted by something other than `unpack` (a container runtime mounting the
image, plain `tar`), run `triton-cachekit relocate <cache dir>` at the final location.
