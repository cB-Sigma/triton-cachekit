"""Pack a cache into a tar archive or an OCI image layout, and unpack it again.

Archive layout (identical inside a tar file and inside the OCI layer)::

    triton-cachekit.manifest.json    what is inside, with a sha256 per file
    cache/<KEY>/<file>               the cache entries, byte for byte

Relocation. A cache entry is only position-independent if its group file is.
Triton 3.8.0/3.9.0 write absolute paths into ``__grp__*.json``; ``unpack`` and
``relocate`` rewrite them for the destination directory. Nothing else in an
entry depends on where the cache directory lives. Whether an entry can be *hit*
on another machine is a separate question (same Triton build, environment and
source location) that packing cannot change; the manifest records what it can
so that ``unpack`` can warn.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import time
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from . import __version__
from ._util import sha256_file
from .layout import GROUP_PREFIX, CacheEntry, scan_cache

MANIFEST_NAME = "triton-cachekit.manifest.json"
PACK_SCHEMA = 1
OCI_ARTIFACT_TYPE = "application/vnd.triton-cachekit.cache.v1"
OCI_LAYER_MEDIA_TYPE = "application/vnd.oci.image.layer.v1.tar+gzip"
OCI_CONFIG_MEDIA_TYPE = "application/vnd.oci.image.config.v1+json"
OCI_MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json"


class PackError(RuntimeError):
    pass


def select_entries(
    entries: Iterable[CacheEntry],
    *,
    include_incomplete: bool = False,
    kinds: tuple[str, ...] = ("kernel",),
) -> tuple[list[CacheEntry], list[CacheEntry]]:
    """Split into (packed, skipped). Stale absolute paths are fine: unpack fixes them."""
    packed, skipped = [], []
    for entry in entries:
        usable = entry.status in ("ok", "stale-paths") or include_incomplete
        if entry.kind in kinds and usable:
            packed.append(entry)
        else:
            skipped.append(entry)
    return packed, skipped


def build_manifest(
    root: Path, entries: list[CacheEntry], fingerprint: dict[str, Any] | None
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    versions: set[str] = set()
    targets: set[str] = set()
    for entry in sorted(entries, key=lambda e: (e.key, e.name or "")):
        files: list[dict[str, Any]] = []
        for info in sorted(entry.files, key=lambda f: f.name):
            files.append(
                {
                    "path": f"{entry.key}/{info.name}",
                    "size": info.size,
                    "sha256": sha256_file(entry.path / info.name),
                }
            )
        if entry.triton_version:
            versions.add(entry.triton_version)
        if entry.kind == "kernel":
            targets.add(entry.target)
        items.append(
            {
                "key": entry.key,
                "kind": entry.kind,
                "kernel": entry.name,
                "target": entry.target,
                "triton_version": entry.triton_version,
                "env": entry.env,
                "group_format": entry.group_format,
                "source_file": entry.source_file,
                "source_line": entry.source_line,
                "files": files,
            }
        )
    absolute = any(e.group_format == "absolute" for e in entries)
    return {
        "schema": PACK_SCHEMA,
        "tool_version": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_cache_dir": str(root.resolve()),
        "triton_versions": sorted(versions),
        "targets": sorted(targets),
        "entry_count": len(items),
        "file_count": sum(len(i["files"]) for i in items),
        "total_size": sum(f["size"] for i in items for f in i["files"]),
        "relocation": {
            "group_files_absolute": absolute,
            "note": (
                "Group files store absolute paths (Triton 3.8.0/3.9.0 layout); they are rewritten "
                "for the destination by 'triton-cachekit unpack'. Extracting this archive with "
                "plain tar requires 'triton-cachekit relocate DIR' afterwards."
                if absolute
                else "Group files store relative names; the entries are position-independent."
            ),
        },
        "portability": (
            "Entries are hit only by the Triton build that wrote them (see triton_versions and "
            "packer_fingerprint), with the same cache-invalidating environment variables, the "
            "same kernel source at the same line and - on Triton builds that hash it - the "
            "same source path."
        ),
        "packer_fingerprint": fingerprint,
        "entries": items,
    }


def _tar_bytes(root: Path, manifest: dict[str, Any]) -> bytes:
    """Deterministic uncompressed tar of the manifest and the listed files."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(payload)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(payload))
        for entry in manifest["entries"]:
            for file in entry["files"]:
                source = root / file["path"]
                info = tarfile.TarInfo(f"cache/{file['path']}")
                info.size = file["size"]
                info.mode = 0o644
                info.mtime = int(source.stat().st_mtime)
                with open(source, "rb") as handle:
                    tar.addfile(info, handle)
    return buffer.getvalue()


def _gzip(data: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(data)
    return buffer.getvalue()


def detect_format(path: Path, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    name = path.name
    if name.endswith((".tar.gz", ".tgz")):
        return "tar.gz"
    if name.endswith(".tar"):
        return "tar"
    return "oci"


def pack(
    cache_dir: Path,
    output: Path,
    *,
    fmt: str | None = None,
    entries: list[CacheEntry] | None = None,
    include_incomplete: bool = False,
    fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the archive and return the manifest."""
    scan = scan_cache(cache_dir)
    chosen, _ = select_entries(
        scan.entries if entries is None else entries, include_incomplete=include_incomplete
    )
    if not chosen:
        raise PackError(f"{cache_dir}: no complete kernel entries to pack")
    manifest = build_manifest(scan.root, chosen, fingerprint)
    raw = _tar_bytes(scan.root, manifest)
    kind = detect_format(output, fmt)
    if kind == "tar":
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(raw)
    elif kind == "tar.gz":
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(_gzip(raw))
    elif kind == "oci":
        _write_oci_layout(output, raw, manifest)
    else:
        raise PackError(f"unknown archive format {kind!r} (use tar, tar.gz or oci)")
    return manifest


def _blob(directory: Path, data: bytes) -> tuple[str, int]:
    digest = hashlib.sha256(data).hexdigest()
    target = directory / "blobs" / "sha256"
    target.mkdir(parents=True, exist_ok=True)
    (target / digest).write_bytes(data)
    return f"sha256:{digest}", len(data)


def _write_oci_layout(output: Path, raw_tar: bytes, manifest: dict[str, Any]) -> None:
    """Write an OCI image layout directory (image-spec v1.1) with one gzip layer."""
    if output.exists() and any(output.iterdir()):
        raise PackError(f"{output}: OCI layout directory already exists and is not empty")
    output.mkdir(parents=True, exist_ok=True)
    layer = _gzip(raw_tar)
    layer_digest, layer_size = _blob(output, layer)
    annotations = {
        "org.opencontainers.image.created": manifest["created"],
        "org.opencontainers.image.title": "triton kernel cache",
        "io.github.triton-cachekit.entries": str(manifest["entry_count"]),
        "io.github.triton-cachekit.triton-versions": ",".join(manifest["triton_versions"]),
        "io.github.triton-cachekit.targets": ",".join(manifest["targets"]),
    }
    config = {
        "created": manifest["created"],
        "architecture": "amd64",
        "os": "linux",
        "config": {"Labels": annotations},
        "rootfs": {"type": "layers", "diff_ids": ["sha256:" + hashlib.sha256(raw_tar).hexdigest()]},
        "history": [
            {"created": manifest["created"], "created_by": f"triton-cachekit {__version__} pack"}
        ],
    }
    config_digest, config_size = _blob(output, json.dumps(config, sort_keys=True).encode("utf-8"))
    image_manifest = {
        "schemaVersion": 2,
        "mediaType": OCI_MANIFEST_MEDIA_TYPE,
        "artifactType": OCI_ARTIFACT_TYPE,
        "config": {
            "mediaType": OCI_CONFIG_MEDIA_TYPE,
            "digest": config_digest,
            "size": config_size,
        },
        "layers": [{"mediaType": OCI_LAYER_MEDIA_TYPE, "digest": layer_digest, "size": layer_size}],
        "annotations": annotations,
    }
    manifest_digest, manifest_size = _blob(
        output, json.dumps(image_manifest, sort_keys=True).encode("utf-8")
    )
    index = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "manifests": [
            {
                "mediaType": OCI_MANIFEST_MEDIA_TYPE,
                "artifactType": OCI_ARTIFACT_TYPE,
                "digest": manifest_digest,
                "size": manifest_size,
                "annotations": {"org.opencontainers.image.ref.name": "latest"},
            }
        ],
    }
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (output / "oci-layout").write_text('{"imageLayoutVersion": "1.0.0"}\n', encoding="utf-8")


def _read_oci_layer(path: Path) -> bytes:
    def read_blob(digest: str) -> bytes:
        algorithm, _, value = digest.partition(":")
        if algorithm != "sha256" or not value.isalnum():
            raise PackError(f"{path}: unsupported digest {digest!r}")
        data = (path / "blobs" / "sha256" / value).read_bytes()
        if hashlib.sha256(data).hexdigest() != value:
            raise PackError(f"{path}: blob {digest} is corrupt (digest mismatch)")
        return data

    try:
        index = json.loads((path / "index.json").read_text(encoding="utf-8"))
        descriptor = index["manifests"][0]
        image_manifest = json.loads(read_blob(descriptor["digest"]))
        layers = image_manifest["layers"]
    except (OSError, KeyError, IndexError, ValueError) as exc:
        raise PackError(f"{path}: not a readable OCI image layout ({exc})") from exc
    if len(layers) != 1:
        raise PackError(f"{path}: expected exactly one layer, found {len(layers)}")
    return gzip.decompress(read_blob(layers[0]["digest"]))


def _open_archive(path: Path) -> bytes:
    if path.is_dir():
        return _read_oci_layer(path)
    data = path.read_bytes()
    if data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    return data


def read_manifest(archive: Path) -> dict[str, Any]:
    raw = _open_archive(archive)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
        try:
            member = tar.getmember(MANIFEST_NAME)
        except KeyError as exc:
            raise PackError(
                f"{archive}: no {MANIFEST_NAME} (not a triton-cachekit archive)"
            ) from exc
        handle = tar.extractfile(member)
        assert handle is not None
        manifest = json.loads(handle.read())
    if manifest.get("schema") != PACK_SCHEMA:
        raise PackError(f"{archive}: unsupported manifest schema {manifest.get('schema')!r}")
    return dict(manifest)


def unpack(
    archive: Path,
    dest: Path,
    *,
    relocate: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract into ``dest`` (a cache directory), verifying every checksum.

    Returns ``{"manifest", "written", "skipped", "relocated"}``.
    """
    raw = _open_archive(archive)
    manifest = read_manifest(archive)
    expected = {f["path"]: f for entry in manifest["entries"] for f in entry["files"]}
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    skipped: list[str] = []
    seen: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
        for member in tar:
            if member.name == MANIFEST_NAME:
                continue
            pure = PurePosixPath(member.name)
            if (
                not member.isfile()
                or pure.is_absolute()
                or ".." in pure.parts
                or len(pure.parts) != 3
                or pure.parts[0] != "cache"
            ):
                raise PackError(f"{archive}: refusing unexpected archive member {member.name!r}")
            relative = "/".join(pure.parts[1:])
            if relative not in expected:
                raise PackError(f"{archive}: member {member.name!r} is not listed in the manifest")
            handle = tar.extractfile(member)
            assert handle is not None
            data = handle.read()
            if hashlib.sha256(data).hexdigest() != expected[relative]["sha256"]:
                raise PackError(f"{archive}: checksum mismatch for {relative}")
            seen.add(relative)
            target = dest / pure.parts[1] / pure.parts[2]
            if target.exists() and not overwrite:
                skipped.append(relative)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + ".tck-tmp")
            temp.write_bytes(data)
            os.utime(temp, (member.mtime, member.mtime))
            os.replace(temp, target)
            written.append(relative)
    missing = sorted(set(expected) - seen)
    if missing:
        raise PackError(
            f"{archive}: {len(missing)} file(s) listed in the manifest are missing, e.g. {missing[0]}"
        )
    relocated = (
        relocate_cache(dest, keys={e["key"] for e in manifest["entries"]}) if relocate else []
    )
    return {"manifest": manifest, "written": written, "skipped": skipped, "relocated": relocated}


def relocate_cache(
    cache_dir: Path, *, keys: set[str] | None = None, dry_run: bool = False
) -> list[str]:
    """Rewrite absolute paths in ``__grp__*.json`` so they point into ``cache_dir``.

    Only group files in the absolute-path encoding are touched; the encoding itself
    is preserved (the Triton that wrote the entry is the only one that can read it).
    Returns the keys whose group files changed.
    """
    changed: list[str] = []
    root = cache_dir.resolve()
    for entry_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if keys is not None and entry_dir.name not in keys:
            continue
        for group_file in sorted(entry_dir.glob(f"{GROUP_PREFIX}*.json")):
            try:
                data = json.loads(group_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            children = data.get("child_paths") if isinstance(data, dict) else None
            if not isinstance(children, dict):
                continue
            updated = {name: str(entry_dir / name) for name in children}
            if updated == children:
                continue
            if not all((entry_dir / name).exists() for name in children):
                continue  # incomplete entry: leave it alone, inspect will report it
            if not dry_run:
                stat = group_file.stat()
                temp = group_file.with_name(group_file.name + ".tck-tmp")
                temp.write_text(json.dumps({**data, "child_paths": updated}), encoding="utf-8")
                os.utime(temp, (stat.st_atime, stat.st_mtime))
                os.replace(temp, group_file)
            if entry_dir.name not in changed:
                changed.append(entry_dir.name)
    return changed


def remove_entry_files(entry: CacheEntry) -> None:
    """Delete one logical entry; remove the key directory when it becomes empty."""
    for info in entry.files:
        (entry.path / info.name).unlink(missing_ok=True)
    remaining = [p for p in entry.path.iterdir() if p.name != "lock"] if entry.path.exists() else []
    if not remaining and entry.path.exists():
        shutil.rmtree(entry.path, ignore_errors=True)
