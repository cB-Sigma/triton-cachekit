from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from triton_cachekit import pack as pk
from triton_cachekit.layout import scan_cache

from .conftest import KEY_A, KEY_B, make_entry


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.name.startswith("__grp__")
    }


@pytest.mark.parametrize("name", ["cache.tar", "cache.tar.gz", "cache-oci"])
def test_round_trip_and_relocation(synthetic_cache: Path, tmp_path: Path, name: str) -> None:
    archive = tmp_path / name
    manifest = pk.pack(synthetic_cache, archive)
    assert manifest["entry_count"] == 2 and manifest["triton_versions"] == ["3.9.0"]
    assert manifest["targets"] == ["cuda:80", "hip:gfx942"]
    assert manifest["relocation"]["group_files_absolute"] is True
    assert pk.read_manifest(archive)["entries"][0]["files"]

    dest = tmp_path / "elsewhere" / "cache"
    result = pk.unpack(archive, dest)
    assert tree(dest) == tree(synthetic_cache)
    assert result["relocated"] == [KEY_A]  # only the absolute-path entry needs rewriting
    assert {e.status for e in scan_cache(dest).entries} == {"ok"}
    group = json.loads((dest / KEY_A / "__grp__axpy.json").read_text())
    assert group["child_paths"]["axpy.cubin"] == str(dest.resolve() / KEY_A / "axpy.cubin")
    # relative-encoding group files are left byte-identical
    assert (dest / KEY_B / "__grp__axpy.json").read_bytes() == (
        synthetic_cache / KEY_B / "__grp__axpy.json"
    ).read_bytes()


def test_without_relocation_the_copy_is_stale(synthetic_cache: Path, tmp_path: Path) -> None:
    archive = tmp_path / "c.tar"
    pk.pack(synthetic_cache, archive)
    import shutil

    dest = tmp_path / "dest"
    pk.unpack(archive, dest, relocate=False)
    shutil.rmtree(synthetic_cache)
    statuses = {e.key: e.status for e in scan_cache(dest).entries}
    assert statuses == {KEY_A: "stale-paths", KEY_B: "ok"}
    assert pk.relocate_cache(dest, dry_run=True) == [KEY_A]
    assert pk.relocate_cache(dest) == [KEY_A]
    assert {e.status for e in scan_cache(dest).entries} == {"ok"}
    assert pk.relocate_cache(dest) == []


def test_pack_is_deterministic(synthetic_cache: Path, tmp_path: Path) -> None:
    def members(path: Path) -> list[tuple[str, int]]:
        with tarfile.open(path) as tar:
            return [(m.name, m.size) for m in tar.getmembers()]

    pk.pack(synthetic_cache, tmp_path / "1.tar")
    pk.pack(synthetic_cache, tmp_path / "2.tar")
    assert members(tmp_path / "1.tar") == members(tmp_path / "2.tar")
    assert members(tmp_path / "1.tar")[0][0] == pk.MANIFEST_NAME


def test_incomplete_entries_are_skipped(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    make_entry(cache)
    make_entry(cache, KEY_B, "0" * 64, "broken", group="none")
    manifest = pk.pack(cache, tmp_path / "c.tar")
    assert [e["key"] for e in manifest["entries"]] == [KEY_A]
    manifest = pk.pack(cache, tmp_path / "all.tar", include_incomplete=True)
    assert manifest["entry_count"] == 2
    with pytest.raises(pk.PackError):
        pk.pack(tmp_path / "cache" / KEY_B, tmp_path / "none.tar")


def test_oci_layout_is_well_formed(synthetic_cache: Path, tmp_path: Path) -> None:
    layout = tmp_path / "oci"
    pk.pack(synthetic_cache, layout)
    assert json.loads((layout / "oci-layout").read_text()) == {"imageLayoutVersion": "1.0.0"}
    index = json.loads((layout / "index.json").read_text())
    descriptor = index["manifests"][0]
    blob = (layout / "blobs" / "sha256" / descriptor["digest"].split(":")[1]).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == descriptor["digest"].split(":")[1]
    assert len(blob) == descriptor["size"]
    image = json.loads(blob)
    assert image["artifactType"] == pk.OCI_ARTIFACT_TYPE
    layer = (layout / "blobs" / "sha256" / image["layers"][0]["digest"].split(":")[1]).read_bytes()
    config = json.loads(
        (layout / "blobs" / "sha256" / image["config"]["digest"].split(":")[1]).read_bytes()
    )
    assert config["rootfs"]["diff_ids"] == [
        "sha256:" + hashlib.sha256(gzip.decompress(layer)).hexdigest()
    ]
    with pytest.raises(pk.PackError, match="not empty"):
        pk.pack(synthetic_cache, layout)


def _rewrite(archive: Path, mutate: object) -> None:
    out = io.BytesIO()
    with tarfile.open(archive) as src, tarfile.open(fileobj=out, mode="w") as dst:
        for member in src.getmembers():
            handle = src.extractfile(member)
            assert handle is not None
            data = handle.read()
            member, data = mutate(member, data)  # type: ignore[operator]
            member.size = len(data)
            dst.addfile(member, io.BytesIO(data))
    archive.write_bytes(out.getvalue())


def test_unpack_rejects_tampering(synthetic_cache: Path, tmp_path: Path) -> None:
    archive = tmp_path / "c.tar"
    pk.pack(synthetic_cache, archive)

    def corrupt(member: tarfile.TarInfo, data: bytes) -> tuple[tarfile.TarInfo, bytes]:
        return member, data + b"x" if member.name.endswith(".cubin") else data

    _rewrite(archive, corrupt)
    with pytest.raises(pk.PackError, match="checksum mismatch"):
        pk.unpack(archive, tmp_path / "d1")

    pk.pack(synthetic_cache, archive)

    def escape(member: tarfile.TarInfo, data: bytes) -> tuple[tarfile.TarInfo, bytes]:
        if member.name.endswith(".cubin"):
            member.name = "cache/../../evil"
        return member, data

    _rewrite(archive, escape)
    with pytest.raises(pk.PackError, match="refusing"):
        pk.unpack(archive, tmp_path / "d2")
    assert not (tmp_path / "evil").exists()

    plain = tmp_path / "plain.tar"
    with tarfile.open(plain, "w") as tar:
        info = tarfile.TarInfo("hello")
        tar.addfile(info, io.BytesIO(b""))
    with pytest.raises(pk.PackError, match="not a triton-cachekit archive"):
        pk.unpack(plain, tmp_path / "d3")


def test_unpack_does_not_overwrite_by_default(synthetic_cache: Path, tmp_path: Path) -> None:
    archive = tmp_path / "c.tar"
    pk.pack(synthetic_cache, archive)
    dest = tmp_path / "dest"
    pk.unpack(archive, dest)
    (dest / KEY_A / "axpy.ptx").write_text("local edit")
    result = pk.unpack(archive, dest)
    assert result["written"] == [] and (dest / KEY_A / "axpy.ptx").read_text() == "local edit"
    pk.unpack(archive, dest, overwrite=True)
    assert (dest / KEY_A / "axpy.ptx").read_text() != "local edit"
