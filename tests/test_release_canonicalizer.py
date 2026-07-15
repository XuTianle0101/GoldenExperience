from __future__ import annotations

import gzip
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from paper.tools.canonicalize_release import canonicalize_sdist, canonicalize_wheel

EPOCH = 1_784_073_600


def _write_sdist(path: Path, *, reverse: bool, mtime: int) -> None:
    entries = [("package-1.0/README.md", b"release\n"), ("package-1.0/data.json", b"{}\n")]
    if reverse:
        entries.reverse()
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="input.tar", mode="wb", fileobj=raw, mtime=mtime) as zipped,
        tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        directory = tarfile.TarInfo("package-1.0")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        directory.mtime = mtime
        archive.addfile(directory)
        for name, payload in entries:
            member = tarfile.TarInfo(name)
            member.mode = 0o644
            member.mtime = mtime
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def _write_wheel(path: Path, *, reverse: bool, year: int) -> None:
    entries = [("package/__init__.py", b""), ("package-1.0.dist-info/METADATA", b"Name: p\n")]
    if reverse:
        entries.reverse()
    with zipfile.ZipFile(path, mode="w") as archive:
        for name, payload in entries:
            entry = zipfile.ZipInfo(name, date_time=(year, 1, 1, 0, 0, 0))
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, payload, compress_type=zipfile.ZIP_DEFLATED)


def test_canonical_sdist_ignores_order_and_container_timestamps(tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_sdist(first, reverse=False, mtime=1_700_000_000)
    _write_sdist(second, reverse=True, mtime=1_750_000_000)

    first_output = tmp_path / "first-canonical.tar.gz"
    second_output = tmp_path / "second-canonical.tar.gz"
    canonicalize_sdist(first, first_output, epoch=EPOCH)
    canonicalize_sdist(second, second_output, epoch=EPOCH)

    assert first_output.read_bytes() == second_output.read_bytes()
    with tarfile.open(first_output, mode="r:gz") as archive:
        assert all(member.mtime == EPOCH for member in archive.getmembers())


def test_canonical_wheel_ignores_order_and_container_timestamps(tmp_path: Path) -> None:
    first = tmp_path / "first.whl"
    second = tmp_path / "second.whl"
    _write_wheel(first, reverse=False, year=2024)
    _write_wheel(second, reverse=True, year=2025)

    first_output = tmp_path / "first-canonical.whl"
    second_output = tmp_path / "second-canonical.whl"
    canonicalize_wheel(first, first_output, epoch=EPOCH)
    canonicalize_wheel(second, second_output, epoch=EPOCH)

    assert first_output.read_bytes() == second_output.read_bytes()
    with zipfile.ZipFile(first_output) as archive:
        assert archive.namelist() == sorted(archive.namelist())


def test_canonical_sdist_rejects_parent_traversal(tmp_path: Path) -> None:
    source = tmp_path / "unsafe.tar.gz"
    with tarfile.open(source, mode="w:gz") as archive:
        payload = b"escape"
        member = tarfile.TarInfo("../escape")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    destination = tmp_path / "canonical.tar.gz"
    with pytest.raises(ValueError, match="escapes"):
        canonicalize_sdist(source, destination, epoch=EPOCH)
    assert not destination.exists()
