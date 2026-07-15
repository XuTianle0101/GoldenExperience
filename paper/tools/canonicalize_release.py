#!/usr/bin/env python3
"""Canonicalize wheel and sdist container metadata for reproducible releases."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

SCHEMA_VERSION = "goldenexperience.canonical_release.v1"
ZIP_MIN_EPOCH = 315532800  # 1980-01-01 UTC


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_archive_name(name: str) -> None:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"archive member escapes the distribution root: {name!r}")


def _temporary_output(destination: Path) -> tuple[BinaryIO, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    return os.fdopen(descriptor, "w+b"), Path(name)


def canonicalize_sdist(source: Path, destination: Path, *, epoch: int) -> None:
    """Repack an sdist with sorted members and fixed tar/gzip metadata."""

    output, temporary = _temporary_output(destination)
    try:
        with output:
            with tarfile.open(source, mode="r:gz") as input_tar:
                members = input_tar.getmembers()
                names = [member.name for member in members]
                for name in names:
                    _validate_archive_name(name)
                if len(names) != len(set(names)):
                    raise ValueError("sdist contains duplicate member names")

                with (
                    gzip.GzipFile(
                        filename="",
                        mode="wb",
                        compresslevel=9,
                        fileobj=output,
                        mtime=epoch,
                    ) as compressed,
                    tarfile.open(
                        fileobj=compressed,
                        mode="w",
                        format=tarfile.PAX_FORMAT,
                    ) as output_tar,
                ):
                    for member in sorted(members, key=lambda item: item.name):
                        normalized = copy.copy(member)
                        normalized.uid = 0
                        normalized.gid = 0
                        normalized.uname = ""
                        normalized.gname = ""
                        normalized.mtime = epoch
                        normalized.pax_headers = {}
                        payload = input_tar.extractfile(member) if member.isfile() else None
                        output_tar.addfile(normalized, payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def canonicalize_wheel(source: Path, destination: Path, *, epoch: int) -> None:
    """Repack a wheel with sorted entries and fixed ZIP metadata."""

    zip_time = time.gmtime(epoch)[:6]
    output, temporary = _temporary_output(destination)
    try:
        with output:
            with zipfile.ZipFile(source, mode="r") as input_zip:
                entries = input_zip.infolist()
                names = [entry.filename for entry in entries]
                for name in names:
                    _validate_archive_name(name)
                if len(names) != len(set(names)):
                    raise ValueError("wheel contains duplicate member names")

                with zipfile.ZipFile(output, mode="w") as output_zip:
                    for entry in sorted(entries, key=lambda item: item.filename):
                        normalized = zipfile.ZipInfo(entry.filename, date_time=zip_time)
                        normalized.compress_type = entry.compress_type
                        normalized.create_system = entry.create_system
                        normalized.external_attr = entry.external_attr
                        normalized.internal_attr = entry.internal_attr
                        normalized.flag_bits = entry.flag_bits
                        normalized.comment = entry.comment
                        output_zip.writestr(
                            normalized,
                            input_zip.read(entry.filename),
                            compress_type=entry.compress_type,
                            compresslevel=9,
                        )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def canonicalize_release(input_dir: Path, output_dir: Path, *, epoch: int) -> dict[str, object]:
    if epoch < ZIP_MIN_EPOCH:
        raise ValueError("source date epoch predates the ZIP timestamp range")
    wheels = sorted(input_dir.glob("*.whl"))
    sdists = sorted(input_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("input directory must contain exactly one wheel and one .tar.gz sdist")

    artifacts: list[dict[str, object]] = []
    for source, kind, canonicalizer in (
        (wheels[0], "wheel", canonicalize_wheel),
        (sdists[0], "sdist", canonicalize_sdist),
    ):
        destination = output_dir / source.name
        canonicalizer(source, destination, epoch=epoch)
        artifacts.append(
            {
                "filename": destination.name,
                "input_sha256": _sha256_file(source),
                "kind": kind,
                "sha256": _sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
    return {
        "artifacts": artifacts,
        "schema_version": SCHEMA_VERSION,
        "source_date_epoch": epoch,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-date-epoch",
        type=int,
        default=int(os.environ.get("SOURCE_DATE_EPOCH", "0")),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = canonicalize_release(
        args.input_dir.resolve(),
        args.output_dir.resolve(),
        epoch=args.source_date_epoch,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
