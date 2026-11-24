#!/usr/bin/env python3
"""Losslessly package replay trajectories; keep summaries and locks readable."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

NAME = re.compile(r"[a-z0-9-]+_seed[0-9]+\.json\Z")


def load_records(directory: Path) -> dict:
    """Read loose records or their SHA-256-verified archive without extraction."""
    loose = sorted(directory.glob("*_seed*.json"))
    archive = directory / "trajectories.zip"
    index_path = directory / "trajectory_index.json"
    if archive.exists():
        index = json.loads(index_path.read_text())
        if any(path.name not in index['files'] for path in loose):
            raise ValueError("Loose replay records exist that are not in the archive index")
        if hashlib.sha256(archive.read_bytes()).hexdigest() != index['archive_sha256']:
            raise ValueError(f"Replay archive identity changed: {archive}")
        with zipfile.ZipFile(archive) as z:
            if set(z.namelist()) != set(index['files']):
                raise ValueError("Replay archive members disagree with the index")
            result = {}
            for name, item in index['files'].items():
                if not NAME.fullmatch(name):
                    raise ValueError("Unexpected archive member name")
                data = z.read(name)
                if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                    raise ValueError(f"Replay trajectory identity changed: {name}")
                local = directory / name
                if local.exists() and local.read_bytes() != data:
                    raise ValueError(f"Loose replay differs from its immutable archive: {local}")
                result[name] = json.loads(data)
            return result
    return {path.name: json.loads(path.read_text()) for path in loose}


def pack(directory: Path):
    files = sorted(directory.glob("*_seed*.json"))
    if not files:
        raise ValueError(f"No loose trajectories to archive in {directory}")
    archive = directory / "trajectories.zip"
    temporary = directory / "trajectories.zip.tmp"
    index = {"schema_version": 1, "file_count": len(files), "files": {}}
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in files:
            if not NAME.fullmatch(path.name):
                raise ValueError(f"Unexpected trajectory filename: {path.name}")
            data = path.read_bytes()
            info = zipfile.ZipInfo(path.name, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            index['files'][path.name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    index['archive_sha256'] = hashlib.sha256(temporary.read_bytes()).hexdigest()
    index['uncompressed_bytes'] = sum(item['bytes'] for item in index['files'].values())
    index['archive_bytes'] = temporary.stat().st_size
    if archive.exists() and archive.read_bytes() != temporary.read_bytes():
        temporary.unlink()
        raise ValueError("Refusing to replace a different replay archive")
    temporary.replace(archive)
    (directory / "trajectory_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    assert len(load_records(directory)) == len(files)
    print(f"Verified {len(files)} exact trajectories: {archive} ({index['archive_bytes']} bytes)")


def unpack(directory: Path):
    records = load_records(directory)
    with zipfile.ZipFile(directory / "trajectories.zip") as z:
        for name in records:
            path = directory / name
            if not path.exists():
                path.write_bytes(z.read(name))
    print(f"Verified/restored {len(records)} trajectories in {directory}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['pack', 'unpack', 'verify'])
    p.add_argument('directories', nargs='+', type=Path)
    a = p.parse_args()
    for directory in a.directories:
        if a.action == 'pack':
            pack(directory)
        elif a.action == 'unpack':
            unpack(directory)
        else:
            print(f"Verified {len(load_records(directory))} trajectories: {directory}")
