"""Strict JSON, content identities and atomic writes shared by experiment tools."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path: str | Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        if overwrite:
            os.replace(temp, path)
        else:
            os.link(temp, path)  # no-clobber even if another process finishes first
            temp.unlink()
    finally:
        if temp.exists():
            temp.unlink()
