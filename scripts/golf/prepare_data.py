#!/usr/bin/env python3
"""Prepare authored CPU text or disjoint FineWeb development/confirmation splits.

This command does not download data. Obtain the pinned official export with
vendor/parameter_golf/data/cached_challenge_fineweb.py before --mode fineweb.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from golfbench.training import fingerprint, prepare_fixture, sha256_file


def read_shard(path: Path):
    import numpy as np
    header = np.fromfile(path, dtype="<i4", count=256)
    if len(header) != 256 or int(header[0]) != 20240520 or int(header[1]) != 1 or int(header[2]) <= 0:
        raise ValueError(f"Invalid official shard header: {path}")
    count = int(header[2])
    if path.stat().st_size != 1024 + 2 * count:
        raise ValueError(f"Shard length does not match header: {path}")
    return np.fromfile(path, dtype="<u2", count=count, offset=1024)


def write_shard(path: Path, tokens) -> None:
    import numpy as np
    if len(tokens) <= 0 or len(tokens) >= 2**31:
        raise ValueError("Invalid token count")
    header = np.zeros(256, dtype="<i4")
    header[:3] = [20240520, 1, len(tokens)]
    with path.open("wb") as out:
        out.write(header.tobytes())
        out.write(tokens.astype("<u2", copy=False).tobytes())


def prepare_fineweb(source_dir: str | Path, tokenizer_path: str | Path,
                    output_dir: str | Path, confirm_fraction: float = 0.2) -> dict:
    import numpy as np
    import sentencepiece as spm
    source, tokenizer, out = Path(source_dir).resolve(), Path(tokenizer_path).resolve(), Path(output_dir).resolve()
    if not 0 < confirm_fraction < 1:
        raise ValueError("confirm_fraction must lie strictly between zero and one")
    if (out / "manifest.json").exists():
        raise FileExistsError("Prepared data manifest already exists; use a new output directory")
    train_files = sorted(source.glob("fineweb_train_*.bin"))
    val_files = sorted(source.glob("fineweb_val_*.bin"))
    if not train_files or not val_files:
        raise ValueError("Need both official fineweb_train_*.bin and fineweb_val_*.bin files")
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
    if sp.bos_id() < 0:
        raise ValueError("Tokenizer must define BOS to preserve document boundaries")
    tokens = np.concatenate([read_shard(path) for path in val_files])
    if tokens.max() >= sp.vocab_size():
        raise ValueError("Validation tokens exceed tokenizer vocabulary")
    starts = np.flatnonzero(tokens == sp.bos_id())
    if len(starts) < 4 or starts[0] != 0:
        raise ValueError("Validation export must begin with BOS and contain at least four documents")
    boundary_document = int(len(starts) * (1 - confirm_fraction))
    boundary = int(starts[boundary_document])
    portions = {"search": tokens[:boundary], "confirm": tokens[boundary:]}
    train_entries = []
    for path in train_files:
        header = np.fromfile(path, dtype="<i4", count=256)
        if len(header) != 256 or int(header[0]) != 20240520 or int(header[1]) != 1 or int(header[2]) <= 0 or path.stat().st_size != 1024 + 2 * int(header[2]):
            raise ValueError(f"Invalid training shard: {path}")
        train_entries.append({"name": path.name, "path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "tokens": int(header[2])})
    if set(entry["sha256"] for entry in train_entries) & {sha256_file(path) for path in val_files}:
        raise ValueError("A training shard is byte-identical to a validation shard")
    manifest = {"schema": "golfbench.data.v1", "source": "parameter_golf_fineweb_partition", "documents_disjoint": True,
                "split_rule": "BOS-aligned contiguous validation document partition; no validation tokens enter training",
                "split_isolation_basis": "Official train/validation export is trusted; search/confirm document index ranges are verified disjoint. Corpus-wide textual deduplication is not claimed.",
                "partition_scope": "development and confirmation subsets; scores are not canonical full-validation leaderboard scores",
                "tokenizer_path": str(tokenizer), "tokenizer_sha256": sha256_file(tokenizer), "vocab_size": sp.vocab_size(), "bos_id": sp.bos_id(),
                "source_validation_files": [{"name": p.name, "sha256": sha256_file(p)} for p in val_files],
                "source_validation_documents": len(starts), "confirmation_fraction_requested": confirm_fraction,
                "splits": {"train": {"files": train_entries, "sha256": fingerprint(train_entries), "tokens": sum(e["tokens"] for e in train_entries)}}}
    for split, part in portions.items():
        directory = out / split
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "fineweb_val_000000.bin"
        write_shard(path, part)
        for train_path in train_files:
            link = directory / train_path.name
            if link.exists() or link.is_symlink():
                raise FileExistsError(f"Refusing to replace existing training data link: {link}")
            link.symlink_to(train_path)
        manifest["splits"][split] = {"file": str(path.relative_to(out)), "sha256": sha256_file(path), "tokens": len(part),
                                      "document_index_start": 0 if split == "search" else boundary_document,
                                      "document_index_end_exclusive": boundary_document if split == "search" else len(starts)}
    manifest["dataset_fingerprint"] = fingerprint(manifest)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fixture", "fineweb"], default="fixture")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-dir", help="Existing official token shard directory")
    parser.add_argument("--tokenizer-path", help="Matching official SentencePiece .model")
    parser.add_argument("--confirm-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if args.mode == "fixture":
        manifest = prepare_fixture(args.output_dir)
    else:
        if not args.source_dir or not args.tokenizer_path:
            parser.error("fineweb mode requires --source-dir and --tokenizer-path")
        manifest = prepare_fineweb(args.source_dir, args.tokenizer_path, args.output_dir, args.confirm_fraction)
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()), "source": manifest["source"], "dataset_fingerprint": manifest["dataset_fingerprint"]}, indent=2))


if __name__ == "__main__":
    main()
