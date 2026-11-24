#!/usr/bin/env python3
"""Verify and extract the official metrics-only NATS-Bench TSS archive."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ARCHIVE_NAME = "NATS-tss-v1_0-3ffb9-simple.tar"
DIRECTORY_NAME = "NATS-tss-v1_0-3ffb9-simple"
SHA256 = "580fd8f3425fed9f495640b0f12ccb7744f89dec7aff263bd75aed37ab0b8bb6"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True)
    p.add_argument("--destination", default="data")
    a = p.parse_args()
    archive = Path(a.archive).resolve()
    destination = Path(a.destination).resolve()
    fingerprint = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            fingerprint.update(chunk)
    if fingerprint.hexdigest() != SHA256:
        raise SystemExit("Archive checksum does not match the verified official release")
    output = destination / DIRECTORY_NAME
    if output.exists():
        raise SystemExit(f"Destination already exists; refusing to overwrite: {output}")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or member.issym() or member.islnk():
                raise SystemExit("Unexpected archive path or link")
        tf.extractall(destination, filter="data")
    provenance = {
        "benchmark": "NATS-Bench TSS",
        "source": "https://github.com/D-X-Y/NATS-Bench",
        "source_archive_name": ARCHIVE_NAME,
        "source_archive_sha256": SHA256,
        "source_archive_bytes": archive.stat().st_size,
        "hp": "200",
        "search_dataset": "cifar10-valid",
        "search_metric": "get_more_info(..., hp='200', is_random=False)['valid-accuracy']",
        "training_seed_policy": "mean of available official training runs (is_random=False)",
        "api_version": "nats-bench 1.8",
        "fingerprint_scope": "SHA256 of the downloaded official archive; extracted files are not rehashed per proposal",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Verified and prepared {output}")


if __name__ == "__main__":
    main()
