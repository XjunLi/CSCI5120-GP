"""Trusted evaluators. Only the explicit validation whitelist reaches agents."""
import csv
import hashlib
import json
import math
from pathlib import Path

from .architecture import canonical, parse_architecture, validate_ops


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value, name, optional=False):
    if optional and value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite metric: {name}")
    return value


class TableOracle:
    """Normalized JSON/CSV export; never infer validation from a test column.

    JSON: {metadata: {...}, rows: [{ops: [...], val_accuracy: n,
           flops: n|null, params: n|null, test: {dataset: n}}]}.
    Rows may additionally carry validation: {dataset: n}, costs: {dataset: {...}}.
    CSV uses ops (JSON) or architecture and explicit val_accuracy/test_* columns;
    provenance metadata is required in <filename>.metadata.json.
    """
    def __init__(self, data, dataset="cifar10-valid", hp="200"):
        self.path = Path(data).expanduser().resolve()
        self.dataset, self.hp = dataset, str(hp)
        if self.path.suffix == ".csv":
            sidecar = Path(str(self.path) + ".metadata.json")
            metadata = json.loads(sidecar.read_text())
            with self.path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                if "ops" in row:
                    row["ops"] = json.loads(row["ops"])
                row["test"] = {k[5:]: v for k, v in row.items() if k.startswith("test_") and v != ""}
                for key in ("flops", "params"):
                    if row.get(key) == "":
                        row[key] = None
        else:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            metadata, rows = raw["metadata"], raw["rows"]
        self.metadata = dict(metadata)
        if str(metadata.get("hp", hp)) != self.hp:
            raise ValueError("Requested hp does not match the table training configuration")
        if not metadata.get("source") or not metadata.get("training_seed_policy"):
            raise ValueError("Table requires source and training_seed_policy provenance")
        self.label = metadata.get("benchmark", "NAS-Bench-201 derived table")
        self.training_seed_policy = metadata["training_seed_policy"]
        self.rows = {}
        for row in rows:
            ops = validate_ops(row["ops"]) if "ops" in row else parse_architecture(row["architecture"])
            if ops in self.rows:
                raise ValueError("Table contains duplicate architecture encodings")
            self.rows[ops] = row
        self.metadata.update({"normalized_data_sha256": sha256_file(self.path), "row_count": len(self.rows)})

    def validation(self, ops):
        row = self.rows[tuple(ops)]
        if "validation" in row:
            value = row["validation"][self.dataset]
        else:
            expected = self.metadata.get("search_dataset", "cifar10-valid")
            if self.dataset != expected:
                raise ValueError(f"Table supports validation for {expected}, not {self.dataset}")
            value = row["val_accuracy"]
        costs = row.get("costs", {}).get(self.dataset, row)
        return {"val_accuracy": number(value, "validation"),
                "flops": number(costs.get("flops"), "flops", optional=True),
                "params": number(costs.get("params"), "params", optional=True)}

    def final_test(self, ops):
        return {f"{dataset}_test_accuracy_hp{self.hp}": number(value, dataset)
                for dataset, value in self.rows[tuple(ops)].get("test", {}).items()}


class NatsOracle:
    label = "NATS-Bench TSS"
    training_seed_policy = "mean of available benchmark training runs (is_random=False)"

    def __init__(self, data, dataset="cifar10-valid", hp="200"):
        from nats_bench import create
        path = Path(data).expanduser().resolve()
        self.dataset, self.hp = dataset, str(hp)
        self.api = create(str(path), "tss", fast_mode=path.is_dir(), verbose=False)
        self.cache = {}
        provenance = {}
        for candidate in (path / "provenance.json", Path(str(path) + ".metadata.json")):
            if candidate.is_file():
                provenance = json.loads(candidate.read_text(encoding="utf-8"))
                break
        fingerprint = sha256_file(path) if path.is_file() else provenance.get("source_archive_sha256")
        if fingerprint is not None and (len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint)):
            raise ValueError("Invalid SHA256 in benchmark provenance")
        self.metadata = {"benchmark": self.label, "source": "https://github.com/D-X-Y/NATS-Bench",
                         "hp": self.hp, "training_seed_policy": self.training_seed_policy,
                         "data_sha256": fingerprint,
                         "data_identity_kind": "file SHA256" if path.is_file() else "source archive SHA256; no per-step extracted-file integrity scan",
                         "source_provenance": provenance, "archive_name": path.name}

    def validation(self, ops):
        ops = tuple(ops)
        if ops not in self.cache:
            index = self.api.query_index_by_arch(canonical(ops))
            if index is None or index < 0:
                raise ValueError("Encoding absent from official table")
            raw = self.api.get_more_info(index, self.dataset, hp=self.hp, is_random=False)
            cost = self.api.get_cost_info(index, self.dataset, hp=self.hp)
            self.cache[ops] = {"val_accuracy": number(raw["valid-accuracy"], "validation"),
                               "flops": number(cost.get("flops"), "flops", True),
                               "params": number(cost.get("params"), "params", True)}
        return dict(self.cache[ops])

    def final_test(self, ops):
        index = self.api.query_index_by_arch(canonical(ops))
        return {f"{dataset}_test_accuracy_hp200": number(
                    self.api.get_more_info(index, dataset, hp="200", is_random=False)["test-accuracy"], dataset)
                for dataset in ("cifar10", "cifar100", "ImageNet16-120")}


def load_oracle(backend, data, dataset="cifar10-valid", hp="200"):
    return {"table": TableOracle, "nats": NatsOracle}[backend](data, dataset, hp)
