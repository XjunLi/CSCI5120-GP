"""NAS-Bench-201 / NATS-Bench TSS six-edge encoding."""
import random

OPS = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")
EDGES = ("0->1", "0->2", "1->2", "0->3", "1->3", "2->3")
SPACE_SIZE = 5 ** 6


def validate_ops(value):
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ValueError("ops must contain exactly six operations")
    if any(not isinstance(op, str) or op not in OPS for op in value):
        raise ValueError("Each operation must exactly match an allowed token")
    return tuple(value)


def canonical(ops):
    a, b, c, d, e, f = validate_ops(ops)
    return f"|{a}~0|+|{b}~0|{c}~1|+|{d}~0|{e}~1|{f}~2|"


def parse_architecture(text):
    import re
    groups = text.split("+")
    if len(groups) != 3:
        raise ValueError("Expected three destination-node groups")
    operations = []
    for destination, group in enumerate(groups, 1):
        pairs = re.findall(r"([^|~]+)~([0-9]+)", group)
        if len(pairs) != destination or [int(x[1]) for x in pairs] != list(range(destination)):
            raise ValueError("Incorrect source-node ordering")
        operations.extend(op for op, _ in pairs)
    ops = validate_ops(operations)
    if canonical(ops) != text:
        raise ValueError("Architecture string is not canonical")
    return ops


def random_unseen(rng, seen):
    if len(seen) >= SPACE_SIZE:
        raise RuntimeError("The encoding space is exhausted")
    while True:
        ops = tuple(rng.choice(OPS) for _ in EDGES)
        if ops not in seen:
            return ops


def warm_start(seed, count):
    rng, seen, values = random.Random(seed), set(), []
    for _ in range(count):
        ops = random_unseen(rng, seen)
        seen.add(ops)
        values.append(ops)
    return values, rng
