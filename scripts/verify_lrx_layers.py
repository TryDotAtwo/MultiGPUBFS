"""Compare native rank histograms; this is NOT an archive/state-set verifier."""
import argparse
import json
import math
from pathlib import Path


def verify_layers(records, expected, n, world):
    if world not in (1, 2, 4, 8) or len(records) != world:
        raise ValueError("RANK_INVENTORY")
    if not expected or any(type(x) is not int or x <= 0 for x in expected):
        raise ValueError("REFERENCE_COUNTS")
    if expected[0] != 1 or sum(expected) != math.factorial(n):
        raise ValueError("REFERENCE_NOT_FULL_SYMMETRIC_GROUP")
    seen = set()
    totals = [0] * len(expected)
    for record in records:
        rank = record.get("rank")
        if type(rank) is not int or not 0 <= rank < world or rank in seen:
            raise ValueError("RANK_INVENTORY")
        seen.add(rank)
        if record.get("status") != "COMPLETE" or record.get("group") != f"s{n}":
            raise ValueError("RUN_INCOMPLETE_OR_WRONG_GROUP")
        rows = record.get("local_layer_sizes")
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise ValueError("DEPTH_COVERAGE")
        if any(type(x) is not int or x < 0 for x in rows):
            raise ValueError("INVALID_COUNT")
        totals = [a + b for a, b in zip(totals, rows)]
    for depth, (actual, wanted) in enumerate(zip(totals, expected)):
        if actual != wanted:
            raise ValueError(f"LAYER_MISMATCH depth={depth} actual={actual} expected={wanted}")
    return dict(status="PASS", scope="LAYER_COUNTS_ONLY", n=n, world=world,
                diameter=len(expected)-1, total_states=sum(totals), layers=totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank_results", type=Path)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[1]
                        / "data/reference/lrx13-layers.json")
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    records = [json.loads(p.read_text()) for p in args.rank_results.glob("rank-*.json")]
    result = verify_layers(records, reference["layers"], reference["n"], args.world)
    result["reference_source"] = reference["source"]
    result["reference_notebook_sha256"] = reference["notebook_sha256"]
    print(json.dumps(result))


if __name__ == "__main__":
    main()
