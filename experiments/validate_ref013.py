"""Validate the committed REF-013 JSONL artifact shape and coverage."""

import json
from pathlib import Path


path = Path(__file__).with_name("REF-013-bitmap-sweep.jsonl")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
assert len(rows) == 16
assert all(row["status"] == "pass" for row in rows)
actual = {(row["pattern"], row["candidate_count"]) for row in rows}
expected = {
    (pattern, 1 << exponent)
    for pattern in ("all-new", "half-seeded-fourfold", "all-seen", "single-key")
    for exponent in (16, 20, 22, 24)
}
assert actual == expected
assert all(row["accepted_count"] <= row["candidate_count"] for row in rows)
print(f"REF-013 JSONL OK: {len(rows)} rows")
