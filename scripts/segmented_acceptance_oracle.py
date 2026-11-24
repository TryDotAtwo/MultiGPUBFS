"""Independent output checks for adjacent-transposition BFS from the identity."""

import json
from math import factorial


def mahonian(n):
    sizes = [1]
    for k in range(2, n + 1):
        next_sizes = [0] * (len(sizes) + k - 1)
        for j, count in enumerate(sizes):
            for delta in range(k):
                next_sizes[j + delta] += count
        sizes = next_sizes
    return sizes


def verify_output(n, report, states_path, *, terminal=None):
    expected = mahonian(n)
    sizes = report["layer_sizes"]
    if not sizes or sizes != expected[:len(sizes)]:
        raise ValueError("Mahonian layer mismatch")
    complete = report["status"] == "COMPLETE"
    states = json.loads(states_path.read_text(encoding="utf-8"))
    if complete:
        if len(sizes) != len(expected) or sum(sizes) != factorial(n):
            raise ValueError("Incomplete Mahonian sequence in complete result")
        last = tuple(reversed(range(n)))
        previous = {
            last[:i] + (last[i + 1], last[i]) + last[i + 2:]
            for i in range(n - 1)
        }
        witness = states if terminal is None else terminal
        current = [tuple(row) for row in witness["current"]]
        retained_previous = [tuple(row) for row in witness["previous_small"]]
        if current != [last] or len(retained_previous) != len(previous) or set(retained_previous) != previous:
            raise ValueError("Terminal layer sets or multiplicities do not match")
        if terminal is not None:
            for field, expected, count in (
                ("current", {last}, min(sizes[-1], 1000)),
                ("previous_small", previous, sizes[-2] if sizes[-2] < 1000 else 0),
            ):
                samples = {tuple(row) for row in states[field]}
                if len(states[field]) != count or len(samples) != count or not samples <= expected:
                    raise ValueError(f"Invalid retained terminal sample in {field}")
    else:
        depth = len(sizes) - 1
        previous_count = sizes[-2] if depth and sizes[-2] < 1000 else 0
        symbols = list(range(n))
        for key, layer, count in (
            ("current", depth, min(sizes[-1], 1000)),
            ("previous_small", depth - 1, previous_count),
        ):
            rows = [tuple(row) for row in states[key]]
            if len(rows) != count or len(set(rows)) != count:
                raise ValueError(f"Incorrect sample size or duplicate states in {key}")
            for row in rows:
                inversions = sum(a > b for i, a in enumerate(row) for b in row[i + 1:])
                if sorted(row) != symbols or inversions != layer:
                    raise ValueError(f"Invalid sampled state for depth {layer} in {key}")
    return {
        "all_completed_layer_counts_match_mahonian": True,
        "full_result": complete,
        "final_two_layers_full_exact_sets_no_duplicates": complete,
        "retained_samples_match_committed_layers_no_duplicates": True,
    }
