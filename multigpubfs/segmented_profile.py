"""Choose a source batch from exact external segmented BFS prefixes."""

_MIN_DEPTH = 4
_MIN_POST_WARMUP_STATES = 32768
_MIN_IMPROVEMENT = 0.05
_PLAN_SIZING_FIELDS = {
    "batch", "capacity", "queue_capacity", "table_slots", "device_bytes",
    "sorted_owner_bytes", "driver_graph_reserve_bytes",
}


def _fixed_plan(plan):
    return {key: value for key, value in plan.items() if key not in _PLAN_SIZING_FIELDS}


def _pilot_evidence(admission, reports, requested_batch):
    devices = admission["devices"]
    rank_plans = admission["rank_plans"]
    if (not devices or len(reports) != len(devices) or len(rank_plans) != len(devices)
            or admission["plan"]["batch"] != requested_batch):
        raise RuntimeError("EXTERNAL_PROFILE_ADMISSION_MISMATCH")

    first = reports[0]
    sizes = first["layer_sizes"]
    if not sizes or sizes[0] != 1 or any(type(count) is not int or count < 1 for count in sizes):
        raise RuntimeError("EXTERNAL_PROFILE_PREFIX_MISMATCH")
    backend = first["backend"]
    strategy = first["segment_strategy"]
    if backend not in ("EXPERIMENTAL_SEGMENTED24", "EXPERIMENTAL_PERMUTATION_BITMAP"):
        raise RuntimeError("EXTERNAL_PROFILE_BACKEND_MISMATCH")

    owner_cuts = None
    if backend == "EXPERIMENTAL_SEGMENTED24":
        owner_cuts = first["segment_memory"].get("admitted_owner_cuts")
        if (not isinstance(owner_cuts, list) or len(owner_cuts) != len(devices) + 1
                or any(type(cut) is not int for cut in owner_cuts)
                or owner_cuts[0] != 0 or owner_cuts[-1] != 1 << 32
                or any(left >= right for left, right in zip(owner_cuts, owner_cuts[1:]))):
            raise RuntimeError("EXTERNAL_PROFILE_OWNER_CUTS_MISMATCH")

    committed, actual_batches = [], []
    for rank, (device, plan, report) in enumerate(zip(devices, rank_plans, reports)):
        if (plan["batch"] != requested_batch
                or report["rank"] != rank or report["world"] != len(devices) or report["device"] != device
                or report["graph_digest"] != admission["graph_digest"] or report["plan"] != plan):
            raise RuntimeError("EXTERNAL_PROFILE_RANK_GEOMETRY_MISMATCH")
        if any(report[key] != first[key] for key in ("status", "reason", "layer_sizes")):
            raise RuntimeError("EXTERNAL_PROFILE_PREFIX_MISMATCH")
        if report["backend"] != backend or report["segment_strategy"] != strategy:
            raise RuntimeError("EXTERNAL_PROFILE_BACKEND_MISMATCH")
        stamps = report["layer_committed_seconds"]
        if (len(stamps) != len(sizes) or stamps[0] != 0
                or any(type(stamp) not in (int, float) or not 0 <= stamp < float("inf") for stamp in stamps)
                or any(a > b for a, b in zip(stamps, stamps[1:]))):
            raise RuntimeError("EXTERNAL_PROFILE_INVALID_COMMIT_TIME")
        memory = report["segment_memory"]
        if owner_cuts is not None:
            rank_cuts = memory.get("admitted_owner_cuts")
            if rank_cuts != owner_cuts or any(type(cut) is not int for cut in rank_cuts):
                raise RuntimeError("EXTERNAL_PROFILE_OWNER_CUTS_MISMATCH")
        actual_batch = memory["admitted_batch"]
        if type(actual_batch) is not int or not 1 <= actual_batch <= requested_batch:
            raise RuntimeError("EXTERNAL_PROFILE_RANK_GEOMETRY_MISMATCH")
        committed.append(stamps)
        actual_batches.append(actual_batch)
    return sizes, committed, tuple(actual_batches), (backend, strategy), owner_cuts


def select_batch_pilot(baseline, probe, max_attempts=3):
    """Return the selected admission and evidence for up to three collective probes."""
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError("EXTERNAL_PROFILE_ATTEMPT_LIMIT")
    batch = baseline["plan"]["batch"]
    batches = tuple(dict.fromkeys((batch, max(1, batch // 4), max(1, batch // 16))))
    receipt = {
        "status": "INSUFFICIENT_PREFIX_CONSERVATIVE_PROFILE",
        "scope": "SAME_SEGMENTED_BACKEND_COMMITTED_LAYER_PREFIX_EXCLUDING_FIRST_LAYER",
        "measured": False,
        "baseline_batch": batch,
        "selected_batch": batch,
        "attempted_batches": [],
        "pilots": [],
        "common_depth": None,
        "common_layer_sizes": [],
        "scores_seconds": [],
    }
    if len(batches) < 2 or max_attempts == 1:
        return baseline, receipt

    eligible = []
    reference, terminal_depth = [], None
    cold_anchor, engine, anchor_cuts = None, None, None
    for requested in batches[:max_attempts]:
        receipt["attempted_batches"].append(requested)
        trial = {"requested_batch": requested, "status": "UNAVAILABLE", "seconds": None}
        receipt["pilots"].append(trial)
        result = probe(requested)
        if result is None:
            if requested == batch:
                break
            continue
        admission, reports = result
        if any(admission[key] != baseline[key] for key in ("graph_digest", "devices")):
            raise RuntimeError("EXTERNAL_PROFILE_ADMISSION_MISMATCH")
        sizes, committed, actual_batches, backend, owner_cuts = _pilot_evidence(admission, reports, requested)
        common = min(len(reference), len(sizes))
        if reference[:common] != sizes[:common]:
            raise RuntimeError("EXTERNAL_PROFILE_PREFIX_MISMATCH")
        if len(sizes) > len(reference):
            reference = sizes
        status, reason = reports[0]["status"], reports[0]["reason"]
        if status == "COMPLETE":
            if terminal_depth is not None and terminal_depth != len(sizes) - 1:
                raise RuntimeError("EXTERNAL_PROFILE_PREFIX_MISMATCH")
            terminal_depth = len(sizes) - 1
        if terminal_depth is not None and len(reference) - 1 > terminal_depth:
            raise RuntimeError("EXTERNAL_PROFILE_PREFIX_MISMATCH")
        trial.update(status=status, reason=reason, completed_depth=len(sizes) - 1,
                     admitted_batches_by_rank=list(actual_batches), admitted_owner_cuts=owner_cuts)

        reference_plan = cold_anchor if cold_anchor is not None else baseline
        if (_fixed_plan(admission["plan"]) != _fixed_plan(reference_plan["plan"])
                or len(admission["rank_plans"]) != len(reference_plan["rank_plans"])
                or any(_fixed_plan(plan) != _fixed_plan(original)
                       for plan, original in zip(admission["rank_plans"], reference_plan["rank_plans"]))
                or (cold_anchor is not None
                    and (admission["owner_cuts"] != cold_anchor["owner_cuts"] or backend != engine
                         or owner_cuts != anchor_cuts))):
            trial["comparison"] = "INCOMPARABLE_GEOMETRY"
            if requested == batch:
                break
            continue
        if cold_anchor is None:
            cold_anchor, engine, anchor_cuts = admission, backend, owner_cuts

        if (len(sizes) - 1 < _MIN_DEPTH or sum(sizes[2:]) < _MIN_POST_WARMUP_STATES
                or (status, reason) not in (("COMPLETE", "FRONTIER_EXHAUSTED"),
                                            ("INCOMPLETE", "PROFILE_LAYER_LIMIT"), ("INCOMPLETE", "DEADLINE"))):
            trial["comparison"] = "INSUFFICIENT_PREFIX"
            if requested == batch:
                break
            continue
        if eligible and (any(value > bound for value, bound in zip(actual_batches, eligible[0][0]))
                         or any(actual_batches == entry[0] for entry in eligible)):
            trial["comparison"] = "NO_SMALLER_DISTINCT_BATCH"
            continue
        eligible.append((actual_batches, admission, committed, trial))

    if len(eligible) < 2:
        return baseline, receipt
    depth = min(len(committed[0]) - 1 for _, _, committed, _ in eligible)
    for _, _, committed, trial in eligible:
        seconds = max(stamps[depth] - stamps[1] for stamps in committed)
        if seconds <= 0:
            raise RuntimeError("EXTERNAL_PROFILE_INVALID_COMMIT_TIME")
        trial["seconds"] = seconds
        trial["comparison"] = "MATCHED_PREFIX"
    anchor = eligible[0][3]["seconds"]
    _, selected, _, winner = min(eligible, key=lambda entry: entry[3]["seconds"])
    if winner["seconds"] >= anchor * (1 - _MIN_IMPROVEMENT):
        selected = baseline
    receipt.update(status="MEASURED_EQUAL_PREFIX_EXTERNAL_BATCH_PROFILE", measured=True,
                   selected_batch=selected["plan"]["batch"], common_depth=depth,
                   common_layer_sizes=reference[:depth + 1],
                   scores_seconds=[trial["seconds"] for trial in receipt["pilots"]],
                   minimum_switch_improvement=_MIN_IMPROVEMENT)
    return selected, receipt
