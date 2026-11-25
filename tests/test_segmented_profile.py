import pytest

from multigpubfs.segmented_profile import select_batch_pilot


@pytest.mark.parametrize("case", [
    "matched", "bitmap", "planned-cuts-changed", "live-cuts-changed",
    "rank-cuts-disagree", "invalid-cuts", "wrong-short-prefix",
])
def test_external_batch_pilots_compare_commits_from_every_rank(case):
    exact = [1, 2, 12000, 12000, 12000, 12000]

    def admission(batch, capacity=65536, cut=1 << 31):
        plan = {"batch": batch, "queue_capacity": 2 * batch, "shards": 4, "world": 2, "capacity": capacity}
        return {"graph_digest": "fixture", "devices": [0, 0], "owner_cuts": [0, cut, 1 << 32],
                "plan": plan, "rank_plans": [plan, plan]}

    attempted = []

    def probe(batch):
        attempted.append(batch)
        cut = 1 << (29 if batch == 16 and case == "planned-cuts-changed" else 30)
        candidate = admission(batch, capacity=32768, cut=cut)
        timings = {
            64: ([0, 10, 11, 12, 13, 14], [0, 1, 2.5, 4, 5.5, 7]),
            16: ([0, 20, 20.5, 21, 22], [0, 1, 2, 2.5, 3.5]),
            4: ([0, 1], [0, 2]),
        }
        reports = []
        for rank, stamps in enumerate(timings[batch]):
            sizes = exact[:len(stamps)]
            if batch == 4 and case == "wrong-short-prefix":
                sizes[-1] += 1
            bitmap = case == "bitmap"
            memory = {"admitted_batch": batch}
            if not bitmap:
                live_cut = 1 << 30
                if batch == 16 and (case == "live-cuts-changed" or case == "rank-cuts-disagree" and rank == 1):
                    live_cut = 1 << 29
                if batch == 16 and case == "invalid-cuts":
                    live_cut = 0
                memory["admitted_owner_cuts"] = [0, live_cut, 1 << 32]
            reports.append({"rank": rank, "world": 2, "device": 0, "plan": candidate["rank_plans"][rank],
                            "graph_digest": "fixture",
                            "backend": "EXPERIMENTAL_PERMUTATION_BITMAP" if bitmap else "EXPERIMENTAL_SEGMENTED24",
                            "segment_strategy": "warp", "segment_memory": memory,
                            "status": "INCOMPLETE", "reason": "PROFILE_LAYER_LIMIT",
                            "layer_sizes": sizes, "layer_committed_seconds": stamps})
        return candidate, reports

    if case == "wrong-short-prefix":
        with pytest.raises(RuntimeError, match="EXTERNAL_PROFILE_PREFIX_MISMATCH"):
            select_batch_pilot(admission(64), probe)
        assert attempted == [64, 16, 4]
        return
    if case in ("rank-cuts-disagree", "invalid-cuts"):
        with pytest.raises(RuntimeError, match="EXTERNAL_PROFILE_OWNER_CUTS_MISMATCH"):
            select_batch_pilot(admission(64), probe)
        assert attempted == [64, 16]
        return

    baseline = admission(64)
    selected, receipt = select_batch_pilot(baseline, probe)
    assert attempted == receipt["attempted_batches"] == [64, 16, 4]
    if case in ("planned-cuts-changed", "live-cuts-changed"):
        assert selected is baseline and not receipt["measured"]
        assert receipt["pilots"][1]["comparison"] == "INCOMPARABLE_GEOMETRY"
        if case == "live-cuts-changed":
            assert receipt["pilots"][1]["admitted_owner_cuts"] == [0, 1 << 29, 1 << 32]
        return
    assert selected["plan"]["batch"] == receipt["selected_batch"] == 16
    assert receipt["common_depth"] == 4 and receipt["common_layer_sizes"] == exact[:5]
    assert receipt["scores_seconds"] == [4.5, 2.5, None]
    expected_cuts = None if case == "bitmap" else [0, 1 << 30, 1 << 32]
    assert all(trial["admitted_owner_cuts"] == expected_cuts for trial in receipt["pilots"])
