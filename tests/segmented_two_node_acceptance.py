"""Run via multigpubfs cluster with one CUDA rank on each of two physical hosts.

For compact traversal with three TX and RX slots, export
MGBFS_SEGMENT_PACKET_RING_SLOTS=3 and MGBFS_SEGMENT_PACKET_RX_SLOTS=3
on both ranks before launch. Leave those variables unset for exact traversal.
Route queries always exercise the segmented compact engine.

Keep the two scheduler node names or provider machine IDs with the results.
The fixture checks distinct rank hostnames and actual communication.
"""

import argparse
from itertools import product
import os
from pathlib import Path

from multigpubfs import GraphDefinition, open_session


def check_traversal_transport(report, transport, slots):
    ranks = report["segment_memory_by_rank"]
    assert len(ranks) == 2
    if transport == "exact":
        assert all(rank["exact_exchange"] and rank.get("packet_transport") is None for rank in ranks)
        return

    packets = [rank["packet_transport"] for rank in ranks]
    protocol = "nccl_compact_credit_v2" if slots[1] == 3 else "nccl_compact_credit_v1"
    assert all(packet["protocol"] == protocol for packet in packets)
    assert all((packet["tx_exchange_slots"], packet["rx_exchange_slots"]) == slots for packet in packets)
    assert sum(packet["submitted_payload_bytes"] for packet in packets) > 0
    assert all(rank["global_sync"]["nccl_round_gathers"] == 0 for rank in ranks)


def check_window_engine(report, hosts):
    assert report["backend"] == "native_cuda_nccl"
    assert report["native_window_storage"] == "SEGMENTED_LAYER_BATCH_HASH"
    assert report["native_window_exchange"] == "COMPACT_PACKET"
    assert report["lineage_read_mode"] == "rank_owned"
    assert [rank["hostname"] for rank in report["native_window_workspace"]["ranks"]] == hosts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--transport", choices=("compact", "exact"), default="compact",
                        help="Exchange for both traversal queries (default: compact)")
    args = parser.parse_args()

    slots = (os.environ.get("MGBFS_SEGMENT_PACKET_RING_SLOTS", "1"),
             os.environ.get("MGBFS_SEGMENT_PACKET_RX_SLOTS", "1"))
    if slots not in (("1", "1"), ("3", "1"), ("3", "3")):
        parser.error("unsupported compact TX/RX slot configuration")
    if args.transport == "exact" and slots != ("1", "1"):
        parser.error("unset compact TX/RX slots for exact traversal")
    slots = tuple(map(int, slots))

    graph = GraphDefinition.permutation(
        [[1, 0, 2, 3], [0, 2, 1, 3], [0, 1, 3, 2]], [0, 1, 2, 3],
    )
    expected = graph.exact_layers(24)
    options = {
        "memory_backend": "segmented24", "segment_strategy": "atomic", "transport": args.transport,
        "batch": 16, "shards": 2, "segment_rows": 16, "queue_capacity": 128,
    }
    with open_session(gpus=2, executable=args.executable, tune="off") as session:
        partial = session.traverse(
            graph, output=args.output / "limited", native_options=dict(options, capacity=1),
        )
        assert partial.report["status"] == "INCOMPLETE"
        assert partial.report["reason"] == "RESOURCE_10"
        assert partial.layer_sizes == [len(layer) for layer in expected[:partial.completed_depth + 1]]
        assert sorted(partial.last_layer()) == expected[partial.completed_depth]

        complete = session.traverse(
            graph, output=args.output / "complete", native_options=dict(options, capacity=32),
        )
        assert complete.bfs_completed
        assert complete.layer_sizes == [len(layer) for layer in expected]
        assert sorted(complete.last_layer()) == expected[-1]
        assert sorted(complete.get_layer(complete.completed_depth - 1)) == expected[-2]
        hosts = complete.report["rank_hostnames"]
        assert len(hosts) == len(set(hosts)) == 2
        assert partial.report["rank_hostnames"] == hosts
        pids = complete.report["rank_worker_pids"]
        assert len(pids) == 2 and partial.report["rank_worker_pids"] == pids

        check_traversal_transport(complete.report, args.transport, slots)

        target = [3, 2, 1, 0]
        nearest = [
            session.solve(graph, targets=[target], direction=direction,
                          output=args.output / f"nearest-{direction}")
            for direction in ("forward", "bidirectional")
        ]
        identities = []
        for result in nearest:
            assert result.status == "COMPLETE"
            assert result.report["backend"] == "native_cuda_nccl"
            assert result.report["native_storage"] == "SEGMENTED_BATCH_HASH"
            assert result.report["native_exchange"] == "COMPACT_PACKET"
            routes = list(result.iter_paths())
            assert len(routes) == 1
            generators = routes[0]["generator_ids"]
            assert len(generators) == len(expected) - 1
            state = graph.start
            for generator in generators:
                permutation = graph.action["generators"][generator]
                state = [state[index] for index in permutation]
            assert state == target
            workspace = result.report["native_workspace"]["ranks"]
            assert [rank["hostname"] for rank in workspace] == hosts
            identities.append([rank["worker_pid"] for rank in workspace])
        assert identities[0] == identities[1]

        aliased = GraphDefinition.permutation([[1, 0], [1, 0]], [0, 1])
        options = dict(targets=[[1, 0]], goal_mode="window", extra_steps=2,
                       max_lineage_bytes=1 << 20, lineage_cache_bytes=128)
        exact = {(0, labels) for length in (1, 3) for labels in product(range(2), repeat=length)}
        for direction in ("forward", "bidirectional"):
            result = session.solve(aliased, direction=direction,
                                   output=args.output / f"window-{direction}", **options)
            assert result.status == "COMPLETE"
            check_window_engine(result.report, hosts)
            assert result.report["completed_through_length"] == 3
            saved = [(row["target_index"], tuple(row["generator_ids"])) for row in result.iter_paths()]
            assert len(saved) == len(exact) and set(saved) == exact

        stopped = session.solve(aliased, direction="bidirectional", max_paths=3,
                                output=args.output / "window-partial", **options)
        assert stopped.status == "INCOMPLETE" and stopped.report["stop_reason"] == "PATH_LIMIT"
        check_window_engine(stopped.report, hosts)
        assert stopped.report["completed_through_length"] == 2
        saved = [(row["target_index"], tuple(row["generator_ids"])) for row in stopped.iter_paths()]
        assert len(saved) == 3 and len(set(saved)) == 3 and set(saved) <= exact
        assert [len(labels) for _, labels in saved] == [1, 1, 3]
    print(f"{args.transport.upper()}_TWO_NODE_PASS")


if __name__ == "__main__":
    main()
