import unittest


try:
    from multigpubfs.distributed_bidirectional import (
        distributed_bidirectional_breadth_first_search,
    )
except (ImportError, ModuleNotFoundError):
    distributed_bidirectional_breadth_first_search = None


class DistributedBidirectionalTests(unittest.TestCase):
    def test_owner_computes_preserves_path_and_round_accounting(self):
        """Catches lost routed states and inconsistent dedup/traffic metrics."""
        self.assertTrue(
            callable(distributed_bidirectional_breadth_first_search),
            "distributed bidirectional BFS API is missing",
        )
        generators = ((1, 0, 2), (0, 2, 1))

        def apply(state, move):
            return tuple(state[source] for source in generators[move])

        def transitions(state):
            return tuple((move, apply(state, move)) for move in range(2))

        start = (0, 1, 2)
        target = (2, 1, 0)
        result = distributed_bidirectional_breadth_first_search(
            transitions,
            transitions,
            start=start,
            target=target,
            owner=lambda state: sum((index + 1) * value for index, value in enumerate(state)) % 3,
            world_size=3,
            expansion_policy="alternating",
        )

        self.assertTrue(result.found)
        self.assertEqual(result.distance, 3)
        state = start
        for move in result.moves:
            state = apply(state, move)
        self.assertEqual(state, target)
        self.assertTrue(result.rounds)
        for round_metrics in result.rounds:
            self.assertEqual(
                round_metrics.generated_transitions,
                round_metrics.source_duplicate_occurrences
                + round_metrics.source_unique_candidates,
            )
            self.assertEqual(
                round_metrics.source_unique_candidates,
                round_metrics.local_after_source_dedup
                + round_metrics.remote_after_source_dedup,
            )
            self.assertEqual(
                round_metrics.source_unique_candidates,
                round_metrics.owner_duplicate_occurrences
                + round_metrics.owner_unique_candidates,
            )
            self.assertEqual(
                round_metrics.owner_unique_candidates,
                round_metrics.already_visited + round_metrics.newly_discovered,
            )
            self.assertLessEqual(
                round_metrics.remote_only_newly_discovered,
                round_metrics.newly_discovered,
            )
            self.assertLessEqual(
                round_metrics.remote_accept_bitmap_bytes,
                round_metrics.remote_after_source_dedup,
            )

    def test_rejects_owner_outside_world(self):
        """Catches silently routing a state to a nonexistent rank."""
        with self.assertRaises(ValueError):
            distributed_bidirectional_breadth_first_search(
                lambda _state: (("edge", 1),),
                lambda _state: (),
                start=0,
                target=1,
                owner=lambda _state: 2,
                world_size=2,
            )


if __name__ == "__main__":
    unittest.main()
