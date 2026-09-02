import unittest
from dataclasses import replace


try:
    from multigpubfs import implicit
except (ImportError, ModuleNotFoundError):
    labeled_breadth_first_search = None
    reconstruct_moves = None
    replay_moves = None
    validate_labeled_bfs_result = None
else:
    labeled_breadth_first_search = getattr(
        implicit, "labeled_breadth_first_search", None
    )
    reconstruct_moves = getattr(implicit, "reconstruct_moves", None)
    replay_moves = getattr(implicit, "replay_moves", None)
    validate_labeled_bfs_result = getattr(
        implicit, "validate_labeled_bfs_result", None
    )


class ImplicitBfsTests(unittest.TestCase):
    def test_enumerates_s3_and_replays_a_shortest_generator_path(self):
        """Catches missing move metadata or incorrect implicit expansion order."""
        self.assertTrue(callable(labeled_breadth_first_search), "labeled BFS is missing")
        self.assertTrue(callable(reconstruct_moves), "path reconstruction is missing")
        self.assertTrue(callable(replay_moves), "move replay is missing")

        generators = {
            "swap01": (1, 0, 2),
            "swap12": (0, 2, 1),
        }

        def apply(state, move):
            permutation = generators[move]
            return tuple(state[source] for source in permutation)

        def transitions(state):
            return tuple((move, apply(state, move)) for move in generators)

        identity = (0, 1, 2)
        target = (2, 1, 0)
        result = labeled_breadth_first_search(transitions, sources=[identity])

        self.assertEqual(
            result.frontiers,
            (
                ((0, 1, 2),),
                ((1, 0, 2), (0, 2, 1)),
                ((1, 2, 0), (2, 0, 1)),
                ((2, 1, 0),),
            ),
        )
        self.assertEqual(result.distance[target], 3)
        self.assertEqual(result.generated_transitions, 12)

        moves = reconstruct_moves(result, target)
        self.assertEqual(moves, ("swap01", "swap12", "swap01"))
        self.assertEqual(replay_moves(identity, moves, apply), target)

    def test_validator_rejects_parent_move_that_does_not_produce_the_child(self):
        """Catches path metadata that has the right depth but cannot replay."""
        self.assertTrue(
            callable(validate_labeled_bfs_result), "labeled BFS validator is missing"
        )
        transitions_by_state = {
            "s": (("left", "a"), ("right", "b")),
            "a": (),
            "b": (),
        }
        result = labeled_breadth_first_search(
            lambda state: transitions_by_state[state], sources=["s"]
        )
        broken_moves = dict(result.parent_move)
        broken_moves["a"] = "right"
        broken = replace(result, parent_move=broken_moves)

        self.assertIn(
            "parent move 'right' from 's' does not produce 'a'",
            validate_labeled_bfs_result(
                lambda state: transitions_by_state[state], broken
            ),
        )


if __name__ == "__main__":
    unittest.main()
