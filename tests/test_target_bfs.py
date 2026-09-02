import unittest


try:
    from multigpubfs import target_search
except (ImportError, ModuleNotFoundError):
    target_breadth_first_search = None
else:
    target_breadth_first_search = getattr(
        target_search, "target_breadth_first_search", None
    )


class TargetBfsTests(unittest.TestCase):
    def test_stop_granularity_changes_work_but_not_shortest_path(self):
        """Catches conflating candidate, parent-batch, and level stop work."""
        self.assertTrue(
            callable(target_breadth_first_search), "target BFS API is missing"
        )
        graph = {
            "s": (("to_p0", "p0"), ("to_p1", "p1"), ("to_p2", "p2")),
            "p0": (("to_t", "t"), ("to_x", "x")),
            "p1": (("to_y", "y"), ("to_z", "z")),
            "p2": (("to_q", "q"), ("to_r", "r")),
            "t": (),
            "x": (),
            "y": (),
            "z": (),
            "q": (),
            "r": (),
        }

        def transitions(state):
            return graph[state]

        candidate = target_breadth_first_search(
            transitions,
            start="s",
            target="t",
            stop_granularity="candidate",
        )
        parent_batch = target_breadth_first_search(
            transitions,
            start="s",
            target="t",
            stop_granularity="parent_batch",
            parent_batch_size=2,
        )
        level = target_breadth_first_search(
            transitions,
            start="s",
            target="t",
            stop_granularity="level",
        )

        for result in (candidate, parent_batch, level):
            self.assertTrue(result.found)
            self.assertEqual(result.distance, 2)
            self.assertEqual(result.moves, ("to_p0", "to_t"))

        self.assertEqual(candidate.generated_transitions, 4)
        self.assertEqual(parent_batch.generated_transitions, 7)
        self.assertEqual(level.generated_transitions, 9)


if __name__ == "__main__":
    unittest.main()
