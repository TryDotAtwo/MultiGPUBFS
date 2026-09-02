import unittest


try:
    from multigpubfs import bidirectional
except (ImportError, ModuleNotFoundError):
    bidirectional_breadth_first_search = None
else:
    bidirectional_breadth_first_search = getattr(
        bidirectional, "bidirectional_breadth_first_search", None
    )


class BidirectionalBfsTests(unittest.TestCase):
    def test_expansion_policies_are_observable_without_changing_distance(self):
        """Catches silently treating frontier cardinality as outgoing work."""
        outgoing = {
            0: tuple((f"0to{x}", x) for x in range(1, 6)),
            1: (("1to8", 8),),
            2: (),
            3: (),
            4: (),
            5: (),
            8: (("8to9", 9),),
            9: (),
        }
        incoming = {state: [] for state in outgoing}
        for parent, edges in outgoing.items():
            for move, child in edges:
                incoming[child].append((move, parent))
        incoming = {state: tuple(edges) for state, edges in incoming.items()}

        smaller = bidirectional_breadth_first_search(
            lambda state: outgoing[state],
            lambda state: incoming[state],
            start=0,
            target=9,
            expansion_policy="smaller_frontier",
        )
        estimated = bidirectional_breadth_first_search(
            lambda state: outgoing[state],
            lambda state: incoming[state],
            start=0,
            target=9,
            expansion_policy="estimated_work",
            forward_work_estimate=lambda state: len(outgoing[state]),
            reverse_work_estimate=lambda state: len(incoming[state]),
        )
        alternating = bidirectional_breadth_first_search(
            lambda state: outgoing[state],
            lambda state: incoming[state],
            start=0,
            target=9,
            expansion_policy="alternating",
        )

        self.assertEqual(smaller.distance, 3)
        self.assertEqual(estimated.distance, 3)
        self.assertEqual(alternating.distance, 3)
        self.assertEqual(smaller.expansion_trace[0], "forward")
        self.assertEqual(estimated.expansion_trace[0], "reverse")
        self.assertEqual(alternating.expansion_trace[:2], ("forward", "reverse"))

    def test_finds_and_replays_shortest_path_in_s3(self):
        """Catches incorrect reverse metadata, meeting, or path concatenation."""
        self.assertTrue(
            callable(bidirectional_breadth_first_search),
            "bidirectional BFS API is missing",
        )
        generators = {
            "swap01": (1, 0, 2),
            "swap12": (0, 2, 1),
        }

        def apply(state, move):
            permutation = generators[move]
            return tuple(state[source] for source in permutation)

        def transitions(state):
            return tuple((move, apply(state, move)) for move in generators)

        start = (0, 1, 2)
        target = (2, 1, 0)
        result = bidirectional_breadth_first_search(
            transitions,
            transitions,
            start=start,
            target=target,
        )

        self.assertTrue(result.found)
        self.assertEqual(result.distance, 3)
        self.assertEqual(result.moves, ("swap01", "swap12", "swap01"))
        state = start
        for move in result.moves:
            state = apply(state, move)
        self.assertEqual(state, target)

    def test_identical_endpoints_return_empty_path_without_expansion(self):
        """Catches expanding or inventing a move for the zero-distance case."""
        result = bidirectional_breadth_first_search(
            lambda state: (("loop", state),),
            lambda state: (("loop", state),),
            start="same",
            target="same",
        )

        self.assertTrue(result.found)
        self.assertEqual(result.distance, 0)
        self.assertEqual(result.moves, ())
        self.assertEqual(result.generated_transitions, 0)

    def test_reports_no_path_in_a_directed_graph(self):
        """Catches false meetings when reverse reachability is disconnected."""
        outgoing = {
            0: (("0to1", 1),),
            1: (),
            2: (),
        }
        incoming = {
            0: (),
            1: (("0to1", 0),),
            2: (),
        }

        result = bidirectional_breadth_first_search(
            lambda state: outgoing[state],
            lambda state: incoming[state],
            start=0,
            target=2,
        )

        self.assertFalse(result.found)
        self.assertIsNone(result.distance)
        self.assertEqual(result.moves, ())


if __name__ == "__main__":
    unittest.main()
