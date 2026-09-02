import unittest
from dataclasses import replace


try:
    from multigpubfs import reference
except ModuleNotFoundError:
    breadth_first_search = None
    validate_bfs_result = None
else:
    breadth_first_search = getattr(reference, "breadth_first_search", None)
    validate_bfs_result = getattr(reference, "validate_bfs_result", None)


class ReferenceBfsTests(unittest.TestCase):
    def test_assigns_shortest_distances_by_complete_levels(self):
        """Catches FIFO/visited bugs that assign a non-shortest distance."""
        self.assertTrue(callable(breadth_first_search), "reference BFS API is missing")

        graph = {
            "s": ("a", "b"),
            "a": ("c",),
            "b": ("c", "d"),
            "c": ("t",),
            "d": ("t",),
            "t": (),
        }

        result = breadth_first_search(lambda vertex: graph[vertex], sources=["s"])

        self.assertEqual(
            result.distance,
            {"s": 0, "a": 1, "b": 1, "c": 2, "d": 2, "t": 3},
        )
        self.assertEqual(
            result.frontiers,
            (("s",), ("a", "b"), ("c", "d"), ("t",)),
        )

    def test_validator_accepts_a_complete_shortest_path_tree(self):
        """Catches a validator that rejects a valid complete BFS result."""
        self.assertTrue(callable(validate_bfs_result), "BFS validator API is missing")

        graph = {
            0: (0, 1, 1),
            1: (0, 2),
            2: (1,),
        }
        result = breadth_first_search(lambda vertex: graph[vertex], sources=[0, 0])

        self.assertEqual(validate_bfs_result(lambda vertex: graph[vertex], result), ())

    def test_validator_rejects_parent_from_the_wrong_level(self):
        """Catches accepting a parent that cannot certify the recorded distance."""
        graph = {
            "s": ("a",),
            "a": ("b",),
            "b": (),
        }
        result = breadth_first_search(lambda vertex: graph[vertex], sources=["s"])
        broken_parent = dict(result.parent)
        broken_parent["b"] = "s"
        broken = replace(result, parent=broken_parent)

        self.assertIn(
            "parent edge/depth invalid for 'b': parent 's' is not at depth 1",
            validate_bfs_result(lambda vertex: graph[vertex], broken),
        )

    def test_validator_rejects_a_silently_dropped_reachable_vertex(self):
        """Catches frontier overflow or false visited hits that lose a vertex."""
        graph = {
            "s": ("a", "lost"),
            "a": (),
            "lost": (),
        }
        result = breadth_first_search(lambda vertex: graph[vertex], sources=["s"])
        broken_distance = dict(result.distance)
        broken_parent = dict(result.parent)
        broken_distance.pop("lost")
        broken_parent.pop("lost")
        broken = replace(
            result,
            distance=broken_distance,
            parent=broken_parent,
            frontiers=(("s",), ("a",)),
        )

        self.assertIn(
            "reachable child 'lost' of 's' is missing",
            validate_bfs_result(lambda vertex: graph[vertex], broken),
        )

    def test_validator_rejects_a_nonminimal_but_parent_consistent_depth(self):
        """Catches a tree that is connected but is not a shortest-path tree."""
        graph = {
            "s": ("a", "x"),
            "a": ("c",),
            "x": ("y",),
            "y": ("c",),
            "c": (),
        }
        result = breadth_first_search(lambda vertex: graph[vertex], sources=["s"])
        broken_distance = dict(result.distance)
        broken_parent = dict(result.parent)
        broken_distance["c"] = 3
        broken_parent["c"] = "y"
        broken = replace(
            result,
            distance=broken_distance,
            parent=broken_parent,
            frontiers=(("s",), ("a", "x"), ("y",), ("c",)),
        )

        self.assertIn(
            "edge 'a' -> 'c' violates shortest distances 1 -> 3",
            validate_bfs_result(lambda vertex: graph[vertex], broken),
        )

    def test_validator_rejects_frontier_that_disagrees_with_distance(self):
        """Catches a level buffer containing a vertex assigned to another depth."""
        graph = {
            0: (1,),
            1: (2,),
            2: (),
        }
        result = breadth_first_search(lambda vertex: graph[vertex], sources=[0])
        broken = replace(result, frontiers=((0,), (1, 2), ()))

        self.assertIn(
            "frontier 1 contains 2 with recorded depth 2",
            validate_bfs_result(lambda vertex: graph[vertex], broken),
        )


if __name__ == "__main__":
    unittest.main()
