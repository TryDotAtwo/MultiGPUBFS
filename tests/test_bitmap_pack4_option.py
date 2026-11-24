import pytest

from multigpubfs import GraphDefinition, launch, nccl_preflight


@pytest.mark.parametrize(
    ("setting", "inherited", "expected"),
    [(True, "0", "1"), (False, "1", "0"), (None, "1", "1")],
)
def test_pack4_reaches_native_preflight(tmp_path, monkeypatch, setting, inherited, expected):
    graph = GraphDefinition.permutation([[1, 0]], [-1, 511])
    env = {"MGBFS_SEGMENT_PACK4": inherited}
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(launch, "native_runtime", lambda _: ("native", {}))
    monkeypatch.setattr(launch, "_require_memory_backend", lambda *_: {"bitmap_pack4_experimental_v1": True})

    class ReachedPreflight(Exception):
        pass

    def prepare(native, devices, options, output, definition):
        assert options["MGBFS_SEGMENT_PACK4"] == expected
        assert options["MGBFS_SEGMENT_STRATEGY"] == "atomic"
        assert options["MGBFS_EXPERIMENTAL_BITMAP"] == "1"
        assert definition is graph
        raise ReachedPreflight

    monkeypatch.setattr(nccl_preflight, "prepare", prepare)
    with pytest.raises(ReachedPreflight):
        launch.run_graph(graph, tmp_path / "result", device=0, memory_backend="permutation_bitmap",
                         bitmap_pack4=setting, _native_env=env)
    assert env["MGBFS_SEGMENT_PACK4"] == inherited


@pytest.mark.parametrize(
    ("backend", "strategy", "degree", "error"),
    [("segmented24", "atomic", 2, "REQUIRES_BITMAP"),
     ("permutation_bitmap", "warp", 2, "STRATEGY_UNSUPPORTED"),
     ("permutation_bitmap", "atomic", 17, "UNSUPPORTED_DEGREE")],
)
def test_pack4_rejects_unsupported_configuration(tmp_path, backend, strategy, degree, error):
    graph = GraphDefinition.permutation([list(range(degree))], list(range(degree)))
    with pytest.raises(ValueError, match=f"BITMAP_PACK4_{error}"):
        launch.run_graph(graph, tmp_path / "result", memory_backend=backend, segment_strategy=strategy,
                         bitmap_pack4=True)


def test_pack4_requires_native_abi(tmp_path, monkeypatch):
    graph = GraphDefinition.permutation([[1, 0]], [0, 1])
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(launch, "native_runtime", lambda _: ("native", {}))
    monkeypatch.setattr(launch, "_require_memory_backend", lambda *_: {"permutation_bitmap_v1": True})
    with pytest.raises(RuntimeError, match="BITMAP_PACK4_NATIVE_ABI_MISSING"):
        launch.run_graph(graph, tmp_path / "result", memory_backend="permutation_bitmap", bitmap_pack4=True)


def test_advanced_cli_forwards_pack4(tmp_path, monkeypatch):
    from multigpubfs import cli

    graph = GraphDefinition.permutation([[1, 0]], [0, 1])
    definition = tmp_path / "graph.json"
    definition.write_text(graph.to_json(), encoding="utf-8")
    seen = []

    def run_graph(actual, output, **options):
        assert actual.digest() == graph.digest()
        seen.append((options["memory_backend"], options["bitmap_pack4"]))
        return {"status": "COMPLETE"}

    monkeypatch.setattr(cli, "run_graph", run_graph)
    command = [str(definition), str(tmp_path / "result"), "--memory-backend", "permutation_bitmap"]
    assert cli.main(command) == 0
    assert cli.main([*command, "--bitmap-pack4"]) == 0
    assert seen == [("permutation_bitmap", None), ("permutation_bitmap", True)]
