import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("huggingface_hub")

from multigpubfs import publish
from multigpubfs.graph_definition import GraphDefinition


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def from_bytes(value):
    return pq.read_table(pa.BufferReader(value))


class LocalHub:
    endpoint = "https://huggingface.co"

    def __init__(self, cache):
        self.cache = cache
        readme = b"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: atlas.parquet
---
# Graph BFS Atlas
"""
        evidence = pa.struct([
            ("seed", pa.int32()), ("retention_role", pa.string()), ("source_file", pa.string()),
            ("archive_member", pa.string()), ("source_index", pa.int32()),
            ("source_schema", pa.int16()), ("current_sample_limit", pa.int32()),
        ])
        layer = pa.struct([
            ("depth", pa.int32()), ("new_unique_states", pa.int64()), ("cumulative_states", pa.int64()),
        ])
        vector = pa.list_(pa.int64())
        state = pa.struct([
            ("state", vector), ("seeds", pa.list_(pa.int32())), ("retention_roles", pa.list_(pa.string())),
            ("evidence", pa.list_(evidence)),
        ])
        schema = pa.schema([
            ("case_id", pa.string()), ("graph_family", pa.string()), ("n", pa.int32()), ("r", pa.int32()),
            ("status", pa.string()), ("max_distance_from_canonical_start", pa.int64()),
            ("observed_states", pa.int64()), ("evidence_prefix", pa.string()), ("source_repo", pa.string()),
            ("source_revision", pa.string()), ("evidence_kind", pa.string()), ("retained_unique_states", pa.int64()),
            ("layers", pa.list_(layer)), ("states", pa.list_(state)), ("start_state", vector),
        ])
        old = {"case_id": "lrx:n2:r1", "graph_family": "lrx", "n": 2, "r": 1, "status": "COMPLETE"}
        sink = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pylist([old], schema=schema), sink, row_group_size=1)
        self.atlas = sink.getvalue().to_pybytes()
        self.original_csv = (
            b"graph_family,n,r,status,depth,new_unique_states,cumulative_states,case_id,start_state\n"
            b"lrx,2,1,COMPLETE,0,1,1,lrx:n2:r1,\n"
        )
        self.snapshots = {}
        self.conflict = self.lost_response = None
        self.before_path_index = None
        self.commits = []
        self.record({
            "README.md": readme, "atlas.parquet": self.atlas, "sources.json": b"[]",
            "lrx-layers.csv": self.original_csv,
        })

    def record(self, files):
        self.head = f"{len(self.snapshots) + 1:040x}"
        self.snapshots[self.head] = files

    def repo_info(self, *, revision, **kwargs):
        return SimpleNamespace(sha=self.head if revision == "main" else revision, private=False, gated=False)

    def get_paths_info(self, *, paths, revision, **kwargs):
        files = self.snapshots[revision]
        return [
            SimpleNamespace(
                path=path, size=len(files[path]),
                blob_id=hashlib.sha1(f"blob {len(files[path])}\0".encode() + files[path]).hexdigest(),
                lfs=SimpleNamespace(sha256=sha(files[path])) if path.endswith(".parquet") else None,
            )
            for path in paths if path in files
        ]

    def hf_hub_download(self, *, filename, revision, **kwargs):
        key = sha(encoded([revision, filename]))[:24]
        path = self.cache / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.snapshots[revision][filename])
        return str(path)

    def create_commit(self, *, parent_commit, commit_message, operations, **kwargs):
        if self.before_path_index is not None and commit_message == "Index MultiGPUBFS path query":
            publish_other = self.before_path_index
            self.before_path_index = None
            publish_other()
        files = self.snapshots[self.head].copy()
        if self.conflict == commit_message:
            self.conflict = None
            files["concurrent-edit.txt"] = b"retain this"
            self.record(files)
        if parent_commit != self.head:
            raise RuntimeError("stale parent")
        for op in operations:
            content = op.path_or_fileobj
            files[op.path_in_repo] = Path(content).read_bytes() if isinstance(content, str) else content
        self.record(files)
        self.commits.append(commit_message)
        if self.lost_response == commit_message:
            self.lost_response = None
            raise ConnectionError("commit succeeded before the connection closed")
        return SimpleNamespace(oid=self.head)


def traversal(output, definition=None, *, sample_limit=1000):
    output.mkdir()
    if definition is None:
        definition = GraphDefinition.permutation([[1, 0]] * 3, [0, 1], name="LRX")
    start, final = definition.start, list(reversed(definition.start))
    snapshot = {
        "schema": 2, "state_encoding": "signed_int64_vectors",
        "current": [final], "previous_small": [start, final],
    }
    if sample_limit is not None:
        snapshot["current_sample_limit"] = sample_limit
    states = encoded(snapshot)
    (output / "states.json").write_bytes(states)
    (output / "report.json").write_bytes(encoded({
        "status": "COMPLETE", "layer_sizes": [1, 1], "states_sha256": sha(states),
        "graph_digest": definition.digest(), "seed": 2,
    }))
    (output / "graph.json").write_text(definition.to_json(), encoding="utf-8")
    return SimpleNamespace(output=output, graph=definition)


def paths(output, mode="window", status="COMPLETE", *, targets=((1, 0), (7, 8)),
          extra=0, completed=None, sequences=None):
    output.mkdir()
    graph = GraphDefinition.permutation([[1, 0], [1, 0]], [0, 1], name="example")
    if sequences is None:
        sequences = [[0], [1]] if mode == "window" and status == "COMPLETE" else [[0]]
    lengths = list(map(len, sequences))
    counts = sorted(Counter(lengths).items())
    table = pa.table({
        "target_index": pa.array([targets.index((1, 0))] * len(sequences), type=pa.int64()),
        "length": pa.array(lengths, type=pa.int64()),
        "generator_ids": pa.array(sequences, type=pa.list_(pa.int32())),
    })
    path = output / "paths.parquet"
    pq.write_table(table, path)
    content = path.read_bytes()
    report = {
        "mode": "paths", "goal_mode": mode, "status": status, "graph_family": "example", "n": 2,
        "graph_sha256": graph.digest(), "generators_sha256": sha(encoded(graph.action)),
        "start_state": graph.start, "targets": targets, "k": extra, "d_min": 1, "max_length": 1 + extra,
        "completed_through_length": 1 + extra if status == "COMPLETE" else completed,
        "counts": [{"length": length, "count": count} for length, count in counts],
        "path_count": len(sequences),
        "path_file": path.name, "path_sha256": sha(content), "path_bytes": len(content),
    }
    (output / "report.json").write_bytes(encoded(report))
    (output / "graph.json").write_text(graph.to_json(), encoding="utf-8")
    return report


def test_publish_traversal_preserves_existing_schema_and_recovers_conflict(tmp_path):
    hub = LocalHub(tmp_path / "cache")
    result = traversal(tmp_path / "traversal", sample_limit=None)
    hub.conflict = "Index MultiGPUBFS traversal"
    answer = publish(result, token=False, author="Ivan", api=hub)
    files = hub.snapshots[answer.revision]
    assert files["concurrent-edit.txt"] == b"retain this"
    table = from_bytes(files["atlas.parquet"])
    old_table = from_bytes(hub.atlas)
    assert table.schema.equals(old_table.schema)
    assert table.slice(0, 1).equals(old_table)
    row = table.to_pylist()[-1]
    assert answer.case_id == row["case_id"]
    assert answer.case_id.startswith("lrx:n2:r1:run-")
    assert row["layers"][-1] == {"depth": 1, "new_unique_states": 1, "cumulative_states": 2}
    occurrences = row["states"][0]["evidence"]
    assert [item["retention_role"] for item in occurrences] == ["current", "previous_small"]
    assert all(item["current_sample_limit"] is None for item in occurrences)
    assert occurrences[0]["source_file"] in hub.snapshots[row["source_revision"]]
    assert files["lrx-layers.csv"].startswith(hub.original_csv)
    assert answer.case_id.encode() in files["lrx-layers.csv"]
    assert {item["path"] for item in json.loads(files["sources.json"])} >= {occurrences[0]["source_file"]}
    assert publish(result.output, token=False, author="Ivan", api=hub) == answer
    assert len(hub.commits) == 2


def test_canonical_family_ids_require_matching_generators_and_start(tmp_path):
    hub = LocalHub(tmp_path / "cache")
    rapaport_m2 = [[1, 0, 2, 3], [1, 0, 3, 2], [0, 2, 1, 3]]
    starter = [0, 1, 2, 2]
    canonical = GraphDefinition.permutation(rapaport_m2, starter, name="rapaport_m2-4")
    relabelled = GraphDefinition.permutation(rapaport_m2, [2, 1, 0, 0], name="rapaport_m2-4")
    different = GraphDefinition.permutation(rapaport_m2[:2], starter, name="rapaport_m2-4")
    for i, (graph, case) in enumerate([
        (canonical, "rapaport_m2:n4:r2"),
        (relabelled, f"rapaport_m2:n4:graph-{relabelled.digest()}"),
        (different, f"rapaport_m2:n4:graph-{different.digest()}"),
    ]):
        result = traversal(tmp_path / f"run{i}", graph)
        assert publish(result, token=False, author="Ivan", api=hub).case_id == case
    lrx = GraphDefinition.permutation([[1, 0]], [0, 1], name="lrx-2")
    result = traversal(tmp_path / "custom-lrx", lrx)
    assert publish(result, token=False, author="Ivan", api=hub).case_id == f"lrx:n2:r1:graph-{lrx.digest()}"


def test_same_named_custom_graph_centres_are_separate_and_case_can_be_overridden(tmp_path):
    hub = LocalHub(tmp_path / "cache")
    first = GraphDefinition.permutation([[1, 2, 0]], [0, 1, 2], name="family")
    second = GraphDefinition.permutation([[1, 2, 0]], [1, 2, 0], name="family")
    third = GraphDefinition.permutation([[2, 1, 0]], [0, 1, 2], name="family")
    for i, graph in enumerate((first, second, third)):
        output = tmp_path / f"graph{i}"
        traversal(output, graph)
        assert publish(output, token=False, author="Ivan", api=hub).case_id == f"family:n3:graph-{graph.digest()}"
    override = traversal(tmp_path / "override", second)
    assert publish(override, token=False, author="Ivan", case_id="custom", api=hub).case_id == "custom"


def test_path_queries_have_pinned_artifacts_and_preserve_traversal(tmp_path):
    from huggingface_hub import DatasetCard

    hub = LocalHub(tmp_path / "cache")
    window = tmp_path / "window"
    nearest = tmp_path / "nearest"
    paths(window)
    paths(nearest, "nearest")
    hub.lost_response = "Index MultiGPUBFS path query"
    answer = publish(window, token=False, author="Ivan", api=hub, max_attempts=1)
    other = publish(nearest, token=False, author="Ivan", api=hub)
    assert answer.query_id != other.query_id
    files = hub.snapshots[hub.head]
    assert files["atlas.parquet"] == hub.atlas
    card = DatasetCard(files["README.md"].decode())
    assert card.data["configs"][0]["default"] is True
    assert card.data["configs"][1]["config_name"] == "path_queries"
    rows = from_bytes(files["path-queries.parquet"]).to_pylist()
    assert {row["goal_mode"] for row in rows} == {"nearest", "window"}
    row = next(row for row in rows if row["query_id"] == answer.query_id)
    assert row["targets"] == [[1, 0], [7, 8]]
    assert row["counts"] == [{"length": 1, "count": "2"}]
    artifacts = hub.snapshots[row["manifest_revision"]]
    manifest = artifacts[row["manifest_path"]]
    assert sha(manifest) == row["manifest_sha256"]
    parsed = json.loads(manifest)
    assert sha(artifacts[parsed["graph_path"]]) == parsed["graph_sha256"]
    part = parsed["parts"][0]
    assert from_bytes(artifacts[part["path"]])["generator_ids"].to_pylist() == [[0], [1]]
    pointer = {key: row[f"manifest_{key}"] for key in ("revision", "path", "sha256")}
    pointer["repo_id"] = row["manifest_repo"]
    assert publish(window, token=False, author="Ivan", api=hub, manifest=pointer).revision == hub.head
    assert publish(window, token=False, author="Ivan", api=hub).revision == hub.head
    assert len(hub.commits) == 4


@pytest.mark.parametrize(("fault", "error"), [
    ("omitted_rows", "row count differs"),
    ("schema", "generator-ID sequence schema"),
    ("target_index", "path rows are inconsistent"),
])
def test_external_manifest_must_have_matching_parquet_evidence(tmp_path, fault, error):
    output = tmp_path / "nearest"
    paths(output, "nearest")
    source = LocalHub(tmp_path / "external-cache")
    publish(output, token=False, author="Ivan", api=source)
    row = from_bytes(source.snapshots[source.head]["path-queries.parquet"]).to_pylist()[0]
    files = source.snapshots[row["manifest_revision"]]
    manifest = json.loads(files[row["manifest_path"]])
    part = manifest["parts"][0]
    original = from_bytes(files[part["path"]])
    if fault == "schema":
        invalid = original.set_column(2, "generator_ids", pa.array([[0]], type=pa.list_(pa.int64())))
    elif fault == "target_index":
        invalid = original.set_column(0, "target_index", pa.array([2], type=pa.int64()))
    else:
        invalid = original.slice(0, 0)
    buffer = pa.BufferOutputStream()
    pq.write_table(invalid, buffer)
    bad_part = buffer.getvalue().to_pybytes()
    part.update(sha256=sha(bad_part), bytes=len(bad_part))
    body = encoded(manifest)

    atlas = LocalHub(tmp_path / "atlas-cache")
    atlas.record({**atlas.snapshots[atlas.head], manifest["graph_path"]: files[manifest["graph_path"]],
                  row["manifest_path"]: body, part["path"]: bad_part})
    pointer = {"repo_id": "another/path-fixture", "path": row["manifest_path"],
               "revision": atlas.head, "sha256": sha(body)}
    with pytest.raises(ValueError, match=error):
        publish(output, token=False, author="Ivan", api=atlas, manifest=pointer)
    assert not atlas.commits


def test_bad_generator_ids_are_rejected_before_staging_even_beyond_old_coverage(tmp_path):
    hub = LocalHub(tmp_path / "cache")
    initial = hub.head
    wrong_nearest = tmp_path / "wrong-nearest"
    paths(wrong_nearest, mode="nearest", sequences=[[8]])
    with pytest.raises(ValueError, match="path rows are inconsistent"):
        publish(wrong_nearest, token=False, author="Ivan", api=hub)
    assert hub.head == initial and not hub.commits

    prior, candidate = (tmp_path / name for name in ("prior", "candidate"))
    paths(prior, status="INCOMPLETE", extra=2, completed=1, sequences=[[0], [1]])
    published = publish(prior, token=False, author="Ivan", api=hub, include_incomplete=True)
    paths(candidate, extra=2, sequences=[[0], [1], [0, 0, 8]])
    with pytest.raises(ValueError, match="path rows are inconsistent"):
        publish(candidate, token=False, author="Ivan", api=hub)
    assert hub.head == published.revision and len(hub.commits) == 2


def test_invalid_and_incomplete_path_results(tmp_path):
    hub = LocalHub(tmp_path / "cache")
    incomplete = tmp_path / "incomplete"
    complete = tmp_path / "complete"
    paths(incomplete, status="INCOMPLETE")
    paths(complete)
    with pytest.raises(ValueError, match="not eligible"):
        publish(incomplete, token=False, author="Ivan", api=hub)
    assert not hub.commits
    first = publish(incomplete, token=False, author="Ivan", api=hub, include_incomplete=True)
    second = publish(complete, token=False, author="Ivan", api=hub)
    assert first.query_id == second.query_id
    rows = from_bytes(hub.snapshots[hub.head]["path-queries.parquet"]).to_pylist()
    assert len(rows) == 1 and rows[0]["status"] == "COMPLETE"
    with pytest.raises(ValueError, match="different terminal"):
        publish(incomplete, token=False, author="Ivan", api=hub, include_incomplete=True)
    (complete / "paths.parquet").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="checksum or size"):
        publish(complete, token=False, author="Ivan", api=hub)
    assert len(hub.commits) == 4

    bogus = tmp_path / "bogus"
    report = paths(bogus, mode="nearest")
    report["path_sha256"] = "0" * 64
    (bogus / "report.json").write_bytes(encoded(report))
    with pytest.raises(ValueError, match="checksum or size"):
        publish(bogus, token=False, author="Ivan", api=hub)
    assert len(hub.commits) == 4


def test_concurrent_path_writers_preserve_ordered_target_identity(tmp_path):
    hub = LocalHub(tmp_path / "cache")
    direct, reordered = (tmp_path / name for name in ("direct", "reordered"))
    paths(direct)
    paths(reordered, targets=((7, 8), (1, 0)))
    concurrent = []
    hub.before_path_index = lambda: concurrent.append(publish(reordered, token=False, author="Ivan", api=hub))
    first = publish(direct, token=False, author="Ivan", api=hub)
    second, = concurrent
    assert first.query_id != second.query_id
    assert first.revision == hub.head
    rows = from_bytes(hub.snapshots[hub.head]["path-queries.parquet"]).to_pylist()
    assert {row["query_id"] for row in rows} == {first.query_id, second.query_id}
    for row in rows:
        staged = hub.snapshots[row["manifest_revision"]]
        manifest = json.loads(staged[row["manifest_path"]])
        artifact = from_bytes(staged[manifest["parts"][0]["path"]]).to_pylist()
        assert {tuple(row["targets"][record["target_index"]]) for record in artifact} == {(1, 0)}
    assert publish(direct, token=False, author="Ivan", api=hub) == first
    assert len(hub.commits) == 4


def test_incomplete_path_replacement_requires_a_longer_consistent_prefix(tmp_path):
    hub = LocalHub(tmp_path / "cache")
    before, shallower, conflicting, duplicate, missing, better = (
        tmp_path / name for name in ("before", "short", "changed", "duplicate", "missing", "better")
    )
    two = [[0], [1]]
    deeper = list(product(range(2), repeat=3))
    paths(before, status="INCOMPLETE", extra=4, completed=2, sequences=two + deeper[:1])
    paths(shallower, status="INCOMPLETE", extra=4, completed=1, sequences=two)
    paths(conflicting, status="INCOMPLETE", extra=4, completed=3, sequences=[[0]] + deeper)
    paths(duplicate, status="INCOMPLETE", extra=4, completed=3, sequences=[[0], [0]] + deeper)
    paths(missing, status="INCOMPLETE", extra=4, completed=3, sequences=two + deeper[1:])
    paths(better, status="INCOMPLETE", extra=4, completed=3, sequences=two + deeper)

    def publish_partial(output, **kwargs):
        return publish(output, token=False, author="Ivan", api=hub, include_incomplete=True, **kwargs)

    initial = publish_partial(before, case_id="keep-case-id")
    for rejected in (shallower, conflicting, duplicate, missing):
        with pytest.raises(ValueError, match="does not extend"):
            publish_partial(rejected)
        assert hub.head == initial.revision
    updated = publish_partial(better)
    assert updated.query_id == initial.query_id
    assert updated.case_id == initial.case_id
    row, = from_bytes(hub.snapshots[hub.head]["path-queries.parquet"]).to_pylist()
    assert row["completed_through_length"] == 3 and row["counts"] == [
        {"length": 1, "count": "2"}, {"length": 3, "count": "8"},
    ]
    with pytest.raises(ValueError, match="does not extend"):
        publish_partial(before)
    assert publish_partial(better) == updated
    assert len(hub.commits) == 4


def test_path_replacement_rechecks_newly_published_prefix_after_a_conflict(tmp_path):
    hub = LocalHub(tmp_path / "cache")
    before, competing, candidate = (tmp_path / name for name in ("before", "competing", "candidate"))
    shorter = [[0], [1]] + list(product(range(2), repeat=3))
    last = list(product(range(2), repeat=5))
    paths(before, status="INCOMPLETE", extra=4, completed=3, sequences=shorter)
    paths(competing, status="INCOMPLETE", extra=4, completed=4, sequences=shorter + last[:1])
    paths(candidate, extra=4, sequences=shorter + last[1:])

    def publish_partial(output):
        return publish(output, token=False, author="Ivan", api=hub, include_incomplete=True)

    initial = publish_partial(before)
    winners = []
    hub.before_path_index = lambda: winners.append(publish_partial(competing))
    with pytest.raises(ValueError, match="previously emitted routes are missing"):
        publish_partial(candidate)

    winner, = winners
    assert hub.head == winner.revision
    assert winner.query_id == initial.query_id
    row, = from_bytes(hub.snapshots[hub.head]["path-queries.parquet"]).to_pylist()
    assert row["completed_through_length"] == 4
    assert row["counts"][-1] == {"length": 5, "count": "1"}


def test_publish_reads_the_path_producer_receipt(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from multigpubfs import path_search

    monkeypatch.setattr(path_search, "_cuda_devices", lambda torch, devices: (torch.device("cpu"),))
    graph = GraphDefinition.permutation([[1, 0], [1, 0]], [0, 1])
    result = path_search.solve(
        graph, targets=[[1, 0], [0, 1]], output=tmp_path / "producer", devices=(0,),
        direction="bidirectional", goal_mode="window", extra_steps=1,
    )
    hub = LocalHub(tmp_path / "cache")
    publication = publish(result, token=False, author="Ivan", api=hub)
    row = from_bytes(hub.snapshots[publication.revision]["path-queries.parquet"]).to_pylist()[0]
    staged = hub.snapshots[row["manifest_revision"]]
    manifest = json.loads(staged[row["manifest_path"]])
    part = from_bytes(staged[manifest["parts"][0]["path"]]).to_pylist()
    assert {tuple(path["generator_ids"]): row["targets"][path["target_index"]] for path in part} == {
        (): [0, 1], (0,): [1, 0], (1,): [1, 0],
    }
