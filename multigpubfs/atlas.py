"""Publish committed BFS evidence to the Graph BFS Atlas."""

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


DEFAULT_REPO = "TryDotAtwo/graph-bfs-atlas"
_PATH_INDEX = "path-queries.parquet"
_FAMILY_CSV = {"lrx": "lrx-layers.csv", "rapaport_m2": "rapaport-m2-layers.csv"}


@dataclass(frozen=True)
class Publication:
    case_id: str
    revision: str
    commit_url: str
    query_id: str | None = None


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _file_digest(file):
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex(value, digits=64):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{%d}" % digits, value) is None:
        raise ValueError(f"expected a {digits}-digit lowercase hexadecimal digest")
    return value


def _repo_path(value):
    path = PurePosixPath(value)
    if (path.as_posix() != value or path.is_absolute() or value == "."
            or ".." in path.parts or "\\" in value or ":" in value):
        raise ValueError(f"invalid artifact path {value!r}")
    return value


def _local_payload(output, path):
    file = (output / _repo_path(path)).resolve()
    if not file.is_relative_to(output):
        raise ValueError(f"artifact {path!r} escapes the run directory")
    return file


@dataclass(frozen=True)
class _File:
    payload: bytes | Path
    sha256: str
    size: int

    @classmethod
    def from_payload(cls, payload):
        return cls(payload, _digest(payload), len(payload))


class _Hub:
    def __init__(self, api, repo, attempts):
        self.api = api
        self.repo = repo
        self.attempts = attempts

    def info(self, revision="main", repo=None):
        return self.api.repo_info(repo_id=repo or self.repo, repo_type="dataset", revision=revision)

    def head(self):
        return _hex(self.info().sha, 40)

    def get(self, path, revision, repo=None):
        return Path(self.api.hf_hub_download(
            repo_id=repo or self.repo, filename=path, revision=revision, repo_type="dataset",
        ))

    def files(self, paths, revision, repo=None):
        files = {}
        paths = list(paths)
        for offset in range(0, len(paths), 500):
            reply = self.api.get_paths_info(
                repo_id=repo or self.repo, paths=paths[offset:offset + 500],
                revision=revision, repo_type="dataset",
            )
            files.update((item.path, item) for item in reply if hasattr(item, "size"))
        return files

    def matches(self, path, item, size, sha256, revision, repo=None, payload=None):
        if item.size != size:
            return False
        if item.lfs:
            return item.lfs.sha256 == sha256
        if isinstance(payload, bytes):
            git_blob = hashlib.sha1(f"blob {size}\0".encode() + payload).hexdigest()
            return item.blob_id == git_blob
        return _file_digest(self.get(path, revision, repo)) == sha256

    def commit(self, prepare, message):
        from huggingface_hub import CommitOperationAdd

        parent = self.head()
        for attempt in range(self.attempts):
            files = prepare(parent)
            if not files:
                return parent
            operations = []
            for path, content in files.items():
                op = CommitOperationAdd(path_in_repo=path, path_or_fileobj=content.payload)
                if isinstance(content.payload, Path) and (
                    op.upload_info.sha256.hex() != content.sha256 or op.upload_info.size != content.size
                ):
                    raise ValueError(f"path artifact checksum or size differs from the receipt: {path}")
                operations.append(op)
            try:
                receipt = self.api.create_commit(
                    repo_id=self.repo, repo_type="dataset", revision="main",
                    parent_commit=parent, commit_message=message, operations=operations,
                )
            except Exception:
                current = self.head()
                if current == parent:
                    raise
                parent = current
                if attempt + 1 == self.attempts:
                    if not prepare(parent):
                        return parent
                    raise
                continue
            revision = _hex(receipt.oid, 40)
            self.check(files, revision)
            return revision
        raise AssertionError("commit attempts must be positive")

    def check(self, files, revision):
        found = self.files(files, revision)
        for path, content in files.items():
            if path not in found or not self.matches(
                path, found[path], content.size, content.sha256, revision, payload=content.payload,
            ):
                raise RuntimeError(f"published artifact failed verification at {revision}: {path}")

    def stage(self, payloads):
        def prepare(revision):
            found = self.files(payloads, revision)
            for path, item in found.items():
                file = payloads[path]
                if not self.matches(path, item, file.size, file.sha256, revision, payload=file.payload):
                    raise ValueError(f"immutable Atlas artifact has different content: {path}")
            return {path: file for path, file in payloads.items() if path not in found}

        return self.commit(prepare, "Stage immutable MultiGPUBFS evidence")


def _definition(result, output):
    graph = getattr(result, "graph", None)
    if graph is None and (output / "graph.json").is_file():
        from .graph_definition import GraphDefinition

        graph = GraphDefinition.from_dict(json.loads((output / "graph.json").read_bytes()))
    return graph


def _historical_parameters(family, start, n, r, graph_digest):
    from .graph_definition import GraphDefinition

    degree = len(start)
    if family not in _FAMILY_CSV or degree < 2 or n not in (None, degree):
        return None
    repeat = degree - len(set(start)) + 1
    last = degree - repeat
    if r not in (None, repeat) or start != list(range(last + 1)) + [last] * (repeat - 1):
        return None

    swap = [1, 0] + list(range(2, degree))
    if family == "lrx":
        generators = [list(range(1, degree)) + [0], [degree - 1] + list(range(degree - 1)), swap]
    else:
        even, odd = list(range(degree)), list(range(degree))
        for permutation, offset in ((even, 0), (odd, 1)):
            for i in range(offset, degree - 1, 2):
                permutation[i], permutation[i + 1] = permutation[i + 1], permutation[i]
        generators = [swap, even, odd]
    if GraphDefinition.permutation(generators, start).digest() == graph_digest:
        return degree, repeat
    return None


def _graph_fields(definition, report, graph_family, n, r, start_state, case_id):
    from .specialized import match_lrx

    data = definition.to_dict() if definition is not None else {}
    action = data.get("action", {})
    lrx = match_lrx(definition) if definition is not None else None
    family = graph_family or report.get("graph_family") or (
        "lrx" if lrx else data.get("name") or action.get("kind")
    )
    if isinstance(family, str):
        name = family.lower().replace(" ", "")
        if re.fullmatch(r"lrx(?:-\d+)?", name):
            family = "lrx"
        elif re.fullmatch(r"rapaport[_-]?m2(?:-\d+)?", name):
            family = "rapaport_m2"
    start = start_state if start_state is not None else report.get("start_state")
    if start is None:
        start = data.get("start")
    if not isinstance(family, str) or not family.strip() or start is None:
        raise ValueError("graph_family and start_state must be recorded by the run or provided")
    start = _vector(start)
    graph_digest = report.get("graph_sha256", report.get("graph_digest"))
    if definition is not None:
        if start != data["start"]:
            raise ValueError("Atlas start must match the recorded graph definition")
        digest = definition.digest()
        if graph_digest is not None and digest != graph_digest:
            raise ValueError("recorded graph digest differs from the graph definition")
        graph_digest = digest
        generators = report.get("generators_sha256")
        if generators is not None and _digest(_json(action)) != generators:
            raise ValueError("recorded generator digest differs from the graph definition")
    n = n if n is not None else report.get("n", action.get("degree"))
    r = r if r is not None else report.get("r", lrx["r"] if lrx else None)
    for value in (n, r):
        if value is not None:
            _natural(value)
    historical = _historical_parameters(family, start, n, r, graph_digest)
    if historical is not None:
        n, r = historical
    case = case_id
    if case is None:
        case = ":".join(
            [family] + [f"{label}{value}" for label, value in (("n", n), ("r", r)) if value is not None]
        )
        if historical is None:
            case = f"{case}:graph-{_hex(graph_digest)}"
    if not isinstance(case, str) or not case.strip():
        raise ValueError("case_id must be nonempty")
    return {"case_id": case, "graph_family": family, "n": n, "r": r, "start_state": start}


def _vector(value):
    if not isinstance(value, (tuple, list)) or not value or any(
        type(item) is not int or not -(1 << 63) <= item < (1 << 63) for item in value
    ):
        raise ValueError("state vectors must be nonempty sequences of signed int64 values")
    return list(value)


def _natural(value):
    if type(value) is not int or value < 0:
        raise ValueError("expected a nonnegative integer")
    return value


def _traversal(report, states, graph):
    layer_sizes = report["layer_sizes"]
    if not layer_sizes or layer_sizes[0] != 1 or any(
        type(size) is not int or size <= 0 for size in layer_sizes
    ):
        raise ValueError("traversal must have positive layer sizes beginning with a single root")
    previous = 0
    layers = []
    for depth, size in enumerate(layer_sizes):
        previous += size
        layers.append({"depth": depth, "new_unique_states": size, "cumulative_states": previous})
    if states.get("schema") != 2 or states.get("state_encoding") != "signed_int64_vectors":
        raise ValueError("unsupported traversal state receipt schema")
    sample_limit = states.get("current_sample_limit", states.get("current_sample_global_limit"))
    if sample_limit is not None:
        _natural(sample_limit)
    seed = report.get("seed")
    retained = {}
    for role in ("current", "previous_small"):
        for index, value in enumerate(states[role]):
            vector = _vector(value)
            if len(vector) != len(graph["start_state"]):
                raise ValueError("retained vector size differs from the recorded start")
            item = retained.setdefault(tuple(vector), {
                "state": vector, "seeds": [] if seed is None else [seed],
                "retention_roles": [], "evidence": [],
            })
            if role not in item["retention_roles"]:
                item["retention_roles"].append(role)
            item["evidence"].append({
                "seed": seed, "retention_role": role, "source_index": index,
                "source_schema": states["schema"], "current_sample_limit": sample_limit,
                "archive_member": None,
            })
    return {
        **graph, "status": report["status"], "max_distance_from_canonical_start": len(layers) - 1,
        "observed_states": previous, "evidence_kind": "source_reported_multigpubfs",
        "retained_unique_states": len(retained), "layers": layers, "states": list(retained.values()),
    }


def _parquet(table, row_group_size=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, row_group_size=row_group_size)
    return _File.from_payload(sink.getvalue().to_pybytes())


def _updated_family_csv(hub, revision, row):
    path = _FAMILY_CSV.get(row["graph_family"])
    if path is None or path not in hub.files([path], revision):
        return {}
    existing = hub.get(path, revision).read_bytes()
    if not existing.endswith(b"\n"):
        raise ValueError(f"expected a newline-terminated layers CSV: {path}")
    fields = ("graph_family", "n", "r", "status", "depth", "new_unique_states",
              "cumulative_states", "case_id", "start_state")
    view = io.StringIO(newline="")
    writer = csv.DictWriter(view, fields, lineterminator="\r\n" if b"\r\n" in existing else "\n")
    shared = {key: row[key] for key in fields if key in row}
    shared["start_state"] = " ".join(map(str, row["start_state"]))
    writer.writerows({**shared, **layer} for layer in row["layers"])
    return {path: _File.from_payload(existing + view.getvalue().encode("utf-8"))}


def _publish_traversal(hub, output, raw, report, graph, definition, author):
    import pyarrow as pa
    import pyarrow.parquet as pq

    states = (output / "states.json").read_bytes()
    if _digest(states) != _hex(report.get("states_sha256")):
        raise ValueError("traversal state receipt SHA-256 mismatch")
    row = _traversal(report, json.loads(states), graph)
    provenance = _json({"schema": 1, "author": author, "graph": graph})
    files = {"report.json": raw, "states.json": states, "provenance.json": provenance}
    if definition is not None:
        files["graph.json"] = definition.to_json().encode()
    content = _digest(_json({name: _digest(value) for name, value in files.items()}))
    prefix = f"artifacts/multigpubfs/traversals/{content}"
    staged = hub.stage({f"{prefix}/{name}": _File.from_payload(value) for name, value in files.items()})
    row.update(source_repo=hub.repo, source_revision=staged, evidence_prefix=prefix)
    for state in row["states"]:
        for evidence in state["evidence"]:
            evidence["source_file"] = f"{prefix}/states.json"
    base_case_id = graph["case_id"]

    def prepare(revision):
        table = pq.read_table(hub.get("atlas.parquet", revision))
        if set(row) != set(table.column_names):
            raise ValueError("Atlas traversal schema changed")
        cases = dict(zip(table["case_id"].to_pylist(), table["evidence_prefix"].to_pylist()))
        existing_case = next((case for case, evidence in cases.items() if evidence == prefix), None)
        if existing_case is not None:
            row["case_id"] = existing_case
            return {}
        row["case_id"] = f"{base_case_id}:run-{content}" if base_case_id in cases else base_case_id
        if row["case_id"] in cases:
            raise ValueError(f"Atlas already contains case {row['case_id']} with different evidence")
        addition = pa.Table.from_pylist([row], schema=table.schema)
        sources = json.loads(hub.get("sources.json", revision).read_bytes())
        sources.extend(
            {"repo": hub.repo, "revision": staged, "path": f"{prefix}/{name}", "sha256": _digest(value)}
            for name, value in files.items()
        )
        return {
            "atlas.parquet": _parquet(pa.concat_tables([table, addition]), row_group_size=1),
            "sources.json": _File.from_payload(json.dumps(sources, indent=2, ensure_ascii=False).encode() + b"\n"),
            **_updated_family_csv(hub, revision, row),
        }

    revision = hub.commit(prepare, "Index MultiGPUBFS traversal")
    return _publication(hub, row["case_id"], revision)


def _query(report, graph):
    targets = [_vector(value) for value in report["targets"]]
    if not targets or len({tuple(value) for value in targets}) != len(targets) or any(
        len(target) != len(graph["start_state"]) for target in targets
    ):
        raise ValueError("path targets must be unique and have the same size as the start")
    k = _natural(report["k"])
    d_min, max_length = report["d_min"], report["max_length"]
    if d_min is not None:
        _natural(d_min)
    if max_length is not None:
        _natural(max_length)
    counts = []
    previous = -1
    for entry in report["counts"]:
        length, count = _natural(entry["length"]), _natural(entry["count"])
        if length <= previous or max_length is None or length > max_length or (
            count and (d_min is None or length < d_min)
        ):
            raise ValueError("path counts are inconsistent with the recorded length bounds")
        counts.append({"length": length, "count": str(count)})
        previous = length
    status = report["status"]
    goal_mode = report.get("goal_mode", "window")
    if goal_mode not in {"nearest", "window"} or (goal_mode == "nearest" and k):
        raise ValueError("invalid path goal mode or cutoff")
    completed = report.get("completed_through_length", max_length if status == "COMPLETE" else None)
    if completed is not None and (max_length is None or _natural(completed) > max_length):
        raise ValueError("committed path length exceeds the recorded cutoff")
    if "path_count" in report and _natural(report["path_count"]) != sum(int(row["count"]) for row in counts):
        raise ValueError("path count does not match the sum of per-length results")
    if (status == "COMPLETE" and (d_min is None or max_length != d_min + k
                                 or not any(row["length"] == d_min and int(row["count"]) for row in counts)
                                 or ("completed_through_length" in report and completed != max_length)
                                 or (goal_mode == "nearest" and sum(int(row["count"]) for row in counts) != 1))
            or status == "UNREACHABLE" and (d_min is not None or max_length is not None or counts)):
        raise ValueError("path status and counts are inconsistent")
    query = {
        "cutoff": "one_nearest_witness" if goal_mode == "nearest" else "global_d_min_plus_k",
        "goal_mode": goal_mode, "graph_sha256": _hex(report["graph_sha256"]),
        "generators_sha256": _hex(report["generators_sha256"]), "start_state": graph["start_state"],
        # Path records address the ordered list by target_index.
        "targets": targets, "k": k,
    }
    return {
        **graph, **query, "query_id": _digest(_json(query)),
        "status": status, "d_min": d_min, "max_length": max_length, "counts": counts,
        "completed_through_length": completed,
    }


def _part_count(query):
    return sum(int(entry["count"]) for entry in query["counts"])


def _extends_path_coverage(old, new):
    if old["d_min"] is not None and new["d_min"] != old["d_min"]:
        return False
    previous_end = old["completed_through_length"]
    completed = new["completed_through_length"]
    before = {entry["length"]: int(entry["count"]) for entry in old["counts"] if int(entry["count"])}
    after = {entry["length"]: int(entry["count"]) for entry in new["counts"] if int(entry["count"])}
    if new["status"] == "UNREACHABLE":
        return previous_end is None and not before
    if completed is None or (previous_end is not None and completed < previous_end):
        return False
    if completed == previous_end and new["status"] != "COMPLETE":
        return False
    if previous_end is not None and (
        {length: count for length, count in before.items() if length <= previous_end}
        != {length: count for length, count in after.items() if length <= previous_end}
    ):
        return False
    # An unfinished length can contain a different subset in each run. Its
    # emitted walks are covered once the replacement completes that length.
    return all(length <= completed and after.get(length, 0) >= count for length, count in before.items())


def _local_manifest(hub, output, raw, report, query, definition, author):
    import pyarrow as pa
    import pyarrow.parquet as pq

    file = report.get("path_file")
    if definition is None:
        raise ValueError("path publication requires graph.json to interpret generator IDs")
    payloads = {}
    parts = []
    if file is not None:
        part = _File(
            _local_payload(output, file), _hex(report.get("path_sha256")), _natural(report.get("path_bytes")),
        )
        if part.payload.stat().st_size != part.size:
            raise ValueError("path artifact checksum or size differs from the receipt")
        with pq.ParquetFile(part.payload) as parquet:
            schema = parquet.schema_arrow
            types = {"target_index": pa.int64(), "length": pa.int64(), "generator_ids": pa.list_(pa.int32())}
            if any(name not in schema.names or schema.field(name).type != kind for name, kind in types.items()):
                raise ValueError("path artifact does not have the generator-ID sequence schema")
            if parquet.metadata.num_rows != _part_count(query):
                raise ValueError("path artifact row count differs from the committed receipt")
        name = "paths.parquet"
        payloads[name] = part
        parts.append({"path": name, "sha256": part.sha256, "bytes": part.size})
    elif _part_count(query):
        raise ValueError("committed path counts require a path artifact or remote manifest")
    payloads["report.json"] = _File.from_payload(raw)
    payloads["graph.json"] = _File.from_payload(definition.to_json().encode())
    body = {"schema": 1, "query": query, "author": author, "parts": parts,
            "report_sha256": payloads["report.json"].sha256,
            "graph_sha256": payloads["graph.json"].sha256}
    prefix = f"artifacts/multigpubfs/paths/{_digest(_json(body))}"
    body["parts"] = [{**part, "path": f"{prefix}/{part['path']}"} for part in parts]
    for name in ("graph", "report"):
        body[f"{name}_path"] = f"{prefix}/{name}.json"
    payloads["manifest.json"] = _File.from_payload(_json(body))
    files = {f"{prefix}/{name}": content for name, content in payloads.items()}
    pointer = {"repo_id": hub.repo, "path": f"{prefix}/manifest.json",
               "sha256": payloads["manifest.json"].sha256}
    return pointer, body["parts"], files


def _check_manifest(hub, pointer, query):
    from huggingface_hub.utils import validate_repo_id

    from .graph_definition import GraphDefinition

    repo = pointer["repo_id"]
    validate_repo_id(repo)
    revision, path, sha256 = _hex(pointer["revision"], 40), _repo_path(pointer["path"]), _hex(pointer["sha256"])
    info = hub.info(revision, repo)
    if info.sha != revision:
        raise ValueError("path manifest must reference an immutable dataset commit")
    if repo != hub.repo and (getattr(info, "private", False) or getattr(info, "gated", False)):
        raise ValueError("external path manifest must be publicly readable")
    raw = hub.get(path, revision, repo).read_bytes()
    if _digest(raw) != sha256:
        raise ValueError("path manifest SHA-256 does not match its pointer")
    body = json.loads(raw)
    mandatory = ("query_id", "status", "targets", "d_min", "max_length", "counts", "completed_through_length")
    if body.get("schema") != 1 or any(body["query"].get(key) != query[key] for key in mandatory):
        raise ValueError("path manifest does not describe the supplied query result")
    parts = body["parts"]
    paths = [_repo_path(part["path"]) for part in parts]
    if len(set(paths)) != len(parts) or path in paths or (_part_count(query) and not parts):
        raise ValueError("path manifest has missing or duplicate parts")
    found = hub.files(paths, revision, repo)
    for part in parts:
        artifact = part["path"]
        if artifact not in found or not hub.matches(
            artifact, found[artifact], _natural(part["bytes"]), _hex(part["sha256"]), revision, repo,
        ):
            raise ValueError(f"path manifest part failed verification: {artifact}")
    graph_path = _repo_path(body["graph_path"])
    raw_graph = hub.get(graph_path, revision, repo).read_bytes()
    if _digest(raw_graph) != _hex(body["graph_sha256"]):
        raise ValueError("path manifest graph failed verification")
    definition = GraphDefinition.from_dict(json.loads(raw_graph))
    if (definition.digest() != query["graph_sha256"] or definition.start != query["start_state"]
            or _digest(_json(definition.action)) != query["generators_sha256"]):
        raise ValueError("path manifest graph does not match the query")
    return {key: pointer[key] for key in ("repo_id", "revision", "path", "sha256")}, parts


def _path_schema(pa):
    vector = pa.list_(pa.int64())
    strings = ("case_id", "graph_family", "query_id", "status", "graph_sha256", "generators_sha256",
               "goal_mode", "cutoff", "manifest_repo", "manifest_revision", "manifest_path", "manifest_sha256",
               "result_sha256", "author")
    return pa.schema(
        [(name, pa.string()) for name in strings]
        + [(name, pa.int64()) for name in ("n", "r", "k", "d_min", "max_length", "completed_through_length")]
        + [("start_state", vector), ("targets", pa.list_(vector)),
           ("counts", pa.list_(pa.struct([("length", pa.int64()), ("count", pa.string())])))]
    )


def _path_card(raw):
    from huggingface_hub import DatasetCard

    card = DatasetCard(raw.decode("utf-8"))
    configs = card.data.get("configs", [])
    query_config = {"config_name": "path_queries", "data_files": [{"split": "train", "path": _PATH_INDEX}]}
    for config in configs:
        if config.get("config_name") == "path_queries":
            if config.get("data_files") != query_config["data_files"]:
                raise ValueError("the existing path_queries configuration has different data files")
            return None
    default = next((config for config in configs if config["config_name"] == "default"), None)
    if default is None:
        raise ValueError("dataset card is missing the existing default Atlas configuration")
    default["default"] = True
    configs.append(query_config)
    card.data["configs"] = configs
    card.content += (
        "\n\n## Path queries\n\n"
        "The path_queries configuration indexes walks identified by sequences of generator IDs. "
        "Window queries include every qualifying walk. Nearest queries store one shortest witness. "
        "The manifest columns pin the parts and their checksums to an immutable Hub commit. "
        "Count values are decimal strings so the index can represent arbitrary-size counts.\n"
    )
    return _File.from_payload(str(card).encode("utf-8"))


def _publish_paths(hub, output, raw, report, graph, definition, author, manifest):
    import pyarrow as pa
    import pyarrow.parquet as pq

    query = _query(report, graph)
    staged_files = None
    if manifest is None:
        manifest, parts, staged_files = _local_manifest(hub, output, raw, report, query, definition, author)
    else:
        manifest, parts = _check_manifest(hub, manifest, query)
    fingerprint = _digest(_json({
        "query": {key: query[key] for key in (
            "query_id", "status", "targets", "d_min", "max_length", "counts", "completed_through_length",
        )},
        "parts": [(part["sha256"], part["bytes"]) for part in parts],
    }))
    row = {
        **query, "author": author, "result_sha256": fingerprint,
        "manifest_repo": manifest["repo_id"], "manifest_revision": manifest.get("revision"),
        "manifest_path": manifest["path"], "manifest_sha256": manifest["sha256"],
    }

    def prepare(revision, inspect_only=False):
        card = _path_card(hub.get("README.md", revision).read_bytes())
        files = {"README.md": card} if card is not None else {}
        if _PATH_INDEX in hub.files([_PATH_INDEX], revision):
            table = pq.read_table(hub.get(_PATH_INDEX, revision))
            rows = table.to_pylist()
            schema = table.schema
        else:
            rows = []
            schema = _path_schema(pa)
        matches = [i for i, existing in enumerate(rows) if existing["query_id"] == row["query_id"]]
        if len(matches) > 1:
            raise ValueError("Atlas path query index contains duplicate query IDs")
        if matches:
            i = matches[0]
            old = rows[i]
            if old["result_sha256"] == fingerprint:
                row["case_id"] = old["case_id"]
                return files
            if old["status"] != "INCOMPLETE":
                raise ValueError("Atlas already contains a different terminal result for this path query")
            if not _extends_path_coverage(old, row):
                raise ValueError("path result does not extend the published completed coverage")
            row["case_id"] = old["case_id"]
            rows[i] = row
        else:
            rows.append(row)
        if inspect_only:
            return None
        files[_PATH_INDEX] = _parquet(pa.Table.from_pylist(rows, schema=schema), row_group_size=1)
        return files

    if staged_files is not None:
        head = hub.head()
        pending = prepare(head, inspect_only=True)
        if pending == {}:
            return _publication(hub, row["case_id"], head, query["query_id"])
        if pending is None:
            row["manifest_revision"] = hub.stage(staged_files)
    revision = hub.commit(prepare, "Index MultiGPUBFS path query")
    return _publication(hub, row["case_id"], revision, query["query_id"])


def _publication(hub, case_id, revision, query_id=None):
    endpoint = getattr(hub.api, "endpoint", "https://huggingface.co").rstrip("/")
    return Publication(case_id, revision, f"{endpoint}/datasets/{hub.repo}/commit/{revision}", query_id)


def publish(result, *, token, author, repo_id=DEFAULT_REPO, include_incomplete=False,
            manifest=None, graph_family=None, n=None, r=None, start_state=None, case_id=None,
            max_attempts=5, api=None):
    """Stage the run's immutable evidence, then atomically index it on Atlas.

    A path result can reference an externally staged schema-1 dataset manifest
    by supplying its repo_id, revision (commit SHA), path and sha256.
    Path query identity includes target order, since artifact rows use target indices.
    Publishing a different partial result requires additional completed coverage.
    A legacy run directory can supply its missing graph metadata by keyword.
    The default case ID includes the exact graph digest for custom actions or
    starts. Canonical Atlas LRX and Rapaport M2 cases retain their historical IDs.
    """
    from huggingface_hub import HfApi

    if not isinstance(author, str) or not author.strip():
        raise ValueError("author must be nonempty")
    if _natural(max_attempts) == 0:
        raise ValueError("max_attempts must be positive")
    output = Path(getattr(result, "output", result)).resolve()
    raw = (output / "report.json").read_bytes()
    report = json.loads(raw)
    paths = report.get("mode") == "paths"
    status = report.get("status")
    accepted = {"COMPLETE", "UNREACHABLE"} if paths else {"COMPLETE"}
    if include_incomplete:
        accepted.add("INCOMPLETE")
    if status not in accepted:
        raise ValueError(f"run status {status!r} is not eligible for Atlas publication")
    if manifest is not None and not paths:
        raise ValueError("remote manifests apply to path results only")
    definition = _definition(result, output)
    graph = _graph_fields(definition, report, graph_family, n, r, start_state, case_id)
    hub = _Hub(api if api is not None else HfApi(token=token), repo_id, max_attempts)
    if paths:
        return _publish_paths(hub, output, raw, report, graph, definition, author.strip(), manifest)
    return _publish_traversal(hub, output, raw, report, graph, definition, author.strip())
