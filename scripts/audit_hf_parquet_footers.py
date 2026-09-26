"""Read every immutable HF Parquet footer without downloading state payloads."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import time

import pyarrow.parquet as pq


PART = re.compile(r"^states/.+-rank-(\d{5})-part-(\d{8})\.parquet$")


def audit_manifest(manifest, opener, *, workers=4, max_rows=1_000_000, progress=None):
    """Return footer-derived counts; raise unless every declared part agrees."""
    world = manifest["world_size"]
    files = manifest["files"]
    if not 1 <= world <= 128 or not files or workers < 1 or max_rows < 1:
        raise ValueError("FOOTER_AUDIT_CONFIG")
    if manifest.get("status", "COMPLETE") != "COMPLETE":
        raise ValueError("FOOTER_RUN_INCOMPLETE")

    def read_one(item):
        path = item["path"]
        match = PART.fullmatch(path)
        if match is None:
            raise ValueError("FOOTER_PATH")
        rank, part = (int(value) for value in match.groups())
        if rank >= world:
            raise ValueError("FOOTER_RANK")
        for attempt in range(5):
            try:
                with opener(path) as source:
                    footer = pq.read_metadata(source)
                fields = tuple((field.name, str(field.type), field.nullable)
                               for field in footer.schema.to_arrow_schema())
                return rank, part, footer.num_rows, fields
            except Exception as error:
                if attempt == 4:
                    raise
                status = getattr(getattr(error, "response", None), "status_code", None)
                time.sleep(20 * (attempt + 1) if status == 429 else 0.5 * (attempt + 1))

    parts = defaultdict(set)
    rows_by_rank = defaultdict(int)
    expected_fields = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # A bounded task window stops quickly on rate-limit or corrupt footer.
        window = workers * 4
        for offset in range(0, len(files), window):
            for rank, part, rows, fields in pool.map(read_one, files[offset:offset + window]):
                if part in parts[rank]:
                    raise ValueError("FOOTER_DUPLICATE_PART")
                if not 1 <= rows <= max_rows:
                    raise ValueError("FOOTER_ROWS_PER_PART")
                if expected_fields is not None and fields != expected_fields:
                    raise ValueError("FOOTER_SCHEMA")
                expected_fields = fields
                parts[rank].add(part)
                rows_by_rank[rank] += rows
            if progress is not None:
                progress(min(offset + window, len(files)), len(files))

    for rank in range(world):
        if parts[rank] != set(range(len(parts[rank]))):
            raise ValueError("FOOTER_PART_SEQUENCE")
    total = sum(rows_by_rank.values())
    if total != manifest["total_unique_states"] or total != sum(manifest["layer_counts"]):
        raise ValueError("FOOTER_ROW_TOTAL")
    return {
        "file_count": len(files),
        "rows_total": total,
        "rows_by_rank": {str(rank): rows_by_rank[rank] for rank in range(world)},
        "parts_by_rank": {str(rank): len(parts[rank]) for rank in range(world)},
        "schema_fields": [list(field) for field in expected_fields],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("FOOTER_REVISION_NOT_IMMUTABLE")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN_MISSING")
    from huggingface_hub import HfFileSystem, hf_hub_download

    manifest_path = hf_hub_download(repo_id=args.repo_id, repo_type="dataset",
                                    revision=args.revision, filename=args.manifest,
                                    token=token)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    fs = HfFileSystem(token=token)
    start = time.monotonic()
    result = audit_manifest(
        manifest,
        lambda path: fs.open(f"datasets/{args.repo_id}@{args.revision}/{path}", "rb"),
        workers=args.workers, max_rows=args.max_rows,
        progress=lambda done, total: print(f"FOOTERS {done}/{total}", flush=True),
    )
    result.update(status="VERIFIED_FOOTERS", repo_id=args.repo_id,
                  revision=args.revision, run_id=manifest["run_id"],
                  elapsed_seconds=round(time.monotonic() - start, 3))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
