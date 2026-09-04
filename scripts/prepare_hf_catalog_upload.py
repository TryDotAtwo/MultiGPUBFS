#!/usr/bin/env python3
"""Map one verified BFS export into append-only Hugging Face catalog paths."""
import argparse
import json
import re
import shutil
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("CATALOG_OUTPUT_EXISTS")
    manifest = json.loads((args.package / "manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((args.package / "verification.json").read_text(encoding="utf-8"))
    run_id = str(manifest.get("run_id", ""))
    if manifest.get("schema") != "MGBFS_HF_DATASET_V1" or verification.get("status") != "PASS":
        raise ValueError("UNVERIFIED_PACKAGE")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise ValueError("RUN_ID_PATH")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("EMPTY_PACKAGE")
    for item in files:
        relative = PurePosixPath(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".parquet":
            raise ValueError("CATALOG_SOURCE_PATH")
        if not relative.parts or relative.parts[0] not in {"states", "layers", "runs"}:
            raise ValueError("CATALOG_TABLE")
        source = args.package.joinpath(*relative.parts)
        if not source.is_file():
            raise ValueError("CATALOG_SOURCE_MISSING")
        target = args.output.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifests = args.output / "manifests"
    checks = args.output / "verification"
    manifests.mkdir(parents=True, exist_ok=True)
    checks.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.package / "manifest.json", manifests / f"{run_id}.json")
    shutil.copy2(args.package / "verification.json", checks / f"{run_id}.json")


if __name__ == "__main__":
    main()
