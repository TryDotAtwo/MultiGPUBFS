"""Verify downloaded bounded-owner evidence; not full BFS certification."""
import argparse
import json
import re
from pathlib import Path

def verify(path, source):
    s = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if not re.fullmatch(r"[0-9a-f]{40}", source) or s["source"] != source or s["status"] != "COMPLETE":
        raise ValueError("Incomplete or different source")
    gpus = s["gpus"]
    if len(gpus) != 2 or {g["index"] for g in gpus} != {0, 1} or len({g["uuid"] for g in gpus}) != 2:
        raise ValueError("Two distinct physical GPUs required")
    if any(g["name"] not in ("Tesla T4", "NVIDIA Tesla T4", "NVIDIA T4") for g in gpus):
        raise ValueError("Wrong hardware")
    uuids = {g["index"]: g["uuid"] for g in gpus}
    expected = {(g, t) for g in (0, 1) for t in ("plain", "memcheck", "racecheck", "initcheck", "synccheck")}
    seen = set()
    for check in s["checks"]:
        key = (check["gpu"], check["tool"])
        if key not in expected or key in seen or check["uuid"] != uuids[key[0]]:
            raise ValueError("Duplicate or misbound check")
        seen.add(key)
        name = f"gpu{key[0]}-{key[1]}.log"
        if check["log"] != name:
            raise ValueError("Log binding")
        log = (path / name).read_text(encoding="utf-8", errors="replace")
        if not re.search(r"^BOUNDED_OWNER_PASS\s*$", log, re.M):
            raise ValueError("No test completion")
        if key[1] == "racecheck":
            summaries = re.findall(r"RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)", log)
            if not summaries or any(x != ("0", "0", "0") for x in summaries):
                raise ValueError("Racecheck failure/incomplete")
        elif key[1] != "plain":
            summaries = re.findall(r"ERROR SUMMARY: (\d+) errors", log)
            if not summaries or any(x != "0" for x in summaries):
                raise ValueError("Sanitizer failure/incomplete")
    if seen != expected:
        raise ValueError("Incomplete check matrix")
    return "VERIFIED_BOUNDED_OWNER_GATE 10/10"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    print(verify(args.path, args.source))
