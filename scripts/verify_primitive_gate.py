"""Validate saved primitive-gate evidence, not production multi-rank BFS."""
import argparse
import json
import re
from pathlib import Path

def verify_summary(summary, source_commit):
    if not re.fullmatch(r"[0-9a-f]{40}",source_commit):
        raise ValueError("IMMUTABLE_SOURCE_REQUIRED")
    if summary.get("status")!="PASS_PRIMITIVE_GATE" or summary.get("source_commit")!=source_commit:
        raise ValueError("INCOMPLETE_OR_WRONG_SOURCE")
    gpus=summary["gpus"]
    if len(gpus)!=2 or sorted(g["index"] for g in gpus)!=[0,1] or len({g["uuid"] for g in gpus})!=2:
        raise ValueError("TWO_DISTINCT_DEVICES_REQUIRED")
    if any(g["name"] not in ("Tesla T4","NVIDIA Tesla T4","NVIDIA T4") for g in gpus):
        raise ValueError("T4_REQUIRED")
    tests=("generate","hash","route","owner","pipeline","materialize","dense_device","ping_pong")
    tools=("plain","memcheck","racecheck","initcheck","synccheck")
    expected={(g["uuid"],test,tool) for g in gpus for test in tests for tool in tools}
    seen=set()
    indices={g["uuid"]:g["index"] for g in gpus}
    entries=[]
    for row in summary["results"]:
        key=(row["gpu"],row["test"],row["tool"])
        if key in seen or key not in expected or row["status"]!="PASS":
            raise ValueError("DUPLICATE_UNKNOWN_OR_FAILED_RESULT")
        seen.add(key)
        entries.append((f"gpu{indices[key[0]]}-{key[1]}-{key[2]}.log",key[2]))
    if seen!=expected:
        raise ValueError("MISSING_RESULT")
    return sorted(entries)

def verify_log(text, tool):
    if not re.search(r"test result: ok\. [1-9][0-9]* passed; 0 failed; 0 ignored;",text) or "test result: FAILED" in text:
        raise ValueError("RUST_TESTS_INCOMPLETE")
    if tool=="plain": return
    if tool not in ("memcheck","racecheck","initcheck","synccheck"):
        raise ValueError("UNKNOWN_SANITIZER")
    errors=re.findall(r"ERROR SUMMARY: ([0-9]+) errors",text)
    races=re.findall(r"RACECHECK SUMMARY: ([0-9]+) hazards displayed \(([0-9]+) errors, ([0-9]+) warnings\)",text)
    if any(int(x)!=0 for x in errors) or any(any(int(x)!=0 for x in row) for row in races):
        raise ValueError("SANITIZER_ERRORS")
    if tool=="racecheck" and not races or tool!="racecheck" and not errors:
        raise ValueError("SANITIZER_SUMMARY_MISSING")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("directory",type=Path)
    p.add_argument("--source",required=True)
    args=p.parse_args()
    summary=json.loads((args.directory/"summary.json").read_text(encoding="utf-8"))
    entries=verify_summary(summary,args.source)
    for name,tool in entries:
        verify_log((args.directory/name).read_text(encoding="utf-8"),tool)
    query=(args.directory/"allocation-queries.log").read_text(encoding="utf-8")
    if "100% tests passed, 0 tests failed out of 1" not in query:
        raise ValueError("QUERY_CTEST_INCOMPLETE")
    print(json.dumps(dict(status="VERIFIED_PRIMITIVE_GATE",source=args.source,combinations=len(entries))))

if __name__=="__main__":
    main()
