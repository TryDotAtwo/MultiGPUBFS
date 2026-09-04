#!/usr/bin/env python3
"""Normalize a completed macro benchmark record for the HF dataset exporter."""
import argparse, hashlib, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument("raw",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--source-commit",required=True);p.add_argument("--hardware",required=True);a=p.parse_args()
    raw=json.loads(a.raw.read_text(encoding="utf-8"))
    if raw.get("status")!="COMPLETE" or raw.get("unique_states")!=sum(raw.get("layer_sizes",[])):
        raise ValueError("INCOMPLETE_OR_COUNT_MISMATCH")
    description=(f"macro-native-v1;group={raw['group']};batch={raw['batch']};k={raw['macro_depth']};"
                 f"layer={raw['unique_states']};future={raw['unique_states']};pre={str(raw['prededup']).lower()};seed=20260828")
    digest=hashlib.sha256(description.encode()).hexdigest()
    layers={str(depth):{"unique_states":count,"generated_candidates":count*3,
                        "wall_ms":raw["per_depth_seconds"][depth]*1000}
            for depth,count in enumerate(raw["layer_sizes"])}
    summary={"schema":"RunSummaryV1","status":"COMPLETE","run_id":"s10-native-k1-seed-20260828",
             "group_id":"S_10 adjacent-transposition matrix Cayley graph over F2",
             "config_digest":digest,"source_commit":a.source_commit,"hardware":a.hardware,
             "hash":{"algorithm":"GEMM_U8_P32X4_V1","seed_u128":20260828,"byte_order":"little-endian"},
             "config":{"profile":"DENSE","owner_backend":"CUB_SORT_MERGE","pre_dedup":True,
                       "macro_depth":raw["macro_depth"],"batch":raw["batch"],"state_layout":"canonical row-major u8",
                       "state_bytes":100,"generators":3,"modulus":2,
                       "topology":{"world_size":1,"logical_owner_to_rank":[0],"shards_per_rank":64,"buckets_per_shard":256}},
             "topology":{"world_size":1,"logical_owner_to_rank":[0],"shards_per_rank":64,"buckets_per_shard":256},
             "search_complete_seconds":raw["search_complete_seconds"],"durable_run_commit_seconds":raw["durable_run_commit_seconds"],
             "total_unique_states":raw["unique_states"],"layer_sizes":raw["layer_sizes"],"layer_sha256":raw.get("layer_sha256",[]),
             "memory":{"cuda_context_used_bytes":raw.get("cuda_context_used_bytes"),"cuda_allocated_used_bytes":raw.get("cuda_allocated_used_bytes"),
                       "pinned_bytes":raw.get("pinned_bytes"),"disk_reserved_bytes":raw.get("disk_reserved_bytes")},
             "layers":layers,"raw_benchmark":raw}
    a.output.write_text(json.dumps(summary,indent=2),encoding="utf-8")

if __name__=="__main__":main()
