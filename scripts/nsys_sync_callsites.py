"""Summarize CUDA host waits by call chain from an Nsight SQLite export."""

import argparse
import json
import sqlite3
from pathlib import Path


def summarize(database: Path) -> dict:
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"CUPTI_ACTIVITY_KIND_RUNTIME", "CUDA_CALLCHAINS", "StringIds"}
        if not required <= tables:
            raise ValueError(f"NSYS_CALLCHAIN_TABLES_MISSING: {sorted(required - tables)}")
        columns = {row[1] for row in db.execute(
            "PRAGMA table_info(CUPTI_ACTIVITY_KIND_RUNTIME)")}
        if not {"nameId", "callchainId", "start", "end"} <= columns:
            raise ValueError(f"NSYS_RUNTIME_COLUMNS: {sorted(columns)}")
        rows = db.execute("""
            SELECT names.value, runtime.callchainId, COUNT(*),
                   SUM(runtime.end - runtime.start)
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
            JOIN StringIds AS names ON names.id = runtime.nameId
            WHERE names.value IN ('cudaStreamSynchronize', 'cudaMemcpy')
            GROUP BY names.value, runtime.callchainId
            ORDER BY names.value, COUNT(*) DESC
        """).fetchall()
        chains = {}
        for _, chain_id, _, _ in rows:
            if chain_id is None or chain_id in chains:
                continue
            chains[chain_id] = [symbol for (symbol,) in db.execute("""
                SELECT names.value
                FROM CUDA_CALLCHAINS AS chain
                LEFT JOIN StringIds AS names ON names.id = chain.symbol
                WHERE chain.id = ? ORDER BY chain.stackDepth
            """, (chain_id,))]
        return {
            "scope": "CUDA runtime sync/copy API callsites in captured window; profiler overhead applies",
            "rows": [
                {"api": name, "callchain_id": chain_id, "calls": count,
                 "api_duration_ns": duration, "symbols": chains.get(chain_id, [])}
                for name, chain_id, count, duration in rows
            ],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.database), indent=2))
