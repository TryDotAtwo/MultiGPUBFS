"""Summarize CUDA host waits by call chain from an Nsight SQLite export."""

import argparse
from contextlib import closing
import json
import sqlite3
from pathlib import Path


def phase_api(db, tables):
    """Inclusive named ranges, matching CPU threads; nested rows are not additive."""
    if 'NVTX_EVENTS' not in tables:
        return dict(status='UNAVAILABLE', rows=[])
    columns = {row[1] for row in db.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_RUNTIME)')}
    if 'globalTid' not in columns:
        return dict(status='UNAVAILABLE', rows=[])
    rows = db.execute('''
        SELECT n.text,s.value,COUNT(*),SUM(a.end-a.start)
        FROM NVTX_EVENTS n JOIN CUPTI_ACTIVITY_KIND_RUNTIME a
          ON a.globalTid=n.globalTid AND a.start>=n.start AND a.end<=n.end
        JOIN StringIds s ON s.id=a.nameId
        WHERE n.text IN ('mgbfs.batch','mgbfs.FinalizeDepth','mgbfs.archive_d2h')
          AND (s.value LIKE '%Synchronize%' OR s.value LIKE 'cudaMemcpy%')
        GROUP BY n.text,s.value ORDER BY n.text,s.value
    ''').fetchall()
    return dict(status='MEASURED', scope='Thread-matched fully contained runtime API intervals; nested ranges overlap; not dependency proof',
                rows=[dict(phase=phase,api=api,calls=calls,api_duration_ns=duration)
                      for phase,api,calls,duration in rows])


def summarize(database: Path) -> dict:
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as db:
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"CUPTI_ACTIVITY_KIND_RUNTIME", "StringIds"}
        if not required <= tables:
            raise ValueError(f"NSYS_CALLCHAIN_TABLES_MISSING: {sorted(required - tables)}")
        rows = []
        api_names = []
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if table not in tables:
                continue
            columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
            if not {"nameId", "callchainId", "start", "end"} <= columns:
                raise ValueError(f"NSYS_API_COLUMNS_{table}: {sorted(columns)}")
            where = ("names.value LIKE 'cudaStreamSynchronize%' OR "
                     "names.value LIKE 'cudaEventSynchronize%' OR "
                     "names.value LIKE 'cudaDeviceSynchronize%' OR "
                     "names.value LIKE 'cuCtxSynchronize%' OR "
                     "names.value LIKE 'cuEventSynchronize%' OR "
                     "names.value LIKE 'cuStreamSynchronize%' OR "
                     "names.value LIKE 'cudaMemcpy%' OR "
                     "names.value LIKE 'cuMemcpy%'")
            rows.extend(db.execute(f"""
                SELECT names.value, api.callchainId, COUNT(*),
                       SUM(api.end - api.start)
                FROM {table} AS api
                JOIN StringIds AS names ON names.id = api.nameId
                WHERE {where}
                GROUP BY names.value, api.callchainId
            """).fetchall())
            api_names.extend(db.execute(f"""
                SELECT names.value, COUNT(*) FROM {table} AS api
                JOIN StringIds AS names ON names.id = api.nameId
                WHERE {where} GROUP BY names.value
            """).fetchall())
        chains = {}
        for _, chain_id, _, _ in rows:
            if 'CUDA_CALLCHAINS' not in tables or chain_id is None or chain_id in chains:
                continue
            chains[chain_id] = [symbol for (symbol,) in db.execute("""
                SELECT names.value
                FROM CUDA_CALLCHAINS AS chain
                LEFT JOIN StringIds AS names ON names.id = chain.symbol
                WHERE chain.id = ? ORDER BY chain.stackDepth
            """, (chain_id,))]
        grouped = {}
        for name, chain_id, count, duration in rows:
            symbols = tuple(chains.get(chain_id, []))
            key = (name, symbols)
            entry = grouped.setdefault(key, {"api": name, "calls": 0,
                                             "api_duration_ns": 0, "symbols": list(symbols)})
            entry["calls"] += count
            entry["api_duration_ns"] += duration
        return {
            "scope": "CUDA runtime sync/copy API callsites in captured window; profiler overhead applies",
            "callchain_status": 'AVAILABLE' if any(any(symbol for symbol in symbols) for symbols in chains.values()) else 'UNAVAILABLE',
            "api_names": [{"api": name, "calls": count} for name, count in api_names],
            "rows": sorted(grouped.values(), key=lambda row: (-row["calls"], row["api"])),
            "phase_api": phase_api(db, tables),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.database), indent=2))
