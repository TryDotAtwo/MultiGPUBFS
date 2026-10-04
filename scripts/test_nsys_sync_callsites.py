import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from nsys_sync_callsites import summarize


class SyncCallsiteTest(unittest.TestCase):
    def test_missing_callchains_keeps_counts_without_inventing_source_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'no-callchains.sqlite'
            with closing(sqlite3.connect(database)) as db:
                db.executescript('''
                    CREATE TABLE StringIds (id INTEGER, value TEXT);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME
                        (nameId INTEGER, callchainId INTEGER, start INTEGER, end INTEGER);
                    INSERT INTO StringIds VALUES (1,'cudaDeviceSynchronize');
                    INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1,7,0,10);
                ''')
            result = summarize(database)
            self.assertEqual(result['callchain_status'], 'UNAVAILABLE')
            self.assertEqual(result['rows'], [{'api':'cudaDeviceSynchronize', 'calls':1,
                'api_duration_ns':10, 'symbols':[]}])

    def test_does_not_hide_event_device_or_driver_host_drains(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'drains.sqlite'
            with closing(sqlite3.connect(database)) as db:
                db.executescript('''
                    CREATE TABLE StringIds (id INTEGER, value TEXT);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME
                        (nameId INTEGER, callchainId INTEGER, start INTEGER, end INTEGER);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER
                        (nameId INTEGER, callchainId INTEGER, start INTEGER, end INTEGER);
                    CREATE TABLE CUDA_CALLCHAINS
                        (id INTEGER, stackDepth INTEGER, symbol INTEGER);
                ''')
                names = ['cudaEventSynchronize', 'cudaDeviceSynchronize',
                         'cuCtxSynchronize', 'cuEventSynchronize', 'cuStreamSynchronize']
                db.executemany('INSERT INTO StringIds VALUES (?,?)', enumerate(names, 1))
                db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,NULL,0,10)',
                               [(1,), (2,)])
                db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (?,NULL,0,10)',
                               [(3,), (4,), (5,)])
                db.commit()
            result = summarize(database)
            self.assertEqual({row['api'] for row in result['rows']}, set(names))
            self.assertEqual(sum(row['calls'] for row in result['rows']), 5)

    def test_groups_cuda_runtime_calls_by_callchain(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "trace.sqlite"
            with closing(sqlite3.connect(database)) as db:
                db.executescript("""
                    CREATE TABLE StringIds (id INTEGER, value TEXT);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME
                        (nameId INTEGER, callchainId INTEGER, start INTEGER, end INTEGER);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER
                        (nameId INTEGER, callchainId INTEGER, start INTEGER, end INTEGER);
                    CREATE TABLE CUDA_CALLCHAINS
                        (id INTEGER, stackDepth INTEGER, symbol INTEGER);
                    INSERT INTO StringIds VALUES
                        (1, 'cudaStreamSynchronize'), (2, 'cudaMemcpy'),
                        (3, 'owner_commit'), (4, 'cuMemcpyDtoH_v2');
                    INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
                        (1, 7, 0, 10), (1, 7, 11, 31), (2, 7, 32, 35);
                    INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (4, NULL, 36, 40);
                    INSERT INTO CUDA_CALLCHAINS VALUES (7, 0, 3);
                """)
            rows = summarize(database)["rows"]
            self.assertEqual(rows, [
                {"api": "cudaStreamSynchronize", "calls": 2,
                 "api_duration_ns": 30, "symbols": ["owner_commit"]},
                {"api": "cuMemcpyDtoH_v2", "calls": 1,
                 "api_duration_ns": 4, "symbols": []},
                {"api": "cudaMemcpy", "calls": 1,
                 "api_duration_ns": 3, "symbols": ["owner_commit"]},
            ])


if __name__ == "__main__":
    unittest.main()
