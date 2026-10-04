import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from nsys_sync_callsites import summarize


class PhaseAttributionTest(unittest.TestCase):
    def test_thread_matching_does_not_charge_worker_wait_to_main_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.sqlite'
            with closing(sqlite3.connect(path)) as db:
                db.executescript('''
                    CREATE TABLE StringIds(id INTEGER,value TEXT);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME
                      (nameId INTEGER,callchainId INTEGER,start INTEGER,end INTEGER,globalTid INTEGER);
                    CREATE TABLE NVTX_EVENTS(text TEXT,start INTEGER,end INTEGER,globalTid INTEGER);
                    INSERT INTO StringIds VALUES (1,'cudaStreamSynchronize'),(2,'cudaMemcpyAsync');
                    INSERT INTO NVTX_EVENTS VALUES
                      ('mgbfs.batch',10,40,1),('mgbfs.archive_d2h',15,25,1),
                      ('mgbfs.FinalizeDepth',50,80,1);
                    INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
                      (1,NULL,20,22,2),(2,NULL,16,18,1),(1,NULL,60,65,1);
                ''')
            report = summarize(path)
            self.assertIn('phase_api', report)
            self.assertEqual(report['phase_api']['rows'], [
                dict(phase='mgbfs.FinalizeDepth',api='cudaStreamSynchronize',calls=1,api_duration_ns=5),
                dict(phase='mgbfs.archive_d2h',api='cudaMemcpyAsync',calls=1,api_duration_ns=2),
                dict(phase='mgbfs.batch',api='cudaMemcpyAsync',calls=1,api_duration_ns=2)])


if __name__ == '__main__':
    unittest.main()
