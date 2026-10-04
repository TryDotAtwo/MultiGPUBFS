import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from resident_session import Session


WORKER = '''import json,sys,time
from pathlib import Path
root=Path(sys.argv[1]);seq=0
while not (root/'shutdown').exists():
 p=root/f'job-{seq:08}.json'
 if not p.exists():time.sleep(.001);continue
 job=json.loads(p.read_text());code=int(job['env'].get('MGBFS_TEST_CODE','0'))
 for rank in [1,0]:
  print('PAYLOAD',rank,job['args'][1],flush=True)
  if code:print(json.dumps(dict(status='ERROR',rank=rank,error='MEMORY_QUERY_DONE' if job['env'].get('MGBFS_MEMORY_QUERY')=='1' else 'TEST_FAILURE')),flush=True)
  print(f'MGBFS_SESSION_DONE sequence={seq} rank={rank} code={code}',flush=True)
 seq+=1
 if code and job['env'].get('MGBFS_MEMORY_QUERY')!='1':break
'''


class ResidentTests(unittest.TestCase):
    def test_jobs_reuse_pid_queries_continue_and_failures_restart(self):
        original = subprocess.Popen
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root/'fake.py'
            worker.write_text(WORKER)
            commands = []
            def spawn(command, **kwargs):
                commands.append(command)
                return original([sys.executable, str(worker), command[-1]], **kwargs)
            session = Session(root/'session')
            command = ['torchrun','--no-python','native','bench','--reference',
                'lrx17r1','2','bootstrap','archive','result']
            env = dict(__import__('os').environ, MGBFS_BENCH_WORLD_SIZE='2')
            with patch('resident_session.subprocess.Popen', side_effect=spawn):
                try:
                    first = session.launch(command, env)
                    text, _ = first.communicate(timeout=5)
                    self.assertEqual(first.returncode, 0)
                    self.assertIn('lrx17r1', text)
                    query = session.launch(command, dict(env, MGBFS_MEMORY_QUERY='1',MGBFS_TEST_CODE='1'))
                    query.communicate(timeout=5)
                    second = session.launch(command, env)
                    second.communicate(timeout=5)
                    self.assertEqual(first.pid, second.pid)
                    fail = session.launch(command, dict(env,MGBFS_TEST_CODE='1'))
                    fail.communicate(timeout=5)
                    third = session.launch(command, env)
                    third.communicate(timeout=5)
                    self.assertNotEqual(first.pid, third.pid)
                    self.assertEqual(len(commands), 2)
                    jobs = [json.loads(p.read_text()) for p in (root/'session/generation-0').glob('job-*.json')]
                    self.assertTrue(all(len(j['args']) == 6 for j in jobs))
                    self.assertNotIn('RANK', jobs[0]['env'])
                finally:
                    session.close()

if __name__ == '__main__':unittest.main()
