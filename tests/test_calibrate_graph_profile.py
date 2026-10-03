import hashlib
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from calibrate_graph_profile import calibrate
from bfs_tail_archive import TailArchive


class CalibrationTests(unittest.TestCase):
    def fixture(self, cfg, source, case, runtime, **kwargs):
        case.mkdir()
        archive=TailArchive(case/'saved',n=7,r=1,start=list(range(7)),actions={'L':'left','R':'right','X':'swap'},
                            program_commit='fixture',launch_config=cfg,sample_interval_seconds=.05)
        mode=int(cfg['env']['MGBFS_CUDA_GRAPH_BATCHES'])
        data=b''.join(x.to_bytes(8,'little') for x in ([2,1,0] if mode else [0,1,2]))
        if self.corrupt and mode:data=data[:-8]+(9).to_bytes(8,'little')
        archive.completed_layer(0,3,[data],.1,{'0':100,'1':100})
        manifest=archive.snapshot(complete=True,reason='complete')
        (case/'result').mkdir()
        for rank in range(2):
            (case/'result'/f'rank-{rank}.json').write_text(json.dumps(dict(
                status='COMPLETE',search_complete_seconds=.9 if mode else 1.,
                batch_graph=dict(full_windows=2 if mode else 0))))
        return manifest

    def test_actual_archive_order_independent_parity_and_decision(self):
        self.corrupt=False
        with tempfile.TemporaryDirectory() as d:
            result=calibrate(dict(n=7,r=1,world=2,env={'MGBFS_TRANSPORT_BACKEND':'NCCL_LSA'}),
                Path(d),Path(d)/'calibration',{},deadline=time.time()+60,
                runner=self.fixture,identity='fixture')
            self.assertEqual(result['status'],'CALIBRATED')
            self.assertEqual(result['graph_batches'],32)
            self.assertEqual(len(result['samples']),6)

    def test_parity_failure_keeps_direct_and_evidence(self):
        self.corrupt=True
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'calibration'
            result=calibrate(dict(n=7,r=1,world=2,env={'MGBFS_TRANSPORT_BACKEND':'NCCL_LSA'}),
                Path(d),root,{},deadline=time.time()+60,runner=self.fixture,identity='fixture')
            self.assertEqual(result['graph_batches'],0)
            self.assertEqual(result['status'],'NOT_CALIBRATED')
            self.assertIn('parity',result['reason'])
            self.assertTrue((root/'run-1/saved/manifest.json').exists())

    def test_deadline_and_unsupported_transport_do_not_run(self):
        for env,deadline in [({},time.time()+60),
                            ({'MGBFS_TRANSPORT_BACKEND':'NCCL_LSA'},time.time())]:
            with tempfile.TemporaryDirectory() as d:
                result=calibrate(dict(n=7,r=1,world=2,env=env),Path(d),Path(d)/'cal',{},
                    deadline=deadline,runner=lambda *a,**k:self.fail('GPU work started'))
                self.assertEqual(result['status'],'NOT_CALIBRATED')


if __name__=='__main__':unittest.main()
