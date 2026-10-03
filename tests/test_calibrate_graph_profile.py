import hashlib
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from calibrate_graph_profile import calibrate,full_state_fingerprint
from bfs_tail_archive import TailArchive


class CalibrationTests(unittest.TestCase):
    def fixture(self, cfg, source, case, runtime, **kwargs):
        case.mkdir()
        n,r=cfg['n'],cfg['r']
        archive=TailArchive(case/'saved',n=n,r=r,start=list(range(n-r+1))+[n-r]*(r-1),actions={'L':'left','R':'right','X':'swap'},
                            program_commit='fixture',launch_config=cfg,sample_interval_seconds=.05)
        mode=int(cfg['env']['MGBFS_CUDA_GRAPH_BATCHES'])
        data=b''.join(x.to_bytes(8,'little') for x in ([2,1,0] if mode else [0,1,2]))
        if self.corrupt and mode:data=data[:-8]+(9).to_bytes(8,'little')
        prefix=getattr(self,'prefix',False)
        count=34 if prefix else 1
        for depth in range(count):
            archive.completed_layer(depth,3,[data],.1,{'0':100,'1':100})
        manifest=archive.snapshot(complete=not prefix,reason='calibration layer limit' if prefix else 'complete')
        (case/'result').mkdir()
        for rank in range(2):
            report=dict(status='INCOMPLETE' if prefix else 'COMPLETE',
                batch_graph=dict(full_windows=2 if mode else 0))
            report['search_prefix_seconds' if prefix else 'search_complete_seconds']=.9 if mode else 1.
            if prefix:report.update(calibration_layers=34,last_completed_layer=33,stop_reason='calibration layer limit')
            (case/'result'/f'rank-{rank}.json').write_text(json.dumps(report))
        return manifest

    def test_fixed_width_sort_matches_bytes_and_rejects_duplicates(self):
        for n, r in ((16, 15), (32, 31)):
            width = 8 if n <= 16 else 16
            values = [1 << (4*(n-1)), 1 << 4, 0]
            data = b''.join(value.to_bytes(width, 'little') for value in values)
            expected = hashlib.sha256(b''.join(sorted(
                data[i:i+width] for i in range(0, len(data), width)))).hexdigest()
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = TailArchive(root/'good', n=n, r=r, start=[0]+[1]*(n-1),
                    actions={'L':'left','R':'right','X':'swap'}, program_commit='fixture',
                    launch_config={}, sample_interval_seconds=.05)
                archive.completed_layer(0, 3, [data], .1, {'0':None})
                manifest = archive.snapshot(True, 'fixture')
                self.assertEqual(full_state_fingerprint(manifest), [(0, 3, expected)])
                duplicate = TailArchive(root/'duplicate', n=n, r=r, start=[0]+[1]*(n-1),
                    actions={'L':'left','R':'right','X':'swap'}, program_commit='fixture',
                    launch_config={}, sample_interval_seconds=.05)
                duplicate.completed_layer(0, 3, [data[:width]*3], .1, {'0':None})
                with self.assertRaisesRegex(ValueError, 'duplicate'):
                    full_state_fingerprint(duplicate.snapshot(True, 'fixture'))

    def test_large_case_uses_matched_prefix_without_leaking_limit(self):
        self.corrupt=False;self.prefix=True
        cfg=dict(n=15,r=4,world=2,env={'MGBFS_TRANSPORT_BACKEND':'NCCL_LSA'})
        with tempfile.TemporaryDirectory() as d:
            result=calibrate(cfg,Path(d),Path(d)/'calibration',{},deadline=time.time()+60,
                             runner=self.fixture,identity='fixture')
            self.assertEqual(result['status'],'CALIBRATED')
            self.assertEqual(result['calibration_layers'],34)
            self.assertEqual(result['scope'],'matched completed-layer prefix only')
            self.assertNotIn('MGBFS_CALIBRATION_LAYERS',cfg['env'])

    def test_actual_archive_order_independent_parity_and_decision(self):
        self.corrupt=False
        with tempfile.TemporaryDirectory() as d:
            result=calibrate(dict(n=7,r=1,world=2,env={'MGBFS_TRANSPORT_BACKEND':'NCCL_LSA'}),
                Path(d),Path(d)/'calibration',{},deadline=time.time()+60,
                runner=self.fixture,identity='fixture')
            self.assertEqual(result['status'],'CALIBRATED')
            self.assertEqual(result['graph_batches'],32)
            self.assertEqual(len(result['samples']),6)
            self.assertFalse((Path(d)/'calibration/run-0/saved').exists())
            self.assertTrue((Path(d)/'calibration/run-0/result/rank-0.json').exists())
            self.assertEqual(result['samples'][0]['layers'][0]['vram_peak_bytes'],{'0':100,'1':100})
            self.assertEqual(result['samples'][0]['vram_sampling_interval_seconds'],.05)
            self.assertFalse(result['samples'][0]['calibration_inputs_retained'])

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
