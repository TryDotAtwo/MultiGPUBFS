import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from automatic_transport import select_transport


class TransportTests(unittest.TestCase):
    def test_actual_gate_and_oracle_required(self):
        for good in (True,False):
            with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{},clear=True):
                runner=Mock()
                verifier=Mock(return_value=dict(status='VERIFIED_FULL_STATE_LAYERS',states=24 if good else 23))
                base=dict(world=2,env={'NCCL_CUMEM_ENABLE':'0'})
                result=select_transport(base,Path('source'),Path(d)/'gate',{},
                    deadline=time.time()+60,runner=runner,verifier=verifier)
                self.assertEqual(result['env']['MGBFS_TRANSPORT_BACKEND'],
                                 'NCCL_LSA' if good else 'HOST_SIZED_NCCL')
                self.assertEqual(base['env'],{'NCCL_CUMEM_ENABLE':'0'})
                self.assertEqual(result['env']['NCCL_CUMEM_ENABLE'],'1' if good else '0')
                cfg=runner.call_args.args[0]
                self.assertEqual((cfg['n'],cfg['r']),(4,1))
                self.assertEqual(cfg['env']['MGBFS_CUDA_GRAPH_BATCHES'],'0')

    def test_failure_retains_host_transport(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{},clear=True):
            result=select_transport(dict(world=2,env={}),Path('source'),Path(d)/'gate',{},
                deadline=time.time()+60,runner=Mock(side_effect=RuntimeError('LSA unavailable')))
            self.assertEqual(result['transport_selection']['status'],'FAILED')
            self.assertEqual(result['env']['MGBFS_TRANSPORT_BACKEND'],'HOST_SIZED_NCCL')

    def test_explicit_setting_and_single_gpu_do_not_launch(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{},clear=True):
            runner=Mock(side_effect=AssertionError('GPU work started'))
            explicit=select_transport(dict(world=2,env={}),Path('source'),Path(d)/'explicit',
                {'MGBFS_TRANSPORT_BACKEND':'HOST_SIZED_NCCL'},deadline=time.time()+60,runner=runner)
            self.assertEqual(explicit['env']['MGBFS_TRANSPORT_BACKEND'],'HOST_SIZED_NCCL')
            select_transport(dict(world=1,env={}),Path('source'),Path(d)/'single',{},
                deadline=time.time()+60,runner=runner)
            runner.assert_not_called()


if __name__=='__main__':unittest.main()
