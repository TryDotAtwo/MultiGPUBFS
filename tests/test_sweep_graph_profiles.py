import sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from sweep_graph_profiles import calibrate_or_reuse
class ResidentCalibrationReuse(unittest.TestCase):
 def test_only_verified_same_geometry_reuses(self):
  with tempfile.TemporaryDirectory() as tmp:
   source=Path(tmp);(source/'mgbfs').write_bytes(b'binary');session=SimpleNamespace();calls=[]
   def measure(*a,**kw):calls.append(1);return {'status':'CALIBRATED','identity':'proof','graph_batches':32,'samples':[{'full_state_parity':True}]}
   cfg={'n':12,'r':1,'world':2,'batch':1024,'binary_path':'mgbfs','env':{'owner':'SHARD_AB'}}
   def run(c):return calibrate_or_reuse(measure,c,source,source/'result',{},deadline=1e12,session=session)
   run(cfg);v=run(dict(cfg,r=2));self.assertEqual(len(calls),1);self.assertFalse(v['target_calibration_executed'])
   run(dict(cfg,batch=2048));self.assertEqual(len(calls),2)
   (source/'mgbfs').write_bytes(b'new');run(cfg);self.assertEqual(len(calls),3)
 def test_failed_parity_not_cached(self):
  with tempfile.TemporaryDirectory() as tmp:
   source=Path(tmp);(source/'mgbfs').write_bytes(b'binary');session=SimpleNamespace();calls=[]
   def measure(*a,**kw):calls.append(1);return {'status':'CALIBRATED','identity':'proof','graph_batches':32,'samples':[{'full_state_parity':False}]}
   cfg={'n':12,'r':1,'world':2,'batch':1024,'binary_path':'mgbfs','env':{}}
   for _ in range(2):calibrate_or_reuse(measure,cfg,source,source/'result',{},deadline=1e12,session=session)
   self.assertEqual(len(calls),2)
