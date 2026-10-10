import unittest,sys,tempfile,json,queue
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from b300_current_production import grid,definition,capacity_stop,payload,verify_oracle,Publisher
from b300_production import inventory
class ProductionTests(unittest.TestCase):
 def test_all_pairs(self):
  values=grid(2,128);self.assertEqual(len(values),8255);self.assertEqual(values[0],(2,1));self.assertEqual(values[-1],(128,128));self.assertEqual(len(set(values)),8255)
 def test_grid_bounds(self):
  for lo,hi in ((1,128),(2,129),(8,4)):
   with self.assertRaises(ValueError):grid(lo,hi)
 def test_four_b300_inventory(self):
  rows='\n'.join(f'{i}, NVIDIA B300, GPU-{i}, 288000, 280000, 10.3' for i in range(4));self.assertEqual(len(inventory(rows,4)),4)
 def test_capacity_errors_only(self):
  for reason in ('RESOURCE_257','RESOURCE_258','RESOURCE_260'):self.assertTrue(capacity_stop(dict(status='INCOMPLETE',reason=reason)))
  for reason in ('RESOURCE_259','RESOURCE_264','TIME_LIMIT','SIGTERM','CUDA_STATUS_2','RESOURCE_0'):self.assertFalse(capacity_stop(dict(status='INCOMPLETE',reason=reason)))
  self.assertFalse(capacity_stop(dict(status='COMPLETE',reason='RESOURCE_258')))
 def test_two_seeds_preserve_graph(self):
  for rr in (1,2,4):
   g0,_=definition(4,rr,0);g1,inverse=definition(4,rr,13);a=g0.exact_layers(24);b=g1.exact_layers(24)
   self.assertEqual(a,[[list(v) for v in sorted(tuple(inverse[x] for x in row) for row in layer)] for layer in b])
 def test_terminal_mismatch_rejected(self):
  g,_=definition(4,1);layers=g.exact_layers(24)
  with tempfile.TemporaryDirectory() as d:
   path=Path(d);states=dict(current=layers[-1],previous_small=layers[-2]);(path/'states.json').write_text(json.dumps(states));report=dict(status='COMPLETE',layer_sizes=list(map(len,layers)));verify_oracle(report,g,path,24)
   states['current']=[g.start];(path/'states.json').write_text(json.dumps(states))
   with self.assertRaises(RuntimeError):verify_oracle(report,g,path,24)
 def test_publisher_failure_before_queue_wait(self):
  p=object.__new__(Publisher);p.error=ValueError('failure');p.pending=queue.Queue(maxsize=1)
  with self.assertRaises(RuntimeError):p.enqueue({})
if __name__=='__main__':unittest.main()
