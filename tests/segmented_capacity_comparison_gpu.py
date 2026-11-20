"""Exact CPU-oracle comparison under equal *artificial* GPU storage byte caps."""
import json,os
from segmented_variable_budget_gpu import run
results=[]
for bitmap,budget in ((False,1526),(True,294)):
 pair=[]
 for preset in ('balanced','max_capacity'):
  os.environ.pop('MGBFS_SEGMENT_COMPACT_POS',None);os.environ.pop('MGBFS_BITMAP_BANKS',None)
  if preset=='max_capacity':
   os.environ['MGBFS_BITMAP_BANKS' if bitmap else 'MGBFS_SEGMENT_COMPACT_POS']='2' if bitmap else '1'
  outcome=run(bitmap,budget,preset);outcome['preset']=preset;pair.append(outcome);results.append(outcome)
  print(json.dumps(outcome),flush=True)
 assert pair[1]['depth']>pair[0]['depth'],pair
print('CONTROLLED_BUDGET_GREATER_COMPLETED_DEPTH_FOR_BOTH_BACKENDS',flush=True)
