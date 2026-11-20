import hashlib,json
from pathlib import Path
import pytest
from scripts.verify_bitmap_benchmark import compare

def make_case(root,name,reverse=False):
 case=root/name
 for rank in range(2):
  directory=case/f'rank-{rank}';directory.mkdir(parents=True)
  previous=[[rank,1],[rank,2]]; previous=previous[::-1] if reverse else previous
  state={'current':[[rank,0]],'previous_small':previous}
  raw=json.dumps(state).encode();(directory/'states.json').write_bytes(raw)
  report={'rank':rank,'world':2,'status':'COMPLETE','graph_digest':'same','layer_sizes':[1,4,2],'states_sha256':hashlib.sha256(raw).hexdigest()}
  (directory/'report.json').write_text(json.dumps(report))
 return case

def test_compare_ignores_order_and_checks_file_integrity(tmp_path):
 a=make_case(tmp_path,'a'); b=make_case(tmp_path,'b',True)
 assert compare([a,b])['status']=='VERIFIED_SAME_COMPLETE_RETAINED_LAYERS'
 p=b/'rank-0'/'states.json';p.write_text(p.read_text()+' ')
 with pytest.raises(ValueError,match='hash mismatch'):compare([a,b])

def test_compare_rejects_different_states_and_partial_retention(tmp_path):
 a=make_case(tmp_path,'a');b=make_case(tmp_path,'b')
 p=b/'rank-0'/'states.json';data=json.loads(p.read_text());data['previous_small'][0]=[99,99]
 raw=json.dumps(data).encode();p.write_bytes(raw);r=b/'rank-0'/'report.json';report=json.loads(r.read_text());report['states_sha256']=hashlib.sha256(raw).hexdigest();r.write_text(json.dumps(report))
 with pytest.raises(ValueError,match='disagree'):compare([a,b])
 data['previous_small'].pop();raw=json.dumps(data).encode();p.write_bytes(raw);report['states_sha256']=hashlib.sha256(raw).hexdigest();r.write_text(json.dumps(report))
 with pytest.raises(ValueError,match='incomplete'):compare([a,b])


def test_compare_across_different_owner_mappings(tmp_path):
 a=make_case(tmp_path,'a');b=make_case(tmp_path,'b')
 p0=b/'rank-0'/'states.json';p1=b/'rank-1'/'states.json'
 s0=json.loads(p0.read_text());s1=json.loads(p1.read_text())
 s0['current'],s1['current']=s1['current'],s0['current']
 for rank,state in enumerate((s0,s1)):
  p=b/f'rank-{rank}'/'states.json';raw=json.dumps(state).encode();p.write_bytes(raw)
  r=b/f'rank-{rank}'/'report.json';report=json.loads(r.read_text());report['states_sha256']=hashlib.sha256(raw).hexdigest();r.write_text(json.dumps(report))
 with pytest.raises(ValueError,match='disagree'):compare([a,b])
 assert compare([a,b],across_owners=True)['status']=='VERIFIED_SAME_COMPLETE_RETAINED_LAYERS'

def test_across_owners_rejects_duplicate_states(tmp_path):
 a=make_case(tmp_path,'a');b=make_case(tmp_path,'b')
 p=b/'rank-0'/'states.json';state=json.loads(p.read_text());state['current']=[[1,0]]
 raw=json.dumps(state).encode();p.write_bytes(raw);r=b/'rank-0'/'report.json';report=json.loads(r.read_text());report['states_sha256']=hashlib.sha256(raw).hexdigest();r.write_text(json.dumps(report))
 with pytest.raises(ValueError,match='duplicate state across owners'):compare([a,b],across_owners=True)
