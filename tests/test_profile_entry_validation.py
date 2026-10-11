from multigpubfs.size_profiles import valid_profile_entries

def entry(frontier=0, bucket=None):
 p=dict(minimum_frontier=frontier,batch=256,generator_backend='cuda')
 if bucket is not None:p.update(frontier_max_rank_bucket=bucket,status='LIVE_CHUNK_EMPIRICAL_PROFILE')
 return p

def test_distinct_buckets_can_share_global_frontier():
 assert valid_profile_entries([entry(),entry(100,5),entry(100,6)],512,('cuda',))

def test_invalid_bucket_and_duplicate_rejected():
 for value in [-1,32,True,'5']:
  assert not valid_profile_entries([entry(),entry(100,value)],512,('cuda',))
 assert not valid_profile_entries([entry(),entry(100,5),entry(200,5)],512,('cuda',))

def test_ordered_base_and_bucket_status_required():
 assert not valid_profile_entries([entry(1)],512,('cuda',))
 assert not valid_profile_entries([entry(),entry()],512,('cuda',))
 p=entry(100,5);p['status']='UNMEASURED_CONSERVATIVE_PROFILE'
 assert not valid_profile_entries([entry(),p],512,('cuda',))
