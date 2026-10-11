import json,os
from pathlib import Path
from multigpubfs.size_profiles import save_online_profiles

def test_bucket_coalesces_different_global_frontiers(tmp_path,monkeypatch):
 monkeypatch.setenv('MGBFS_PROFILE_CACHE',str(tmp_path))
 profile={'status':'MEASURED_FRONTIER_SIZE_PROFILES','identity':{'fixture':1},'batch':256,'size_profiles':[{'minimum_frontier':0,'batch':64,'generator_backend':'cuda'}]}
 events=[{'winner':0,'choices':[[64,False]],'frontier_global':v,'frontier_max_rank':20000+i,'bucket':14} for i,v in enumerate([40000,50000])]
 save_online_profiles({'autotune':profile,'online_size_profile_events':events})
 saved=json.loads(next(tmp_path.glob('size-*.json')).read_text())
 assert len(saved['size_profiles'])==2
 assert saved['size_profiles'][1]['maximum_measured_frontier']==50000
 assert saved['size_profiles'][1]['frontier_max_rank_bucket']==14

def test_inconsistent_bucket_rejected(tmp_path,monkeypatch):
 monkeypatch.setenv('MGBFS_PROFILE_CACHE',str(tmp_path))
 profile={'status':'MEASURED_FRONTIER_SIZE_PROFILES','identity':{'fixture':1},'batch':256,'size_profiles':[{'minimum_frontier':0,'batch':64,'generator_backend':'cuda'}]}
 save_online_profiles({'autotune':profile,'online_size_profile_events':[{'winner':0,'choices':[[64,False]],'frontier_global':40000,'frontier_max_rank':20000,'bucket':1}]})
 saved=json.loads(next(tmp_path.glob('size-*.json')).read_text())
 assert len(saved['size_profiles'])==1
