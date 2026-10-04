import hashlib,json,os,subprocess,sys,tarfile,tempfile,urllib.request
from pathlib import Path
PIN='1c08dc877bee62edb387b558cde01f22e958830f'
REPO='TryDotAtwo/multigpubfs-bfs-results'
ROOT=Path(tempfile.mkdtemp(prefix='mgbfs-retained-s8-',dir='/tmp'))
OUT=Path('/kaggle/working/retained-s8-hf');OUT.mkdir()
report={'status':'INCOMPLETE','source':PIN,'scope':'retained physical two-T4 S8 archives; no new GPU run'}
def save(): (OUT/'summary.json').write_text(json.dumps(report,indent=2))
def stage(name):
 report['stage']=name;save();print('STAGE',name,flush=True)
def run(args,label,cwd=None):
 p=subprocess.run(args,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=900)
 (OUT/(label+'.log')).write_text(p.stdout)
 if p.returncode: raise RuntimeError(label+'_FAILED')
 return p.stdout
save()
try:
 stage('dependencies')
 run([sys.executable,'-m','pip','install','--quiet','pyarrow==19.0.1','huggingface_hub>=0.34'],'dependencies')
 source=ROOT/'source';run(['git','clone','--quiet','https://github.com/TryDotAtwo/MultiGPUBFS.git',str(source)],'clone')
 run(['git','checkout','--detach',PIN],'checkout',source)
 archive=ROOT/'evidence.tar.gz'
 stage('release_download')
 urllib.request.urlretrieve('https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/validation-vast-54182144-20261004/vast-54182144-s8-timeline.tar.gz',archive)
 digest=hashlib.sha256(archive.read_bytes()).hexdigest()
 if digest!='589b137c665692bd43c42fd7634b90c6151003c1d17bd3816fb4e3beeb68a9ac':raise RuntimeError('RELEASE_SHA256')
 evidence=ROOT/'evidence';evidence.mkdir()
 with tarfile.open(archive,'r:gz') as t:
  for m in t.getmembers():
   if m.issym() or m.islnk() or not (m.isfile() or m.isdir()):raise RuntimeError('TAR_MEMBER_TYPE')
   if not (evidence/m.name).resolve().is_relative_to(evidence.resolve()):raise RuntimeError('TAR_PATH')
  t.extractall(evidence,filter='data')
 sys.path.insert(0,str(source/'scripts'))
 from replay_lsa_cancel_candidate import verify_process_archives
 from export_hf_dataset import frames
 from huggingface_hub import HfApi
 from kaggle_secrets import UserSecretsClient
 stage('inventory')
 candidates=[]
 for f in evidence.rglob('*.json'):
  try:d=json.loads(f.read_text())
  except (ValueError,UnicodeError):continue
  if isinstance(d,dict) and d.get('graph',{}).get('rows')==8 and 'topology' in d:candidates.append((f,d))
 inventory=[{'path':str(p.relative_to(evidence)),'bytes':p.stat().st_size} for p in evidence.rglob('*') if p.is_file()]
 (OUT/'retained-inventory.json').write_text(json.dumps(inventory,indent=2))
 report['retained_config_candidates']=[{'path':str(f.relative_to(evidence)),'profile':d.get('frontier_profile')} for f,d in candidates];save()
 cases=sorted({p.parent for p in evidence.rglob('archive-rank-0.mgbfsar1')})
 if len(cases)!=2:raise RuntimeError('EXPECTED_TWO_PROFILE_ARCHIVE_PAIRS')
 report['archive_sha256']=digest;report['runs']=[];save()
 for case in cases:
  ranks=[json.loads((case/f'result/rank-{r}.json').read_text()) for r in range(2)]
  profile=ranks[0]['frontier_profile']
  if profile not in ('DENSE','HASH_FIRST') or ranks[1]['frontier_profile']!=profile:raise RuntimeError('PROFILE')
  stage('cpu_oracle_'+profile)
  oracle=verify_process_archives(case,n=8,world=2,expected_run_contract='RunConfigV1',expected_owner_backend='CUCO_RANK',require_bank_reuse=True)
  if oracle['unique_states']!=40320:raise RuntimeError('S8_ORACLE')
  report.setdefault('cpu_oracles',{})[profile]=oracle;save()
  cfg=[d for _,d in candidates if d.get('frontier_profile')==profile]
  if not cfg:raise RuntimeError('PROFILE_CONFIG_MISSING')
  config=cfg[0]
  if any(d!=config for d in cfg):raise RuntimeError('PROFILE_CONFIG_AMBIGUOUS')
  first=next(frames(case/'archive-rank-0.mgbfsar1'));config_digest=first[4]
  seed=int(ranks[0]['hash_seed_hex'],16)
  if seed!=int.from_bytes(bytes(config['seed']),'little'):raise RuntimeError('SEED_CONFIG')
  run_id='s8-vast-54182144-local-first-'+profile.lower()
  summary={'schema':'RunSummaryV1','status':'COMPLETE','run_id':run_id,'group_id':'S_8 LRX matrix Cayley graph over F2','group_spec':'s8','config_digest':config_digest,'config':config,'topology':config['topology'],'hash':{'algorithm':'GEMM_U8_P32X4_V1','seed_u128':seed,'byte_order':'little-endian'},'total_unique_states':40320,'layer_sizes':oracle['layer_sizes'],'source_commit':'52bb77f','publication_tool_commit':PIN,'raw_rank_results':ranks,'cpu_oracle':oracle,'provenance':{'release_sha256':digest,'profiled':True,'performance_claim':False,'source_release':'validation-vast-54182144-20261004'},'layers':{}}
  sp=OUT/(run_id+'-summary.json');sp.write_text(json.dumps(summary,indent=2))
  package=OUT/(run_id+'-package')
  stage('parquet_export_'+profile)
  run([sys.executable,'scripts/export_hf_dataset.py','--run-id',run_id,'--summary',str(sp),'--archive','0='+str(case/'archive-rank-0.mgbfsar1'),'--archive','1='+str(case/'archive-rank-1.mgbfsar1'),'--output',str(package),'--rows-per-shard','10000'],'export-'+profile,source)
  stage('parquet_verify_'+profile)
  run([sys.executable,'scripts/verify_hf_dataset.py',str(package)],'verify-'+profile,source)
  verification=json.loads((package/'verification.json').read_text())
  if verification.get('status')!='PASS' or not verification.get('hash_state_pairs_verified'):raise RuntimeError('PARQUET_VERIFICATION')
  staging=ROOT/(run_id+'-catalog')
  run([sys.executable,'scripts/prepare_hf_catalog_upload.py',str(package),str(staging)],'catalog-'+profile,source)
  report.setdefault('verified_packages', []).append({'profile':profile,'run_id':run_id,'verification':verification,'oracle':oracle});save()
  stage('hf_secret_'+profile)
  try:
   token=UserSecretsClient().get_secret('HF_TOKEN')
   if not token:raise RuntimeError('HF_SECRET_EMPTY')
  except Exception as secret_error:
   cause=secret_error.__cause__
   details={'profile':profile,'stage':'KAGGLE_SECRET_ACCESS','exception_type':type(secret_error).__name__,'exception_module':type(secret_error).__module__,'cause_type':type(cause).__name__ if cause else None,'http_status':getattr(cause,'code',None),'reason_type':type(getattr(cause,'reason',None)).__name__,'errno':getattr(getattr(cause,'reason',None),'errno',None)}
   report.setdefault('publication_blockers',[]).append(details)
   save()
   continue
  api=HfApi(token=token)
  stage('hf_repo_info_'+profile)
  revision=api.repo_info(repo_id=REPO,repo_type='dataset').sha
  remote=set(api.list_repo_files(repo_id=REPO,repo_type='dataset',revision=revision))
  paths=[str(p.relative_to(staging)).replace(os.sep,'/') for p in staging.rglob('*') if p.is_file()]
  if remote.intersection(paths):raise RuntimeError('APPEND_ONLY_PATH_ALREADY_EXISTS')
  receipt=api.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(staging),path_in_repo='',parent_commit=revision,commit_message='Verified retained S8 '+profile+' full-state archive')
  after=set(api.list_repo_files(repo_id=REPO,repo_type='dataset',revision=receipt.oid))
  if not set(paths).issubset(after):raise RuntimeError('PUBLISHED_PATH_MISSING')
  report['runs'].append({'run_id':run_id,'profile':profile,'oracle':oracle,'verification':verification,'commit':receipt.oid,'commit_url':receipt.commit_url,'files':paths});save()
 report['status']='COMPLETE' if len(report['runs'])==2 else 'VERIFIED_PENDING_HF_AUTH';save();print(json.dumps(report))
except Exception as e:
 import re
 report['failure_type']=type(e).__name__;report['status']='FAILED'
 reason=str(e)
 if re.fullmatch(r'[A-Z0-9_]{1,160}',reason):report['failure_code']=reason
 save()
 print('RETAINED_S8_PUBLICATION_FAILED',type(e).__name__,flush=True)
 raise RuntimeError('RETAINED_S8_PUBLICATION_FAILED') from None
