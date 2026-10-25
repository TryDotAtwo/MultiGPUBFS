"""Bounded, isolated recovery experiments. Never reset GPUs or retry production BFS."""
import argparse
import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

from .diagnostics import Runner, _write

POLICIES = (
    ('local-shm', {'NCCL_NET':'Socket','NCCL_IB_DISABLE':'1','NCCL_NET_GDR_LEVEL':'LOC','NCCL_P2P_DISABLE':'1','NCCL_SHM_DISABLE':'0'}),
    ('local-p2p', {'NCCL_NET':'Socket','NCCL_IB_DISABLE':'1','NCCL_NET_GDR_LEVEL':'LOC','NCCL_P2P_DISABLE':'0','NCCL_SHM_DISABLE':'0'}),
    ('socket', {'NCCL_NET':'Socket','NCCL_IB_DISABLE':'1','NCCL_NET_GDR_LEVEL':'LOC','NCCL_P2P_DISABLE':'1','NCCL_SHM_DISABLE':'1'}),
    ('host-staged-ib', {'NCCL_NET_GDR_LEVEL':'LOC','NCCL_P2P_DISABLE':'1'}),
    ('inherited', {}),
)
EXPECTED = [1,6,26,110,414]

def ut_graph(q):
    from . import GraphDefinition
    matrices=[]
    for value in (1,q-1):
        for row in range(3):
            m=[int(i//4==i%4) for i in range(16)];m[4*row+row+1]=value
            matrices.append({'matrix':m,'modulo':q})
    return GraphDefinition.from_dict({'schema':2,'name':'UT(4,'+str(q)+')',
        'action':{'kind':'matrix','rows':4,'cols':4,'generators':matrices},
        'start':[int(i//4==i%4) for i in range(16)],'generator_names':[str(i) for i in range(6)],
        'expected_max_unique_states':None})

def walk(value):
    if isinstance(value,dict):
        yield value
        for child in value.values():yield from walk(child)
    elif isinstance(value,list):
        for child in value:yield from walk(child)

def validate_production(report,devices):
    if report.get('devices')!=devices:raise RuntimeError('RECOVERY_GPU_SET_MISMATCH')
    if report.get('layer_sizes',[])[:5]!=EXPECTED:raise RuntimeError('RECOVERY_UT47_PREFIX_NOT_VERIFIED')
    tune=report.get('autotune') or {}
    if not (tune.get('measured') is True or (tune.get('global_profile') or {}).get('measured') is True):
        raise RuntimeError('RECOVERY_AUTOTUNE_NOT_MEASURED')
    if any(x.get('candidate_failures') for x in walk(tune)):
        raise RuntimeError('RECOVERY_AUTOTUNE_CANDIDATE_FAILED')
    if report.get('status') not in ('COMPLETE','INCOMPLETE'):raise RuntimeError('RECOVERY_BFS_STATUS')
    if report['status']=='COMPLETE' and sum(report['layer_sizes'])!=47**6:raise RuntimeError('RECOVERY_CARDINALITY')
    return {'measured_autotune':True,'prefix_verified':True,'full_graph_complete':report['status']=='COMPLETE'}

def verified(step):
    return (step.get('status')=='PASS' and step.get('result',{}).get('status')=='PASS' and step.get('result',{}).get('data',{}).get('status') not in ('FAIL','ERROR','UNAVAILABLE','INCOMPLETE'))

def acceptance(step,devices):
    data=step.get('result',{}).get('data',{})
    return (verified(step) and data.get('devices')==devices and
            data.get('cpu_oracles_verified') is True and data.get('measured_autotune') is True and
            data.get('prefix_verified') is True and step.get('cleanup_complete') is True)

def evidence(root):
    result={}
    for name,command in [('topology',['nvidia-smi','topo','-m']),('gpu-info',['nvidia-smi','-q']),
                         ('kernel-errors',['dmesg','--level=err,warn'])]:
        try:
            r=subprocess.run(command,capture_output=True,timeout=8)
            raw=(r.stdout+b'\nSTDERR\n'+r.stderr)[-1024*1024:]
            (root/(name+'.log')).write_bytes(raw)
            result[name]={'exit_code':r.returncode,'availability':'AVAILABLE' if r.returncode==0 else 'UNAVAILABLE'}
        except (OSError,subprocess.TimeoutExpired) as error:result[name]={'availability':'UNAVAILABLE','error':str(error)}
    return result

def experiment(config,root):
    from .native_distribution import native_runtime, runtime_identity
    from .deep_diagnostics import component, oracle
    from .launch import run_graph
    from .generic_session import persistent_graph_workers
    kind=config['kind'];devices=config.get('devices')
    if kind=='evidence':return evidence(root)
    if kind=='identity':return fingerprint()
    if kind=='deep':return component(config['case'],root)
    native,env=native_runtime()
    env.update(NCCL_DEBUG='INFO',NCCL_DEBUG_FILE=str(root/'nccl-%h-%p.log'),
        MGBFS_TRACE_NATIVE_CALLS='1',MGBFS_TUNING_EVIDENCE_ROOT=str(root/'tuning-evidence'),
        MGBFS_PROFILE_CACHE=str(root/'profile-cache'),MGBFS_GENERIC_SESSION_LOG_ROOT=str(root/'session-live'))
    (root/'session-live').mkdir(exist_ok=True)
    os.environ.update({k:env[k] for k in ('MGBFS_PROFILE_CACHE','MGBFS_GENERIC_SESSION_LOG_ROOT')})
    if kind=='exchange':
        from .nccl_preflight import prepare
        _,receipt=prepare(native,devices,env,root/'exchange',graph=ut_graph(3))
        if receipt.get('status') not in ('VERIFIED','SINGLE_GPU'):raise RuntimeError('EXCHANGE_NOT_VERIFIED')
        return {'devices':devices,'exchange':receipt}
    if kind!='accept':raise ValueError('Unknown experiment')
    results=[]
    with persistent_graph_workers() as pool:
        try:
            # Two identical requests exercise resident reuse, followed by a real measured pilot.
            for index in range(2):
                graph=ut_graph(3);layers,terminal=oracle(graph);out=root/('oracle-'+str(index))
                report=run_graph(graph,out,devices=devices,autotune=True,max_seconds=30,_native_env=env)
                if report.get('devices')!=devices or report['status']!='COMPLETE' or report['layer_sizes']!=layers:
                    raise RuntimeError('RECOVERY_CPU_LAYER_ORACLE_MISMATCH')
                saved=json.loads((out/'states.json').read_text());rows=saved['current'] if isinstance(saved,dict) else saved
                if sorted(tuple(x) for x in rows)!=terminal:raise RuntimeError('RECOVERY_CPU_STATE_ORACLE_MISMATCH')
                results.append({'status':report['status'],'layers':layers,'states_sha256':report.get('states_sha256'),
                    'worker_pids':[w.process.pid for w in pool.workers]})
            report=run_graph(ut_graph(47),root/'ut47',devices=devices,autotune=True,max_seconds=180,_native_env=env)
            verdict=validate_production(report,devices)
        finally:
            if pool.root.exists():
                shutil.copytree(pool.root,root/'session-evidence',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('job-*','*.bin','states*'))
    return dict(verdict,devices=devices,cpu_oracles_verified=True,oracles=results,
        runtime=runtime_identity(native,env),production={k:report.get(k) for k in ('status','reason','layer_sizes','autotune','plan')})

class Supervisor:
    def __init__(self,root,seconds):
        self.runner=Runner(root,60);self.runner.deep_capture=True
        self.deadline=time.monotonic()+seconds;self.blocked=False;self.devices=[]
        self.runner.report.update(mode='recovery',total_timeout_seconds=seconds,
            scope='Single-host bounded experiments; no reset, driver changes, production restart or proof of every graph')

    def case(self,name,config,policy=None,seconds=60,gpu=False):
        r=self.runner;remaining=self.deadline-time.monotonic()-10
        reason=('CUDA_HEALTH_BLOCKED' if gpu and self.blocked else
                'OWNED_CHILD_CLEANUP_BLOCKED' if getattr(r,'deep_cleanup_blocked',False) else
                'GLOBAL_DIAGNOSTIC_BUDGET_EXHAUSTED' if remaining<min(10,seconds) else None)
        if reason:
            r.skip(name,reason);return r.report['steps'][-1]
        folder=r.root/name;folder.mkdir();_write(folder/'case-input.json',config)
        env=dict(os.environ);env.update(policy or {});env['PYTHONPATH']=str(Path(__file__).resolve().parent.parent)
        env['MGBFS_DIAGNOSTIC_SUPERVISED']='1'
        env['MGBFS_DIAGNOSTIC_TRACE_FILE']=str(folder/'native-startup.log')
        old=r.timeout;r.timeout=min(seconds,remaining)
        command=[sys.executable,'-u','-m','multigpubfs.recovery_diagnostics']
        command+=['--profile',config['profile'],'--output',str(folder),'--check'] if config['kind']=='check' else ['--child',str(folder)]
        try:step=r.run(name,command,environment=env)
        finally:r.timeout=old
        step['policy']=policy or {};step['diagnostic_case']=config
        step['deadline_scope']='total-budget' if remaining<seconds else 'stage';step['requested_stage_seconds']=seconds
        cleanup=folder/'cleanup.json'
        step['cleanup_complete']=cleanup.is_file() and json.loads(cleanup.read_text()).get('complete') is True
        if gpu and verified(step) and not step['cleanup_complete']:
            step.update(status='FAIL',error='CLEANUP_NOT_VERIFIED');self.blocked=True
        r.save();return step

    def health(self,name,policy=None):
        step=self.case(name,{'kind':'deep','case':{'kind':'component','stage':'memory','devices':self.devices,'seconds':20}},policy,45,gpu=True)
        if step['status'] in ('FAIL','TIMEOUT'):self.blocked=True
        return verified(step)

    def after_failure(self,name,policy):
        self.case(name+'-system-evidence',{'kind':'evidence'},seconds=30)
        healthy=self.health(name+'-fresh-cuda-health',policy)
        if not healthy and self.blocked:
            self.runner.report['recovery_blocked_reason']='Fresh processes cannot complete CUDA memory readback after failure; no transport verdict for unstarted alternatives. A new allocation/node or administrator recovery may be required.'
        return healthy and not self.blocked

def fingerprint():
    from .server_recipe import server_identity,code_digest
    from .native_distribution import native_runtime,runtime_identity
    native,env=native_runtime()
    server=server_identity();critical={k:v for k,v in os.environ.items() if (k.startswith(('CUDA_','NCCL_')) and k not in ('NCCL_DEBUG','NCCL_DEBUG_FILE','NCCL_DEBUG_SUBSYS')) or k in ('LD_LIBRARY_PATH','LD_PRELOAD','MGBFS_EXECUTABLE','MGBFS_NATIVE_OFFLINE')}
    server['environment_sha256']=hashlib.sha256(json.dumps(critical,sort_keys=True).encode()).hexdigest()
    return {'server':server,'code_sha256':code_digest(),'runtime':runtime_identity(native,env)}

def read_profile(path):
    path=Path(path).absolute();profile=json.loads(path.read_text())
    if profile.get('schema')!=1 or profile.get('status')!='VERIFIED':raise RuntimeError('RECOVERY_PROFILE_NOT_VERIFIED')
    if profile['identity']!=fingerprint():raise RuntimeError('RECOVERY_HOST_PACKAGE_OR_ENV_CHANGED: rerun doctor in this GPU job')
    proof=(path.parent/profile['proof']['path']).resolve()
    if not proof.is_relative_to(path.parent.resolve()) or not proof.is_file() or proof.is_symlink():raise RuntimeError('RECOVERY_PROOF_PATH')
    if hashlib.sha256(proof.read_bytes()).hexdigest()!=profile['proof']['sha256']:raise RuntimeError('RECOVERY_PROOF_CHECKSUM')
    step=json.loads(proof.read_text())
    if not acceptance(step,profile['devices']):raise RuntimeError('RECOVERY_PROOF_NOT_ACCEPTED')
    if profile['policy']!=step.get('policy'):raise RuntimeError('RECOVERY_POLICY_PROOF_MISMATCH')
    return profile

def findings(root,steps):
    """Evidence-linked classifications, not invented hardware root causes."""
    rules=[
      ('IB_LOCAL_PROTECTION','IBV_WC_LOC_PROT_ERR','InfiniBand reported a local protection error. This localizes the failing transport, not the unique cause. Use a verified IB-disabled recipe; retain HCA/vendor evidence for the operator.'),
      ('CUDA_UNKNOWN','CUDA_999','CUDA reported unknown error 999. Check the named native call and fresh-process health; do not infer memory exhaustion from this code.'),
      ('OFFLINE_COMPONENT_MISSING','NATIVE_ASSET_OFFLINE_MISSING','Prefetch the named architecture component into the same shared cache before the GPU job, or permit the existing verified component download.'),
      ('STALE_FILESYSTEM','Stale file handle','Use job-local scratch for temporary worker exchange and a stable output directory; preserve the original failing path.'),
      ('OUT_OF_MEMORY','out of memory','Check per-rank free memory and measured NCCL reserve; reduce admitted capacity only after communication health passes.'),
      ('CPU_RESOURCE','Resource temporarily unavailable','Check allocated CPU/process limits and worker count; request enough CPU resources for all ranks.'),
      ('PERMISSION','Permission denied','Inspect the exact denied file/device. Do not change device permissions or reset the GPU automatically.'),
    ]
    result=[];seen=set()
    for path in sorted(root.rglob('*.log')):
        if path.is_symlink():continue
        with path.open('rb') as stream:
            size=path.stat().st_size;stream.seek(max(0,size-(1<<20)));text=stream.read(1<<20).decode(errors='replace')
        for code,needle,action in rules:
            if needle in text and code not in seen:
                seen.add(code);result.append({'code':code,'evidence':path.relative_to(root).as_posix(),'action':action,'certainty':'observed signature; unique root cause not established'})
    for step in steps:
        if step['status'] in ('FAIL','TIMEOUT'):
            result.append({'code':'FAILED_STAGE','stage':step['name'],'error':step.get('error'),
                'evidence':step.get('log',step['name']+'/result.json'),
                'action':'Inspect the last native/API phase and the immediately following fresh CUDA health test.'})
    return result


def package(root):
    allowed={'report.json','report.txt','result.json','probe.json','phase.json','cleanup.json',
        'process-snapshot.json','post-cleanup-snapshot.json','case-input.json','failure.json','request.json',
        'recovery-profile.json','recovery-proof.json','recovery-summary.txt','launch-command.txt','graph.json',
        'cayleypy-example.py','api-proof.json','auto-registration.json','failure-context.json','summary.json','findings.json','analysis.json','analysis.txt','environment.json','driver.json','runtime.json','working-config.json'}
    entries=[];omitted=[];total=0
    with zipfile.ZipFile(root/'diagnostics.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob('*')):
            if not f.is_file() or f.is_symlink() or any(part.startswith('job-') for part in f.relative_to(root).parts):continue
            if not (f.name in allowed or f.suffix in ('.log','.ndjson') or f.name.startswith(('rank-','phase-','response-')) and f.suffix=='.json'):continue
            size=f.stat().st_size;name=f.relative_to(root).as_posix()
            limit=4<<20
            if total+min(size,limit)>128<<20:omitted.append({'path':name,'reason':'archive metadata budget'});continue
            with f.open('rb') as stream:
                if size>limit:stream.seek(-limit,2);omitted.append({'path':name,'reason':'tail only','original_bytes':size});name+='.tail.txt'
                data=stream.read(limit)
            total+=len(data);entries.append({'path':name,'sha256':hashlib.sha256(data).hexdigest(),'size':len(data)});z.writestr(name,data)
        z.writestr('manifest.json',json.dumps({'files':entries,'omissions':omitted,'scope':'metadata only; no BFS state arrays or complete job environments'},indent=2))
    with zipfile.ZipFile(root/'diagnostics.zip') as z:
        if z.testzip():raise RuntimeError('DIAGNOSTIC_ARCHIVE_CRC')
        for entry in entries:
            if hashlib.sha256(z.read(entry['path'])).hexdigest()!=entry['sha256']:raise RuntimeError('DIAGNOSTIC_ARCHIVE_SHA')

def run(output=None,devices=None,total_timeout=1800):
    if not math.isfinite(total_timeout) or total_timeout<1:raise ValueError('Positive finite total timeout required')
    if devices is not None and (not devices or len(set(devices))!=len(devices) or any(type(d)is not int or d<0 for d in devices)):raise ValueError('Distinct nonnegative devices required')
    root=Path(output or ('mgbfs-recovery-'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'-'+str(os.getpid()))).absolute()
    s=Supervisor(root,total_timeout);r=s.runner;selected=None;command=None
    try:
        from .diagnostics import prepare_delivery,recover_visible_devices
        prepare_delivery(r,s.deadline)
        for name,stage in [('filesystem','filesystem'),('system-info','system'),('environment','environment'),('nvml','nvml'),('driver','driver'),('runtime','runtime')]:
            step=s.case(name,{'kind':'deep','case':{'kind':'component','stage':stage}},seconds=45)
            if stage=='driver':driver=step
        s.case('topology',{'kind':'evidence'},seconds=30)
        count=s.case('discovery',{'kind':'deep','case':{'kind':'component','stage':'count'}},seconds=45)
        if verified(count):visible=count['result']['data']['visible_devices'];s.devices=devices or list(range(visible))
        else:s.devices=recover_visible_devices(driver,devices) or []
        if not s.devices:raise RuntimeError('NO_INDEPENDENT_CUDA_DEVICE_DISCOVERY')
        if verified(count) and any(d>=visible for d in s.devices):raise ValueError('Selected device not visible')
        r.report['devices']=s.devices
        identity=s.case('runtime-identity',{'kind':'identity'},seconds=60)
        if not verified(identity):raise RuntimeError('RUNTIME_IDENTITY_NOT_VERIFIED')
        base_policy={}
        healthy=s.health('initial-cuda-health')
        if not healthy and not getattr(r,'deep_cleanup_blocked',False) and verified(driver):
            # A startup loading contrast is safe only in a fresh, bounded process.
            s.blocked=False;base_policy={'CUDA_MODULE_LOADING':'LAZY','CUDA_MODULE_DATA_LOADING':'LAZY'}
            healthy=s.health('initial-cuda-health-lazy',base_policy)
        if healthy:
            attempts=[]
            for stage in ('hardware','meminfo','kernel'):
                test=s.case('all-gpus-'+stage,{'kind':'deep','case':{'kind':'component','stage':stage,'devices':s.devices,'seconds':30}},base_policy,seconds=60,gpu=True)
                if test['status'] in ('FAIL','TIMEOUT') and not s.after_failure('all-gpus-'+stage,base_policy):break
            for name,settings in POLICIES:
                policy=dict(base_policy,**settings)
                exchange=s.case('transport-'+name,{'kind':'exchange','devices':s.devices},policy,240,gpu=True)
                attempts.append({'name':name,'status':exchange['status']})
                if exchange['status']=='SKIPPED':continue
                if not verified(exchange):
                    if not s.after_failure('transport-'+name,policy):break
                    continue
                proof=s.case('accept-'+name,{'kind':'accept','devices':s.devices},policy,300,gpu=True)
                if acceptance(proof,s.devices):selected=(name,policy,proof);break
                if not s.after_failure('accept-'+name,policy):break
            r.report['transport_attempts']=attempts
            attempted={x['name'] for x in attempts}
            for candidate,_ in POLICIES:
                if candidate not in attempted:r.skip('transport-'+candidate,'WORKING_CANDIDATE_ALREADY_VERIFIED' if selected else 'CUDA_HEALTH_BLOCKED' if s.blocked else 'NOT_REACHED')
            if selected:
                name,policy,proof=selected
                # Independent API primitives concurrently on the full selected set.
                for module in ('EAGER','LAZY'):
                    for order in ('engine-first','context-first'):
                        label='api-'+module+'-'+order
                        cfg={'kind':'api','devices':s.devices,'module':module,'data_loading':module,'order':order,'control_thread':True}
                        step=s.case(label,{'kind':'deep','case':cfg},policy,60,gpu=True)
                        if step['status'] in ('FAIL','TIMEOUT') and not s.after_failure(label,policy):break
                # One distributed case per mode, never a full catalog per GPU.
                base={'kind':'bfs','fixture':'permutation','devices':s.devices}
                for label,changes in [('hash',{}),('sort',{'history':'sorted'}),('gemm',{'generator':'gemm'}),('parent',{'transport':'parent'}),('radix',{'candidate_order':'radix'}),('shards4',{'shards':4}),('repeated',{'fixture':'repeated'}),('matrix',{'fixture':'matrix'}),('matrix-gemm',{'fixture':'matrix-gemm','generator':'gemm'}),('wide',{'fixture':'wide'}),('shard-ab-hash',{'fixture':'lrx','backend':'shard_ab_hash'}),('shard-ab-sort',{'fixture':'lrx','backend':'shard_ab_sort_merge'})]:
                    step=s.case('mode-'+label,{'kind':'deep','case':dict(base,**changes)},policy,60,gpu=True)
                    if step['status'] in ('FAIL','TIMEOUT') and not s.after_failure('mode-'+label,policy):break
                if not s.blocked and s.health('final-cuda-health',policy):
                    _write(root/'recovery-proof.json',proof)
                    profile={'schema':1,'status':'VERIFIED','devices':s.devices,'policy':policy,'identity':identity['result']['data'],
                        'proof':{'path':'recovery-proof.json','sha256':hashlib.sha256((root/'recovery-proof.json').read_bytes()).hexdigest()},
                        'scope':'CPU oracle UT(4,3), measured tuning and bounded UT(4,47); no guarantee of full graph fit or optimal throughput'}
                    _write(root/'recovery-profile.json',profile)
                    check=s.case('public-launcher-check',{'kind':'check','profile':str(root/'recovery-profile.json')},seconds=300,gpu=True)
                    if acceptance(check,s.devices):
                        api=s.case('cayleypy-persistent-check',{'kind':'api-check','profile':str(root/'recovery-profile.json'),'devices':s.devices},seconds=180,gpu=True)
                        from .recovery_registry import install,api_accepted
                        if not api_accepted(api,s.devices):raise RuntimeError('CAYLEYPY_API_CHECK_FAILED')
                        # Validate automatic selection in an isolated cache and fresh process first.
                        staged=root/'auto-cache'
                        auto=s.case('cayleypy-auto-check',{'kind':'api-check','profile':str(root/'recovery-profile.json'),'devices':s.devices,'auto':True,'api_proof':api,'cache':str(staged)},seconds=180,gpu=True)
                        if not api_accepted(auto,s.devices):raise RuntimeError('CAYLEYPY_AUTO_CHECK_FAILED')
                        cached=install(root/'recovery-profile.json',auto)
                        _write(root/'api-proof.json',auto)
                        _write(root/'auto-registration.json',{'status':'REGISTERED','profile':str(cached)})
                        (root/'cayleypy-example.py').write_text('import multigpubfs\nfrom cayleypy import CayleyGraph, PermutationGroups\nmultigpubfs.enable_native()\nwith multigpubfs.persistent_graph_workers():\n    for n in (8, 9, 10):\n        result = CayleyGraph(PermutationGroups.coxeter(n), device="cpu").bfs()\n        print(n, result.bfs_completed, result.completed_depth)\n')
                        command=shlex.join([sys.executable,'-m','multigpubfs.recovery_diagnostics','--profile',str(root/'recovery-profile.json'),'--output',str(root.parent/(root.name+'-bfs'))])
                    else:s.after_failure('public-launcher-check',policy)
            elif not s.blocked:
                for a,b in itertools.combinations(s.devices,2):
                    pair=s.case('pair-'+str(a)+'-'+str(b),{'kind':'exchange','devices':[a,b]},POLICIES[0][1],120,gpu=True)
                    if pair['status'] in ('FAIL','TIMEOUT') and not s.after_failure('pair-'+str(a)+'-'+str(b),POLICIES[0][1]):break
                r.report['pair_scope']='Pair results localize links; no pair is acceptance of the complete GPU set.'
    except (Exception,KeyboardInterrupt) as error:
        r.report['steps'].append({'name':'recovery-controller','status':'FAIL','error':str(error),'traceback':traceback.format_exc()})
    if s.blocked or getattr(r,'deep_cleanup_blocked',False):command=None
    r.report['findings']=findings(root,r.report['steps']);_write(root/'findings.json',r.report['findings'])
    r.report['coverage']={'executed':[x['name'] for x in r.report['steps'] if x['status']!='SKIPPED'],'skipped':[{'name':x['name'],'reason':x.get('reason')} for x in r.report['steps'] if x['status']=='SKIPPED'],'not_claimed':['all possible graphs','all possible hardware faults','full UT(4,47) completion','multi-node jobs','optimal transport performance']}
    r.report['recovery']={'status':'READY' if command else 'BLOCKED' if s.blocked else 'NOT_VERIFIED','command':command,
        'selected':selected[0] if selected else None,'requested_devices':s.devices,
        'scope':'No diagnostic can prove absence of all bugs. Production is never automatically restarted.'}
    if command:(root/'launch-command.txt').write_text(command+'\n')
    else:
        (root/'recovery-profile.json').unlink(missing_ok=True)
        (root/'launch-command.txt').unlink(missing_ok=True)
    r.finish()
    summary=['RECOVERY '+r.report['recovery']['status'],r.report.get('recovery_blocked_reason',''),
        'The diagnostic FAIL status preserves observed failures even when a validated workaround is READY.',
        'Ready command: '+str(command) if command else 'No validated launch command. See first failed phase, rank logs and post-failure health.',
        'GPU resets, driver changes and production retries were not attempted.']
    for item in r.report['findings']:summary.append(item['code']+': '+item['action']+' Evidence: '+str(item['evidence']))
    (root/'recovery-summary.txt').write_text('\n'.join(summary)+'\n');package(root)
    print('\n'.join(summary),flush=True);print('BUNDLE '+str(root/'diagnostics.zip'),flush=True)
    return r.report

def enable_profile(profile_path,output_root='bfs-results'):
    """Explicitly activate the verified recipe through the CayleyPy API."""
    profile=read_profile(profile_path)
    os.environ.update(profile['policy'])
    from . import NativeOptions,enable_native
    return enable_native(NativeOptions(devices=tuple(profile['devices']),autotune=True,max_seconds=None,output_root=output_root), recovery=False)


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--child',type=Path);parser.add_argument('--profile',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--graph',type=Path);parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv)
    if args.child:
        root=args.child;config=json.loads((root/'case-input.json').read_text())
        try:
            if config['kind']=='api-check':
                from .recovery_api import check
                data=check(config,root)
            elif config['kind']=='check':
                profile=read_profile(config['profile']);os.environ.update(profile['policy'])
                data=experiment({'kind':'accept','devices':profile['devices']},root)
            else:data=experiment(config,root)
            _write(root/'result.json',{'status':'PASS','data':data});return 0
        except Exception as error:
            _write(root/'result.json',{'status':'FAIL','error':str(error),'traceback':traceback.format_exc()});traceback.print_exc();return 1
    if not args.profile or not args.output:parser.error('--profile and --output required')
    profile=read_profile(args.profile);os.environ.update(profile['policy'])
    if args.check:
        root=args.output.absolute();root.mkdir(parents=True,exist_ok=True)
        try:
            result=experiment({'kind':'accept','devices':profile['devices']},root)
            _write(root/'result.json',{'status':'PASS','data':result});return 0
        except Exception as error:
            _write(root/'result.json',{'status':'FAIL','error':str(error),'traceback':traceback.format_exc()});traceback.print_exc();return 1
    from . import GraphDefinition,run_graph,persistent_graph_workers
    graph=GraphDefinition.from_dict(json.loads(args.graph.read_text())) if args.graph else ut_graph(47)
    # Production runtime has no artificial time limit. A failure is retained, never retried.
    root=args.output.absolute();root.mkdir(parents=True,exist_ok=False)
    os.environ['MGBFS_TUNING_EVIDENCE_ROOT']=str(root/'tuning-evidence')
    (root/'session-live').mkdir(exist_ok=True)
    os.environ['MGBFS_GENERIC_SESSION_LOG_ROOT']=str(root/'session-live')
    os.environ.update(NCCL_DEBUG='INFO',NCCL_DEBUG_FILE=str(root/'nccl-%h-%p.log'),MGBFS_TRACE_NATIVE_CALLS='1')
    try:
        with persistent_graph_workers():run_graph(graph,root/'result',devices=profile['devices'],autotune=True,max_seconds=None)
        return 0
    except BaseException:
        (root/'failure.log').write_text(traceback.format_exc());raise
    finally:package(root);print('BUNDLE '+str(root/'diagnostics.zip'),flush=True)

if __name__=='__main__':raise SystemExit(main())
