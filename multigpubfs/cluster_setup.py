"""Prepare a user-owned environment and verify a single-host GPU allocation."""
import argparse
from contextlib import contextmanager
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

VERSION = '0.2.0.dev35'
RELEASE = 'https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261025-v35'
CAYLEYPY = 'cayleypy @ https://github.com/cayleypy/cayleypy/archive/3cd11ecefc98e54d6fc0ccdec967d0bcc8ca4a93.tar.gz'
TORCH = 'torch==2.9.0+cpu'

def check_allocation(env):
    for key in ('SLURM_JOB_NUM_NODES','SLURM_NNODES'):
        if int(env.get(key,'1')) != 1:
            raise RuntimeError('MULTINODE_NOT_SUPPORTED: allocate one node with all requested GPUs')
    if int(env.get('SLURM_NTASKS','1')) != 1 or int(env.get('SLURM_PROCID','0')) != 0:
        raise RuntimeError('ONE_PARENT_REQUIRED: use one scheduler task; the library starts its GPU ranks')
    return {'scope':'single-host','cpus_per_task':env.get('SLURM_CPUS_PER_TASK'),
            'visible_devices':env.get('CUDA_VISIBLE_DEVICES'),'scheduler':'slurm' if 'SLURM_JOB_ID' in env else 'unspecified'}

def claim_environment(root,version):
    root=Path(root)
    marker=root/'mgbfs-setup.json'
    if root.exists():
        if not marker.is_file():
            if not root.is_dir() or any(root.iterdir()):raise RuntimeError('EXISTING_ENVIRONMENT_NOT_OWNED: choose a new --environment directory')
        else:
            data=json.loads(marker.read_text())
            if data.get('schema')!=1 or data.get('owner')!='multigpubfs-setup' or data.get('version')!=version:
                raise RuntimeError('EXISTING_ENVIRONMENT_IDENTITY: choose a new --environment directory')
    root.mkdir(parents=True,exist_ok=True)
    marker.write_text(json.dumps({'schema':1,'owner':'multigpubfs-setup','version':version}))

@contextmanager
def setup_lock(root):
    lock=Path(root)/'.setup-lock'
    try:lock.mkdir()
    except FileExistsError:raise RuntimeError('SETUP_ALREADY_RUNNING: if a prior process was killed, verify it is gone before removing .setup-lock')
    try:yield
    finally:lock.rmdir()

def command(python,script):
    return shlex.join([str(python),str(script)])

def accepted(report,registered):
    from .recovery_registry import api_accepted
    return (registered and report.get('recovery',{}).get('status')=='READY'
            and any(s.get('name')=='cayleypy-auto-check' and api_accepted(s,report.get('devices'))
                    for s in report.get('steps',[])))

def write_bundle(root,report):
    root=Path(root);(root/'setup.json').write_text(json.dumps(report,indent=2))
    with zipfile.ZipFile(root/'setup.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ('setup.json','install.log','ready-command.txt','example.py','activation.sh'):
            p=root/name
            if p.is_file():
                with p.open('rb') as f:
                    f.seek(max(0,p.stat().st_size-(4<<20)));z.writestr(name,f.read(4<<20))
        for name in ('diagnostics','generated-api'):
            p=root/name/'diagnostics.zip'
            if p.is_file() and p.stat().st_size<=128<<20:z.write(p,name+'.zip')

def fetch(url,path,limit):
    with urllib.request.urlopen(url,timeout=60) as response,Path(path).open('wb') as output:
        count=0
        while True:
            block=response.read(1<<20)
            if not block:break
            count+=len(block)
            if count>limit:raise RuntimeError('DOWNLOAD_SIZE_LIMIT')
            output.write(block)

def get_wheel(root,local=None,sha=None):
    name='multigpubfs-'+VERSION+'-py3-none-any.whl'
    if local:
        path=Path(local).resolve()
        if not sha or path.name!=name:raise ValueError('Local --wheel requires matching release filename and --sha256')
        expected=sha.lower()
    else:
        Path(root).mkdir(parents=True,exist_ok=True)
        sums=Path(root)/'SHA256SUMS'
        if not sums.is_file():fetch(RELEASE+'/SHA256SUMS',sums,1<<20)
        entries=[line.split() for line in sums.read_text().splitlines()]
        matches=[a[0] for a in entries if len(a)==2 and a[1].lstrip('*')==name]
        if len(matches)!=1:raise RuntimeError('RELEASE_CHECKSUM_ENTRY')
        expected=matches[0];path=Path(root)/name
        if not path.is_file():fetch(RELEASE+'/'+name,path,16<<20)
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected:raise RuntimeError('WHEEL_CHECKSUM_MISMATCH')
    return path,actual

def run_step(args,log,deadline,env=None):
    remaining=deadline-time.monotonic()
    if remaining<=0:raise RuntimeError('INSTALL_BUDGET_EXHAUSTED')
    # Install children do not start GPU workers. GPU diagnosis owns its own cleanup.
    with log.open('a') as stream:
        stream.write('COMMAND '+shlex.join([str(x) for x in args])+'\n');stream.flush()
        subprocess.run([str(x) for x in args],stdout=stream,stderr=subprocess.STDOUT,
                       check=True,timeout=remaining,env=env)

def main(argv=None):
    parser=argparse.ArgumentParser(description='Prepare an isolated CayleyPy environment and verify all visible GPUs on one allocated node.')
    parser.add_argument('--environment',type=Path,default=Path.home()/'.local/share/multigpubfs/environments'/VERSION)
    parser.add_argument('--cache',type=Path,default=Path.home()/'.cache/multigpubfs')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--devices',help='Logical IDs; default all visible GPUs')
    parser.add_argument('--prefetch-only',action='store_true',help='Prepare without GPUs; this does not certify a compute node')
    parser.add_argument('--sm',default='all',help='Prefetch architectures, comma separated; default all (prefetch-only)')
    parser.add_argument('--total-timeout',type=float,default=1800,help='Diagnostic execution budget; production BFS stays unlimited')
    parser.add_argument('--install-timeout',type=float,default=1200)
    parser.add_argument('--wheel',type=Path,help='Pre-downloaded matching wheel')
    parser.add_argument('--sha256',help='Required with --wheel')
    args=parser.parse_args(argv)
    if sys.platform!='linux':parser.error('Linux x86_64 NVIDIA compute environment required')
    if any(not math.isfinite(v) or v<=0 for v in (args.total_timeout,args.install_timeout)):parser.error('Positive finite budgets required')
    root=(args.output or Path('mgbfs-setup-'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'-'+str(os.getpid()))).absolute()
    root.mkdir(parents=True,exist_ok=False)
    report={'schema':1,'status':'PREPARING','package':VERSION,'scope':'single-host GPU allocation; no scheduler submission or driver changes'}
    try:
        report['allocation']=check_allocation(os.environ)
        if sys.prefix==sys.base_prefix and not shutil_available_venv():
            raise RuntimeError('PYTHON_VENV_UNAVAILABLE: install the cluster Python venv module or use a Python distribution with venv')
        target=args.environment.expanduser().absolute();cache=args.cache.expanduser().absolute()
        claim_environment(target,VERSION)
        with setup_lock(target):
            deadline=time.monotonic()+args.install_timeout;log=root/'install.log'
            py=target/'venv/bin/python'
            if not py.exists():run_step([sys.executable,'-m','venv',target/'venv'],log,deadline)
            wheel,sha=get_wheel(target/'wheels',args.wheel,args.sha256);report['wheel_sha256']=sha
            env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
            env.update(MGBFS_RECOVERY_CACHE=str(cache/'recovery'),MGBFS_NATIVE_CACHE=str(cache/'native'),PYTHONUNBUFFERED='1')
            stamp=target/'installed.json'
            wanted={'wheel_sha256':sha,'torch':TORCH,'cayleypy':CAYLEYPY}
            reuse=stamp.is_file() and json.loads(stamp.read_text())==wanted
            if not reuse:
                print('SETUP installing isolated Python dependencies',flush=True)
                run_step([py,'-m','pip','install','--upgrade','pip','setuptools','wheel'],log,deadline,env)
                run_step([py,'-m','pip','install',TORCH,'--index-url','https://download.pytorch.org/whl/cpu'],log,deadline,env)
                run_step([py,'-m','pip','install',CAYLEYPY,str(wheel)],log,deadline,env)
            run_step([py,'-m','pip','check'],log,deadline,env)
            run_step([py,'-c','import cayleypy,multigpubfs; assert callable(multigpubfs.enable_native)'],log,deadline,env)
            stamp.write_text(json.dumps(wanted));report['environment_reused']=reuse
            activate='unset PYTHONPATH PYTHONHOME\nexport MGBFS_RECOVERY_CACHE='+shlex.quote(str(cache/'recovery'))+'\nexport MGBFS_NATIVE_CACHE='+shlex.quote(str(cache/'native'))+'\nexport PATH='+shlex.quote(str(py.parent))+':"$PATH"\n'
            (root/'activation.sh').write_text(activate)
            if args.prefetch_only:
                cmd=[py,'-m','multigpubfs','prefetch']
                cmd+=['--all'] if args.sm=='all' else ['--sm',args.sm]
                run_step(cmd,log,deadline,env)
                report['status']='PREFETCHED_NOT_GPU_VERIFIED'
                print('PREFETCHED: run setup inside the allocated GPU job to verify this node',flush=True)
            else:
                print('SETUP testing all selected GPUs; diagnostic budget '+str(args.total_timeout)+' seconds',flush=True)
                # Job-local writable scratch; report/cache stay in explicitly selected stable paths.
                with tempfile.TemporaryDirectory(prefix='mgbfs-setup-',dir='/tmp') as scratch:
                    env['TMPDIR']=scratch
                    cmd=[str(py),'-m','multigpubfs','doctor','--recover','--total-timeout',str(args.total_timeout),'--output',str(root/'diagnostics')]
                    if args.devices:cmd+=['--devices',args.devices]
                    result=subprocess.run(cmd,env=env,cwd=root)
                env.pop('TMPDIR',None)
                diagnostic=json.loads((root/'diagnostics/report.json').read_text())
                registered=(root/'diagnostics/auto-registration.json').is_file()
                if result.returncode!=0 or not accepted(diagnostic,registered):
                    raise RuntimeError('CLUSTER_NOT_VERIFIED: see diagnostics/recovery-summary.txt and setup.zip')
                example=root/'example.py'
                example.write_text('import multigpubfs\nfrom cayleypy import CayleyGraph, PermutationGroups\nmultigpubfs.enable_native()\nwith multigpubfs.persistent_graph_workers():\n    for n in (3, 4, 3):\n        r = CayleyGraph(PermutationGroups.coxeter(n), device="cpu").bfs()\n        print(n, r.bfs_completed, r.layer_sizes)\n')
                # A separate public API process verifies the actual user's generated example.
                from .diagnostics import Runner
                verifier=Runner(root/'generated-api',120);verifier.deep_capture=True
                step=verifier.run('example',[str(py),str(example)],environment=env)
                verifier.finish()
                cleanup=root/'generated-api/example/cleanup.json'
                if step['status']!='PASS' or not cleanup.is_file() or not json.loads(cleanup.read_text()).get('complete'):
                    raise RuntimeError('GENERATED_API_EXAMPLE_FAILED')
                report.update(status='READY',devices=diagnostic['devices'],generated_example_verified=True)
                ready='source '+shlex.quote(str(root/'activation.sh'))+'\n'+command(py,example)+'\n'
                (root/'ready-command.txt').write_text(ready);print('READY\n'+ready,flush=True)
        report['environment']=str(target);report['cache']=str(cache)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='BLOCKED',error=str(error),exception_type=type(error).__name__)
        print('BLOCKED: '+str(error),file=sys.stderr,flush=True)
    finally:
        write_bundle(root,report);print('BUNDLE '+str(root/'setup.zip'),flush=True)
    return 0 if report['status'] in ('READY','PREFETCHED_NOT_GPU_VERIFIED') else 1

def shutil_available_venv():
    import importlib.util
    return importlib.util.find_spec('venv') is not None
