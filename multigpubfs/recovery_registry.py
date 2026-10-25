"""User-local, verified recovery recipes. No GPU probing at import time."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import warnings

_applied = None

def cache_root():
    return Path(os.environ.get('MGBFS_RECOVERY_CACHE', str(Path(os.environ.get('XDG_CACHE_HOME', str(Path.home()/'.cache')))/'multigpubfs'/'recovery'))).expanduser().absolute()

def key(identity):
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()

def api_accepted(step, devices):
    data=step.get('result',{}).get('data',{})
    return (step.get('status')=='PASS' and step.get('cleanup_complete') is True
            and step.get('result',{}).get('status')=='PASS' and data.get('devices')==devices
            and data.get('api_verified') is True and data.get('persistent_reuse') is True)

def install(path,api_step):
    from .recovery_diagnostics import read_profile
    path=Path(path).absolute();profile=read_profile(path)
    allowed={'NCCL_NET','NCCL_IB_DISABLE','NCCL_NET_GDR_LEVEL','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE','CUDA_MODULE_LOADING','CUDA_MODULE_DATA_LOADING'}
    if not set(profile['policy'])<=allowed or not all(isinstance(v,str) for v in profile['policy'].values()):raise RuntimeError('RECOVERY_POLICY_NOT_ALLOWED')
    if not api_accepted(api_step,profile['devices']):raise RuntimeError('RECOVERY_API_NOT_VERIFIED')
    root=cache_root();root.mkdir(parents=True,exist_ok=True)
    name=key(profile['identity']);folder=Path(tempfile.mkdtemp(prefix='.pending-',dir=root))
    try:
        for filename,value in [('recovery-profile.json',profile),('api-proof.json',api_step)]:
            (folder/filename).write_text(json.dumps(value,sort_keys=True))
        shutil.copyfile(path.parent/profile['proof']['path'],folder/'recovery-proof.json')
        profile['proof']['path']='recovery-proof.json'
        (folder/'recovery-profile.json').write_text(json.dumps(profile,sort_keys=True))
        manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in folder.iterdir()}
        (folder/'manifest.json').write_text(json.dumps(manifest,sort_keys=True))
        # Unique immutable bundles; atomic pointer replacement is the publication step.
        target=root/(name+'-'+folder.name.removeprefix('.pending-'))
        folder.rename(target)
        fd,tmp=tempfile.mkstemp(prefix='.pointer-',dir=root)
        with os.fdopen(fd,'w') as out:json.dump({'bundle':target.name},out)
        os.replace(tmp,root/(name+'.json'))
        return target/'recovery-profile.json'
    finally:
        if folder.exists():shutil.rmtree(folder)

def restore():
    global _applied
    if _applied:
        for k,v in _applied['policy'].items():
            if os.environ.get(k)==v:
                old=_applied['original'][k]
                if old is None:os.environ.pop(k,None)
                else:os.environ[k]=old
    _applied=None

def activate(options):
    global _applied
    if os.environ.get('MGBFS_RECOVERY_AUTO')=='0':
        opt_out();return options
    from .recovery_diagnostics import fingerprint,read_profile
    from . import generic_session
    if _applied:
        if fingerprint()!=_applied['effective']:raise RuntimeError('RECOVERY_ACTIVE_ENV_CHANGED: disable_native and rerun doctor')
        if options.devices is not None and tuple(options.devices)!=tuple(_applied['devices']):raise RuntimeError('RECOVERY_ACTIVE_DEVICES_CHANGED')
        return replace(options,devices=tuple(_applied['devices']))
    root=cache_root()
    if not root.is_dir() or not any(root.glob('*.json')):return options
    identity=fingerprint();pointer=root/(key(identity)+'.json')
    if not pointer.is_file():
        warnings.warn('No matching recovery profile for this node/package/environment; ordinary activation continues. Run doctor --recover to verify this job.',RuntimeWarning)
        return options
    try:
        entry=json.loads(pointer.read_text());name=entry['bundle']
        if Path(name).name!=name:raise RuntimeError('RECOVERY_CACHE_PATH')
        folder=root/name
        if folder.is_symlink():raise RuntimeError('RECOVERY_CACHE_SYMLINK')
        manifest=json.loads((folder/'manifest.json').read_text())
        if set(manifest)!={'recovery-profile.json','recovery-proof.json','api-proof.json'}:raise RuntimeError('RECOVERY_CACHE_MANIFEST')
        for filename,sha in manifest.items():
            f=folder/filename
            if f.is_symlink() or hashlib.sha256(f.read_bytes()).hexdigest()!=sha:raise RuntimeError('RECOVERY_CACHE_CHECKSUM')
        profile=read_profile(folder/'recovery-profile.json')
        if not api_accepted(json.loads((folder/'api-proof.json').read_text()),profile['devices']):raise RuntimeError('RECOVERY_CACHE_API_PROOF')
    except (OSError,ValueError,KeyError,RuntimeError) as error:
        warnings.warn('Ignoring invalid recovery profile: '+str(error),RuntimeWarning);return options
    if options.devices is not None and tuple(options.devices)!=tuple(profile['devices']):
        warnings.warn('Recovery profile devices differ from explicit devices; ordinary activation continues.',RuntimeWarning);return options
    if generic_session._active is not None and getattr(generic_session._active,'workers',True):raise RuntimeError('RECOVERY_WORKERS_ALREADY_STARTED: enable_native before persistent workers')
    original={k:os.environ.get(k) for k in profile['policy']}
    os.environ.update(profile['policy'])
    try:effective=fingerprint()
    except BaseException:
        for k,v in original.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
        raise
    _applied={'policy':profile['policy'],'original':original,'effective':effective,'devices':profile['devices'],'profile':str(folder/'recovery-profile.json')}
    print('MultiGPUBFS: verified recovery profile '+_applied['profile'],flush=True)
    return replace(options,devices=tuple(profile['devices']))


def bind_options(options):
    if _applied:
        if options.devices is not None and tuple(options.devices)!=tuple(_applied['devices']):raise RuntimeError('RECOVERY_ACTIVE_DEVICES_CHANGED: disable_native before selecting other GPUs')
        return replace(options,devices=tuple(_applied['devices']))
    return options


def opt_out():
    from . import generic_session
    if _applied and generic_session._active is not None and getattr(generic_session._active,'workers',True):raise RuntimeError('RECOVERY_WORKERS_ALREADY_STARTED: close workers before opting out')
    restore()
