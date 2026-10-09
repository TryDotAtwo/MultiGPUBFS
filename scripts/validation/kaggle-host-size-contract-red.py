import json, os, subprocess, pathlib, shutil, time
out=pathlib.Path('/kaggle/working/host-size-contract');out.mkdir(exist_ok=True)
source='9344f7f94480de4d9f21df8025069cc77bf4dca8'
summary={'source':source,'status':'STARTED','scope':'remote CPU allocation RED gate, not GPU lifetime acceptance'}
def run(cmd,name,env=None):
    p=subprocess.run(cmd,cwd='/tmp',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600)
    (out/(name+'.log')).write_text(p.stdout);print(name,p.returncode,p.stdout[-2500:],flush=True)
    return p
try:
    env=os.environ.copy();env['CARGO_HOME']='/tmp/mgbfs-cargo';env['RUSTUP_HOME']='/tmp/mgbfs-rustup'
    env['PATH']='/tmp/mgbfs-cargo/bin:'+env['PATH']
    p=run(['bash','-lc','curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.90.0'],'rust',env);assert p.returncode==0
    p=run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/TryDotAtwo/MultiGPUBFS.git','/tmp/host-size-source'],'clone',env);assert p.returncode==0
    p=run(['git','-C','/tmp/host-size-source','sparse-checkout','set','--cone','crates','rust'],'sparse',env);assert p.returncode==0
    p=run(['git','-C','/tmp/host-size-source','checkout',source],'checkout',env);assert p.returncode==0
    p=run(['git','-C','/tmp/host-size-source','restore','--source='+source,'--worktree','--ignore-skip-worktree-bits','--','Cargo.toml','Cargo.lock'],'workspace-roots',env);assert p.returncode==0
    p=run(['git','-C','/tmp/host-size-source','ls-files','--stage','Cargo.toml','Cargo.lock'],'workspace-index',env);assert p.returncode==0
    assert pathlib.Path('/tmp/host-size-source/Cargo.toml').is_file(), 'restored workspace manifest absent'
    cmd=['cargo','test','--manifest-path','/tmp/host-size-source/Cargo.toml','--locked','-p','mgbfs-runtime','--test','distributed_memory','host_size_handshake_has_storage_disjoint_from_live_owner_count','--','--nocapture']
    p=run(cmd,'red',env)
    assert p.returncode!=0 and 'no independent handshake word' in p.stdout and 'test result: FAILED' in p.stdout, 'not expected allocation RED'
    summary['status']='EXPECTED_RED';summary['returncode']=p.returncode
except Exception as e:
    summary['status']='ERROR';summary['error']=str(e)
finally:
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
