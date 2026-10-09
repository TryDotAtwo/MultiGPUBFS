PIN='aa0d5ad659a9be7db0c603c716c3405db97b1f1a'
import subprocess, pathlib, json
root=pathlib.Path('/tmp/mgbfs-native-rank-green')
out=pathlib.Path('/kaggle/working');out.mkdir(exist_ok=True)
def run(cmd, **kw):
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)
r=run(['git','clone','--quiet','https://github.com/TryDotAtwo/MultiGPUBFS.git',str(root)]);assert r.returncode==0,r.stdout
r=run(['git','checkout','--quiet',PIN],cwd=root);assert r.returncode==0,r.stdout
cargo=pathlib.Path.home()/'.cargo/bin/cargo'
if not cargo.exists():
    r=run(['bash','-lc','curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal']);assert r.returncode==0,r.stdout
r=run([str(cargo),'test','-p','mgbfs-runtime','--test','native_rank_transport','--','--nocapture'],cwd=root,timeout=900)
(out/'contract-test.log').write_text(r.stdout)
full=run([str(cargo),'test','--workspace'],cwd=root,timeout=900)
(out/'workspace-test.log').write_text(full.stdout)
summary={'source_commit':PIN,'contract_returncode':r.returncode,'workspace_returncode':full.returncode,'status':'CPU_GREEN' if r.returncode==0 and full.returncode==0 else 'FAILED'}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));assert r.returncode==0 and full.returncode==0
