PIN='45cb861364d32ac090268bdffa93c10f5d37cc44'
import subprocess, pathlib, json
root=pathlib.Path('/tmp/mgbfs-native-rank-red')
out=pathlib.Path('/kaggle/working');out.mkdir(exist_ok=True)
def run(cmd, **kw):
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)
r=run(['git','clone','--quiet','https://github.com/TryDotAtwo/MultiGPUBFS.git',str(root)]);assert r.returncode==0,r.stdout
r=run(['git','checkout','--quiet',PIN],cwd=root);assert r.returncode==0,r.stdout
cargo=pathlib.Path.home()/'.cargo/bin/cargo'
if not cargo.exists():
    r=run(['bash','-lc','curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal']);assert r.returncode==0,r.stdout
r=run([str(cargo),'test','-p','mgbfs-runtime','--test','native_rank_transport','--','--nocapture'],cwd=root,timeout=900)
(out/'red-test.log').write_text(r.stdout)
expected=r.returncode!=0 and 'missing host-sized receive plane: recv_states' in r.stdout and 'test result: FAILED' in r.stdout
summary={'source_commit':PIN,'returncode':r.returncode,'expected_contract_failure':expected,'status':'RED_CONFIRMED' if expected else 'UNEXPECTED_RESULT'}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));assert expected,r.stdout[-4000:]
