PIN='f69871a9a1158380fb66846d88d9e9eb70a9b588'
import subprocess, pathlib, json
root=pathlib.Path('/tmp/mgbfs-production-transport-red');out=pathlib.Path('/kaggle/working');out.mkdir(exist_ok=True)
def run(cmd, **kw):return subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,**kw)
r=run(['git','clone','--quiet','https://github.com/TryDotAtwo/MultiGPUBFS.git',str(root)]);assert r.returncode==0,r.stdout
r=run(['git','checkout','--quiet',PIN],cwd=root);assert r.returncode==0,r.stdout
cargo=pathlib.Path.home()/'.cargo/bin/cargo'
if not cargo.exists():
 r=run(['bash','-lc','curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal']);assert r.returncode==0,r.stdout
red=run([str(cargo),'test','-p','mgbfs-core','--test','production_transport','--','--nocapture'],cwd=root,timeout=900);(out/'production-transport-red.log').write_text(red.stdout)
expected=red.returncode!=0 and 'unknown field `transport_backend`' in red.stdout and 'test result: FAILED' in red.stdout
full=run([str(cargo),'test','--workspace','--exclude','multigpubfs-gpu','--exclude','mgbfs-core'],cwd=root,timeout=900);(out/'cpu-workspace.log').write_text(full.stdout)
summary={'source_commit':PIN,'red_returncode':red.returncode,'expected_unknown_transport_field':expected,'cpu_workspace_exclusions':['multigpubfs-gpu requires CUDA build artifact','mgbfs-core carries deliberate RED test'],'cpu_workspace_returncode':full.returncode,'status':'RED_CONFIRMED_CPU_CHECKED' if expected and full.returncode==0 else 'FAILED'}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));assert expected and full.returncode==0
