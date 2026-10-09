PIN='cb94f45aa23f971d49eaca9c2cf6102331c05e84'
import subprocess, pathlib, json
root=pathlib.Path('/tmp/mgbfs-production-transport-green');out=pathlib.Path('/kaggle/working');out.mkdir(exist_ok=True)
def run(cmd, **kw):return subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,**kw)
r=run(['git','clone','--quiet','https://github.com/TryDotAtwo/MultiGPUBFS.git',str(root)]);assert r.returncode==0,r.stdout
r=run(['git','checkout','--quiet',PIN],cwd=root);assert r.returncode==0,r.stdout
cargo=pathlib.Path.home()/'.cargo/bin/cargo'
if not cargo.exists():
 r=run(['bash','-lc','curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal']);assert r.returncode==0,r.stdout
full=run([str(cargo),'test','--workspace','--exclude','multigpubfs-gpu'],cwd=root,timeout=1200);(out/'cpu-workspace.log').write_text(full.stdout)
summary={'source_commit':PIN,'cpu_workspace_exclusions':['multigpubfs-gpu requires CUDA build artifact'],'cpu_workspace_returncode':full.returncode,'status':'CPU_GREEN' if full.returncode==0 else 'FAILED'}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));assert full.returncode==0,full.stdout[-8000:]
