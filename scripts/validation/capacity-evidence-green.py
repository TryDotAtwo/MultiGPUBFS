PIN='f8b92f8005cdbff176f579bb475af46f7e2d1269'
import subprocess,pathlib,json
root=pathlib.Path('/tmp/mgbfs-capacity-evidence-red');out=pathlib.Path('/kaggle/working')
subprocess.run(['git','clone','--quiet','https://github.com/TryDotAtwo/MultiGPUBFS.git',str(root)],check=True)
subprocess.run(['git','checkout','--quiet',PIN],cwd=root,check=True)
r=subprocess.run(['python','-m','unittest','discover','-s','scripts','-p','test_capacity_fault_reached.py','-v'],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
(out/'capacity-red.log').write_text(r.stdout)
passed=r.returncode==0 and 'Ran 4 tests' in r.stdout
summary={'source':PIN,'returncode':r.returncode,'status':'GREEN_CONFIRMED' if passed else 'FAILED'}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));assert passed
