PIN='35fd2a82ba2e3e825f2b459b1b7a181db92e67c8'
import subprocess,pathlib,json
root=pathlib.Path('/tmp/mgbfs-capacity-evidence-red');out=pathlib.Path('/kaggle/working')
subprocess.run(['git','clone','--quiet','https://github.com/TryDotAtwo/MultiGPUBFS.git',str(root)],check=True)
subprocess.run(['git','checkout','--quiet',PIN],cwd=root,check=True)
r=subprocess.run(['python','-m','unittest','discover','-s','scripts','-p','test_capacity_fault_reached.py','-v'],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
(out/'capacity-red.log').write_text(r.stdout)
expected=r.returncode!=0 and 'FAIL: test_armed_ordinary_nccl_capacity_abort_is_reached' in r.stdout and 'FAIL: test_other_rank_arming_does_not_validate_requested_injection' in r.stdout
summary={'source':PIN,'returncode':r.returncode,'expected_failures':expected,'status':'RED_CONFIRMED' if expected else 'UNEXPECTED'}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));assert expected
