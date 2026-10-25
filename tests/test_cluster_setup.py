import json
from pathlib import Path
import pytest
from multigpubfs import cluster_setup as c

def test_multinode_rejected():
 with pytest.raises(RuntimeError,match='MULTINODE'):
  c.check_allocation({'SLURM_JOB_NUM_NODES':'2'})
def test_multitask_rejected():
 with pytest.raises(RuntimeError,match='ONE_PARENT'):
  c.check_allocation({'SLURM_NTASKS':'8','SLURM_PROCID':'0'})
def test_single_parent_accepts_no_scheduler():
 assert c.check_allocation({})['scope']=='single-host'
def test_cpu_limit_recorded():
 assert c.check_allocation({'SLURM_CPUS_PER_TASK':'2'})['cpus_per_task']=='2'
def test_foreign_directory_not_modified(tmp_path):
 (tmp_path/'user.txt').write_text('keep')
 with pytest.raises(RuntimeError,match='EXISTING_ENVIRONMENT'):
  c.claim_environment(tmp_path,'dev')
 assert (tmp_path/'user.txt').read_text()=='keep'
def test_claim_is_reusable_only_for_same_schema(tmp_path):
 p=tmp_path/'managed'
 c.claim_environment(p,'dev')
 c.claim_environment(p,'dev')
 assert json.loads((p/'mgbfs-setup.json').read_text())['schema']==1
def test_existing_file_refused(tmp_path):
 p=tmp_path/'file';p.write_text('keep')
 with pytest.raises((OSError,RuntimeError)):c.claim_environment(p,'dev')
def test_ready_requires_registration_and_api():
 r={'recovery':{'status':'READY'},'devices':[0,1],'steps':[]}
 assert not c.accepted(r,True)
 r['steps']=[{'name':'cayleypy-auto-check','status':'PASS','cleanup_complete':True,'result':{'status':'PASS','data':{'api_verified':True,'persistent_reuse':True,'devices':[0,1]}}}]
 assert c.accepted(r,True)
 assert not c.accepted(r,False)
def test_no_shell_execution(monkeypatch,tmp_path):
 import inspect
 assert 'shell=True' not in inspect.getsource(c)
def test_commands_quote_paths():
 command=c.command(Path('/a b/python'),Path('/c d/example.py'))
 import shlex
 assert shlex.split(command)==['/a b/python','/c d/example.py']
def test_cli_dispatch(monkeypatch):
 from multigpubfs import cli
 monkeypatch.setattr(c,'main',lambda argv: 29 if argv==['--help'] else 1)
 assert cli.main(['setup','--help'])==29
def test_failure_report_archive(tmp_path):
 c.write_bundle(tmp_path,{'status':'BLOCKED','error':'NO_GPU'})
 import zipfile
 with zipfile.ZipFile(tmp_path/'setup.zip') as z:
  assert json.loads(z.read('setup.json'))['status']=='BLOCKED'
def test_lock_excludes_parallel_setup(tmp_path):
 with c.setup_lock(tmp_path):
  with pytest.raises(RuntimeError,match='SETUP_ALREADY_RUNNING'):
   with c.setup_lock(tmp_path):pass
 with c.setup_lock(tmp_path):pass

def test_cached_wheel_uses_no_network(monkeypatch,tmp_path):
 import hashlib
 name='multigpubfs-'+c.VERSION+'-py3-none-any.whl'
 (tmp_path/name).write_bytes(b'wheel')
 sha=hashlib.sha256(b'wheel').hexdigest()
 (tmp_path/'SHA256SUMS').write_text(sha+'  '+name+'\n')
 monkeypatch.setattr(c,'fetch',lambda *a:pytest.fail('cached setup used network'))
 assert c.get_wheel(tmp_path)==(tmp_path/name,sha)

def test_bad_cached_wheel_rejected_without_silent_repair(monkeypatch,tmp_path):
 name='multigpubfs-'+c.VERSION+'-py3-none-any.whl'
 (tmp_path/name).write_bytes(b'bad')
 (tmp_path/'SHA256SUMS').write_text('0'*64+'  '+name+'\n')
 monkeypatch.setattr(c,'fetch',lambda *a:pytest.fail('cache corruption used network'))
 with pytest.raises(RuntimeError,match='CHECKSUM'):c.get_wheel(tmp_path)
