"""Bounded cold native queries; never a BFS completion or restart signal."""
import json,math,os,subprocess,time
from contextlib import contextmanager
from contextvars import ContextVar

_profile_deadline = ContextVar("profile_deadline", default=None)


class ProfileBudgetExpired(RuntimeError):
 pass


@contextmanager
def profile_budget(deadline):
 token=_profile_deadline.set(deadline)
 try:yield
 finally:_profile_deadline.reset(token)


def profile_timeout(seconds):
 deadline=_profile_deadline.get()
 if deadline is None:return seconds
 left=deadline-time.monotonic()
 if left<=0:raise ProfileBudgetExpired("AUTOTUNE_STARTUP_DEADLINE")
 return min(seconds,left)

class NativeGraphAdmissionError(RuntimeError):
 def __init__(self, message, native_code=None):
  super().__init__(message)
  self.native_code=native_code

def check_graph_admission(process):
 if not process.returncode:return
 # The standalone CLI writes the terminal record to stderr. Accept the
 # equivalent structured response from wrappers that capture it in stdout.
 stderr=(process.stderr or '')[-4000:]
 stdout=(process.stdout or '')[-4000:]
 for line in reversed((stdout+'\n'+stderr).splitlines()):
  if not line.startswith('{'):continue
  try:record=json.loads(line)
  except ValueError:continue
  if isinstance(record,dict) and record.get('status')=='ERROR' and isinstance(record.get('error'),str):
   raise NativeGraphAdmissionError('NATIVE_GRAPH_ADMISSION_FAILED: '+record['error'],record['error'])
 raise NativeGraphAdmissionError('NATIVE_GRAPH_ADMISSION_FAILED: '+(stderr or stdout))

def startup_timeout(env=None):
 try:seconds=float((env if env is not None else os.environ).get('MGBFS_NATIVE_QUERY_SECONDS','60'))
 except (TypeError,ValueError):raise ValueError('INVALID_NATIVE_QUERY_TIMEOUT')
 if not math.isfinite(seconds) or seconds<=0:raise ValueError('INVALID_NATIVE_QUERY_TIMEOUT')
 return seconds

def startup_timeout_error(error):
 scope='persistent worker wait expired; worker termination is not implied' if error.cmd=='generic resident job' else 'standalone query child terminated by subprocess timeout'
 return RuntimeError('NATIVE_STARTUP_QUERY_TIMEOUT: '+scope+'; no completed result and no restart; command='+repr(error.cmd))

def query(command,*,env=None,resident=True,**kwargs):
 normal_timeout=startup_timeout(env)
 seconds=profile_timeout(normal_timeout)
 from .generic_session import query as resident_query
 try:
  response=resident_query(command,env if env is not None else dict(os.environ),seconds,kwargs) if resident else None
  return response if response is not None else subprocess.run(command,env=env,timeout=seconds,**kwargs)
 except subprocess.TimeoutExpired as error:
  if seconds<normal_timeout and error.cmd!='generic resident job':
   raise ProfileBudgetExpired("AUTOTUNE_STARTUP_DEADLINE") from error
  raise startup_timeout_error(error) from error

def run(command,*,timeout,env=None,**kwargs):
 try:return subprocess.run(command,env=env,timeout=timeout,**kwargs)
 except subprocess.TimeoutExpired as error:
  raise RuntimeError('NATIVE_EXECUTION_TIMEOUT: child terminated; no completed result and no restart') from error
