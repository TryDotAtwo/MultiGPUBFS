"""Bounded cold native queries; never a BFS completion or restart signal."""
import math,os,subprocess

def startup_timeout(env=None):
 try:seconds=float((env if env is not None else os.environ).get('MGBFS_NATIVE_QUERY_SECONDS','60'))
 except (TypeError,ValueError):raise ValueError('INVALID_NATIVE_QUERY_TIMEOUT')
 if not math.isfinite(seconds) or seconds<=0:raise ValueError('INVALID_NATIVE_QUERY_TIMEOUT')
 return seconds

def startup_timeout_error(error):
 scope='persistent worker wait expired; worker termination is not implied' if error.cmd=='generic resident job' else 'standalone query child terminated by subprocess timeout'
 return RuntimeError('NATIVE_STARTUP_QUERY_TIMEOUT: '+scope+'; no completed result and no restart; command='+repr(error.cmd))

def query(command,*,env=None,**kwargs):
 seconds=startup_timeout(env)
 from .generic_session import query as resident_query
 try:
  response=resident_query(command,env if env is not None else dict(os.environ),seconds,kwargs)
  return response if response is not None else subprocess.run(command,env=env,timeout=seconds,**kwargs)
 except subprocess.TimeoutExpired as error:
  raise startup_timeout_error(error) from error

def run(command,*,timeout,env=None,**kwargs):
 try:return subprocess.run(command,env=env,timeout=timeout,**kwargs)
 except subprocess.TimeoutExpired as error:
  raise RuntimeError('NATIVE_EXECUTION_TIMEOUT: child terminated; no completed result and no restart') from error
