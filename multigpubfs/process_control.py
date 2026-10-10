"""Bounded cold native queries; never a BFS completion or restart signal."""
import math,os,subprocess

def query(command,*,env=None,**kwargs):
 try:seconds=float((env if env is not None else os.environ).get('MGBFS_NATIVE_QUERY_SECONDS','60'))
 except (TypeError,ValueError):raise ValueError('INVALID_NATIVE_QUERY_TIMEOUT')
 if not math.isfinite(seconds) or seconds<=0:raise ValueError('INVALID_NATIVE_QUERY_TIMEOUT')
 from .generic_session import query as resident_query
 try:
  response=resident_query(command,env if env is not None else dict(os.environ),seconds,kwargs)
  return response if response is not None else subprocess.run(command,env=env,timeout=seconds,**kwargs)
 except subprocess.TimeoutExpired as error:
  raise RuntimeError('NATIVE_STARTUP_QUERY_TIMEOUT: child terminated; no completed result and no restart') from error

def run(command,*,timeout,env=None,**kwargs):
 try:return subprocess.run(command,env=env,timeout=timeout,**kwargs)
 except subprocess.TimeoutExpired as error:
  raise RuntimeError('NATIVE_EXECUTION_TIMEOUT: child terminated; no completed result and no restart') from error
