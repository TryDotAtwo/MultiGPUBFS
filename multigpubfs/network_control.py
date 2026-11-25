"""Launch and receipt rendezvous with bounded host-side window replies."""
import json,threading,time,urllib.request,urllib.error,hmac
from http.client import IncompleteRead
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
_MAX_BLOCK = 1 << 20

def _deadline_remaining(deadline):
 remaining=deadline-time.perf_counter()
 if remaining<=0:raise TimeoutError('DISTRIBUTED_CONTROL_JOB_DEADLINE')
 return remaining

def _pause_retry(deadline):
 time.sleep(.05 if deadline is None else min(.05,_deadline_remaining(deadline)))

def _job_sequence(key):
    parts = key.split("/", 2)
    if len(parts) != 3 or parts[0] != "job":
        return None
    sequence = parts[1]
    return int(sequence) if len(sequence) <= 20 and sequence.isascii() and sequence.isdecimal() else None

class ControlStore:
 def __init__(self,host,port,rank,session,timeout,token=''):
  self.url=f'http://{host}:{port}';self.rank=rank;self.session=session;self.timeout=timeout;self.token=token;self.server=None
  if rank==0:
   values={};blocks={};acknowledged={};lock=threading.Condition();expected=session;secret=token
   retired_job=-1
   class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def block_request(self):
     if self.headers.get('X-MGBFS-Session')!=expected:self.send_error(409);return
     key=self.headers.get('X-MGBFS-Key')
     if not key:self.send_error(400);return
     try:index=int(self.headers.get('X-MGBFS-Block','-1'))
     except ValueError:self.send_error(400);return
     if index<0:self.send_error(400);return
     data=b''
     job=_job_sequence(key)
     if self.path=='/put-block':
      try:size=int(self.headers.get('Content-Length','0'))
      except ValueError:self.send_error(400);return
      if not 0<size<=_MAX_BLOCK:self.send_error(413);return
      payload=self.rfile.read(size)
      if len(payload)!=size:self.send_error(400);return
      with lock:
       next_index=acknowledged.get(key,0)
       if (job is not None and job<=retired_job) or index<next_index:status=200
       elif index==next_index and blocks.get(key,payload)==payload:
        blocks[key]=payload
        lock.notify_all()
        status=200
       else:status=409
     elif self.path=='/ack-block':
      with lock:
       next_index=acknowledged.get(key,0)
       if (job is not None and job<=retired_job) or index<next_index:status=200
       elif index==next_index and key in blocks:
        del blocks[key]
        acknowledged[key]=index+1
        values.pop(self.headers.get('X-MGBFS-Command'),None)
        lock.notify_all()
        status=200
       else:status=409
     else:
      def ready():
       next_index=acknowledged.get(key,0)
       return ('error' in values or (job is not None and job<=retired_job)
               or index<next_index or (index==next_index and key in blocks))
      with lock:
       lock.wait_for(ready,timeout=.5)
       next_index=acknowledged.get(key,0)
       if 'error' in values:status=424
       elif (job is not None and job<=retired_job) or index<next_index:status=409
       elif index==next_index and key in blocks:status=200;data=blocks[key]
       else:status=204
     self.send_response(status);self.send_header('Content-Length',str(len(data)))
     self.end_headers()
     if data:self.wfile.write(data)
    def do_POST(self):
     nonlocal retired_job
     if not hmac.compare_digest(self.headers.get('X-MGBFS-Token',''),secret):self.send_error(403);return
     if self.path in ('/put-block','/get-block','/ack-block'):self.block_request();return
     size=int(self.headers.get('Content-Length','0'))
     if not 0<size<=64<<20:self.send_error(413);return
     try:
      body=json.loads(self.rfile.read(size))
      if body.get('session')!=expected:self.send_error(409);return
      key=body['key']
      if not isinstance(key,str):self.send_error(400);return
      job=_job_sequence(key)
      with lock:
       if self.path=='/put':
        if key=='error':values.setdefault(key,body['value'])
        elif job is not None and job<=retired_job:pass
        elif key in values and values[key]!=body['value']:self.send_error(409);return
        else:values[key]=body['value']
        result={'present':True}
        lock.notify_all()
       elif self.path=='/delete-prefix':
        # A request retried before this retirement may still reach a handler.
        # Keep one watermark for the serial cluster jobs rather than their old replies.
        if job is not None and key==f'job/{job}/':retired_job=max(retired_job,job)
        for old in list(values):
         if old.startswith(key):del values[old]
        for registry in (blocks,acknowledged):
         for old in list(registry):
          if old.startswith(key):del registry[old]
        result={'present':True}
        lock.notify_all()
       elif self.path=='/collect-ranks':
        count=body['value']
        if type(count) is not int or not 1<=count<=128:self.send_error(400);return
        keys=[f'{key}{rank}' for rank in range(count)]
        complete=all(item in values for item in keys)
        result={'present':complete,'value':[values[item] for item in keys] if complete else None,'error':values.get('error')}
       elif self.path=='/await':
        stop=body['value']
        lock.wait_for(lambda:key in values or stop in values or 'error' in values,timeout=.5)
        done=stop in values
        result={'present':done or key in values,'value':None if done else values.get(key),'error':values.get('error')}
       else:
        result={'present':key in values,'value':values.get(key),'error':values.get('error')}
      raw=json.dumps(result).encode();self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
     except (ValueError,KeyError,TypeError):self.send_error(400)
   self.server=ThreadingHTTPServer(('0.0.0.0',port),Handler);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
 def _request(self,path,key,value=None,*,timeout=3):
  raw=json.dumps({'session':self.session,'key':key,'value':value}).encode();request=urllib.request.Request(self.url+path,data=raw,headers={'Content-Type':'application/json','X-MGBFS-Token':self.token})
  with urllib.request.urlopen(request,timeout=timeout) as response:return json.load(response)
 def _block(self,path,key,index,data=b'',*,command_key=None,deadline=None):
  end=time.monotonic()+self.timeout
  headers={'X-MGBFS-Token':self.token,'X-MGBFS-Session':self.session,'X-MGBFS-Key':key,'X-MGBFS-Block':str(index)}
  if command_key is not None:headers['X-MGBFS-Command']=command_key
  request=urllib.request.Request(self.url+path,data=data,headers=headers)
  while True:
   timeout=self._request_timeout(deadline)
   try:
    with urllib.request.urlopen(request,timeout=timeout) as response:
     if response.status==200:
      size=response.headers.get('Content-Length','')
      if not size.isdecimal() or not 0<=int(size)<=_MAX_BLOCK:raise RuntimeError('DISTRIBUTED_CONTROL_BLOCK_LENGTH')
      payload=response.read(_MAX_BLOCK+1)
      if len(payload)==int(size):return payload
      time.sleep(.05)
   except urllib.error.HTTPError as error:
    if error.code==424:self.check_error()
    raise RuntimeError('DISTRIBUTED_CONTROL_REJECTED_'+str(error.code)) from error
   except (OSError,IncompleteRead):
    time.sleep(.05)
   if time.monotonic()>end:raise RuntimeError('DISTRIBUTED_CONTROL_TIMEOUT_NO_RESTART '+key)
 def put_block(self,key,index,data,*,deadline=None):
  if not isinstance(data,bytes) or not 0<len(data)<=_MAX_BLOCK:raise ValueError('DISTRIBUTED_CONTROL_BLOCK_SIZE')
  self._block('/put-block',key,index,data,deadline=deadline)
 def get_block(self,key,index,*,deadline=None):
  return self._block('/get-block',key,index,deadline=deadline)
 def acknowledge_block(self,key,index,*,command_key,deadline=None):
  if self.rank!=0:raise RuntimeError('CONTROL_RELEASE_REQUIRES_RANK_ZERO')
  self._block('/ack-block',key,index,command_key=command_key,deadline=deadline)
 def put(self,key,value,*,deadline=None):
  end=time.monotonic()+self.timeout
  while True:
   request_timeout=self._request_timeout(deadline)
   try:return self._request('/put',key,value,timeout=request_timeout)
   except urllib.error.HTTPError as error:raise RuntimeError('DISTRIBUTED_CONTROL_REJECTED_'+str(error.code)) from error
   except (OSError,urllib.error.URLError):
    if time.monotonic()>end:raise RuntimeError('DISTRIBUTED_CONTROL_UNAVAILABLE_NO_RESTART')
    _pause_retry(deadline)
 def check_error(self,*,deadline=None):
  wait={} if deadline is None else {'timeout':self._request_timeout(deadline)}
  try:v=self._request('/get','error',**wait)
  except (OSError,urllib.error.URLError):
   if deadline is not None:_deadline_remaining(deadline)
   return
  if v['present']:raise RuntimeError('DISTRIBUTED_PEER_ERROR: '+str(v['value']))
  if deadline is not None:_deadline_remaining(deadline)
 def try_get(self,key,*,ranks=None,deadline=None):
  path='/get' if ranks is None else '/collect-ranks'
  wait={} if deadline is None else {'timeout':self._request_timeout(deadline)}
  try:v=self._request(path,key,ranks,**wait)
  except urllib.error.HTTPError as error:raise RuntimeError('DISTRIBUTED_CONTROL_REJECTED_'+str(error.code)) from error
  except (OSError,urllib.error.URLError):
   if deadline is not None:_deadline_remaining(deadline)
   return None
  if v['error'] is not None:raise RuntimeError('DISTRIBUTED_PEER_ERROR: '+str(v['error']))
  if deadline is not None:_deadline_remaining(deadline)
  return v['value'] if v['present'] else None
 @staticmethod
 def _request_timeout(deadline):
  if deadline is None:return 3
  return min(3,_deadline_remaining(deadline))
 def get(self,key,timeout=None,*,deadline=None):
  end=time.monotonic()+(self.timeout if timeout is None else timeout)
  while True:
   request_timeout=self._request_timeout(deadline)
   try:
    v=self._request('/get',key,timeout=request_timeout)
    if v['error'] is not None and key!='error' and not key.startswith('error-ack/'):raise RuntimeError('DISTRIBUTED_PEER_ERROR: '+str(v['error']))
    if v['present']:return v['value']
   except urllib.error.HTTPError as error:raise RuntimeError('DISTRIBUTED_CONTROL_REJECTED_'+str(error.code)) from error
   except (OSError,urllib.error.URLError):pass
   if time.monotonic()>end:raise RuntimeError('DISTRIBUTED_CONTROL_TIMEOUT_NO_RESTART '+key)
   _pause_retry(deadline)
 def await_value(self,key,*,stop_key,deadline=None):
  # Quiet peers stay available as long as the coordinator is responsive.
  end=time.monotonic()+self.timeout
  while True:
   wait={} if deadline is None else {'timeout':self._request_timeout(deadline)}
   try:
    value=self._request('/await',key,stop_key,**wait)
   except urllib.error.HTTPError as error:raise RuntimeError('DISTRIBUTED_CONTROL_REJECTED_'+str(error.code)) from error
   except OSError:
    if time.monotonic()>end:raise RuntimeError('DISTRIBUTED_CONTROL_TIMEOUT_NO_RESTART '+key)
    _pause_retry(deadline)
    continue
   if value['error'] is not None:raise RuntimeError('DISTRIBUTED_PEER_ERROR: '+str(value['error']))
   if deadline is not None:_deadline_remaining(deadline)
   if value['present']:return value['value']
   end=time.monotonic()+self.timeout
 def report_error(self,error,world):
  """Preserve the first failure and give surviving ranks a shared drain deadline."""
  deadline=time.perf_counter()+5
  self.put('error',str(error),deadline=deadline)
  if self.rank==0:
   for rank in range(1,world):
    self.get(f'error-ack/{rank}',deadline=deadline)
  else:
   self.put(f'error-ack/{self.rank}',True,deadline=deadline)
 def release_prefix(self,prefix):
  if self.rank!=0:raise RuntimeError('CONTROL_RELEASE_REQUIRES_RANK_ZERO')
  return self._request('/delete-prefix',prefix)
 def close(self):
  if self.server:self.server.shutdown();self.server.server_close();self.thread.join()
