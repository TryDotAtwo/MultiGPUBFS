"""Small launch/receipt rendezvous; GPU states never enter this control hot path."""
import json,threading,time,urllib.request,urllib.error,hmac
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
class ControlStore:
 def __init__(self,host,port,rank,session,timeout,token=''):
  self.url=f'http://{host}:{port}';self.rank=rank;self.session=session;self.timeout=timeout;self.token=token;self.server=None
  if rank==0:
   values={};lock=threading.Lock();expected=session;secret=token
   class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
     if not hmac.compare_digest(self.headers.get('X-MGBFS-Token',''),secret):self.send_error(403);return
     size=int(self.headers.get('Content-Length','0'))
     if not 0<size<=64<<20:self.send_error(413);return
     try:
      body=json.loads(self.rfile.read(size))
      if body.get('session')!=expected:self.send_error(409);return
      key=body['key']
      with lock:
       if self.path=='/put':
        if key in values and values[key]!=body['value']:self.send_error(409);return
        values[key]=body['value'];result={'present':True}
       elif self.path=='/delete-prefix':
        for old in list(values):
         if old.startswith(key):del values[old]
        result={'present':True}
       else:result={'present':key in values,'value':values.get(key),'error':values.get('error')}
      raw=json.dumps(result).encode();self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
     except (ValueError,KeyError,TypeError):self.send_error(400)
   self.server=ThreadingHTTPServer(('0.0.0.0',port),Handler);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
 def _request(self,path,key,value=None):
  raw=json.dumps({'session':self.session,'key':key,'value':value}).encode();request=urllib.request.Request(self.url+path,data=raw,headers={'Content-Type':'application/json','X-MGBFS-Token':self.token})
  with urllib.request.urlopen(request,timeout=3) as response:return json.load(response)
 def put(self,key,value):
  end=time.monotonic()+self.timeout
  while True:
   try:return self._request('/put',key,value)
   except urllib.error.HTTPError as error:raise RuntimeError('DISTRIBUTED_CONTROL_REJECTED_'+str(error.code)) from error
   except (OSError,urllib.error.URLError):
    if time.monotonic()>end:raise RuntimeError('DISTRIBUTED_CONTROL_UNAVAILABLE_NO_RESTART')
    time.sleep(.05)
 def check_error(self):
  try:v=self._request('/get','error')
  except (OSError,urllib.error.URLError):return
  if v['present']:raise RuntimeError('DISTRIBUTED_PEER_ERROR: '+str(v['value']))
 def get(self,key,timeout=None):
  end=time.monotonic()+(self.timeout if timeout is None else timeout)
  while True:
   try:
    v=self._request('/get',key)
    if v['error'] is not None and key!='error' and not key.startswith('error-ack/'):raise RuntimeError('DISTRIBUTED_PEER_ERROR: '+str(v['error']))
    if v['present']:return v['value']
   except urllib.error.HTTPError as error:raise RuntimeError('DISTRIBUTED_CONTROL_REJECTED_'+str(error.code)) from error
   except (OSError,urllib.error.URLError):pass
   if time.monotonic()>end:raise RuntimeError('DISTRIBUTED_CONTROL_TIMEOUT_NO_RESTART '+key)
   time.sleep(.05)
 def release_prefix(self,prefix):
  if self.rank!=0:raise RuntimeError('CONTROL_RELEASE_REQUIRES_RANK_ZERO')
  return self._request('/delete-prefix',prefix)
 def close(self):
  if self.server:self.server.shutdown();self.server.server_close();self.thread.join()
