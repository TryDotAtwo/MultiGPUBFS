"""Bounded failed-job metadata capture. Never copy job env, GPU states or retry work."""
import hashlib,json,os,time
from pathlib import Path

def _copy(source,destination,limit,offset=0):
 size=source.stat().st_size;start=max(min(offset,size),size-limit)
 with source.open('rb') as stream:stream.seek(start);data=stream.read(limit)
 destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_suffix(destination.suffix+'.tmp');temporary.write_bytes(data);os.replace(temporary,destination)
 return {'bytes':len(data),'source_bytes':size,'offset':start,'truncated':start>offset,'sha256':hashlib.sha256(data).hexdigest()}

def preserve_native_failure(jobs,output,error,*,when,max_bytes=2*1024*1024):
 """Best effort: evidence I/O must never replace the original native failure."""
 report={'schema':1,'status':'CAPTURED','error':str(error),'when':when,'captured_unix':time.time(),'ranks':[],
  'scope':'Failed job phase/response and bounded current-job log only; no environment, states, restart or acceptance claim'}
 try:
  output=Path(output);root=output/'failure-evidence';root.mkdir(parents=True,exist_ok=True)
  for ordinal,job in enumerate(jobs):
   worker=getattr(job,'worker',None);rank=getattr(worker,'rank',ordinal);device=getattr(worker,'device',rank);sequence=getattr(job,'sequence',None)
   row={'rank':rank,'device':device,'pid':getattr(job,'pid',None),'sequence':sequence,'exit_code':getattr(job,'returncode',None),'files':{},'capture_errors':[]}
   destination=root/f'rank-{rank}';destination.mkdir(exist_ok=True)
   if worker is not None and sequence is not None:
    sources={'phase.json':worker.root/f'phase-{sequence:08}.json','response.json':worker.root/f'response-{sequence:08}.json','worker.log':worker.root/'worker.log'}
   else:sources={'phase.json':output/f'rank-{ordinal}.phase.json','worker.log':output/f'rank-{ordinal}.log'}
   for name,source in sources.items():
    try:
     if source.is_file():
      row['files'][name]=_copy(source,destination/name,max_bytes,getattr(job,'log_offset',0) if name=='worker.log' and worker is not None else 0)
    except Exception as failure:row['capture_errors'].append({'file':name,'error':str(failure)})
   report['ranks'].append(row)
  summary=root/'summary.json';old=json.loads(summary.read_text()) if summary.is_file() else {'schema':1,'captures':[]}
  old['captures'].append(report);old['status']=report['status'];old['error']=str(error)
  temporary=summary.with_suffix('.tmp');temporary.write_text(json.dumps(old,indent=2));os.replace(temporary,summary)
 except Exception as failure:report.update(status='CAPTURE_INCOMPLETE',capture_error=str(failure))
 return report
