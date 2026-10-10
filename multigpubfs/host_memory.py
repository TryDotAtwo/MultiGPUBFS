"""Cold launch inventory; no host checks in the GPU iteration path."""
from pathlib import Path
import re,shutil,tempfile

def inventory(proc=Path('/proc')):
 limits=[];sources=[]
 try:
  for line in (proc/'meminfo').read_text().splitlines():
   if line.startswith('MemAvailable:'):limits.append(int(line.split()[1])*1024);sources.append('MemAvailable')
 except (OSError,ValueError):pass
 try:
  groups=[line.split(':',2) for line in (proc/'self/cgroup').read_text().splitlines()]
  for line in (proc/'self/mountinfo').read_text().splitlines():
   left,right=line.split(' - ',1);fields=left.split();kind=right.split()[0]
   if kind not in ('cgroup','cgroup2'):continue
   unescape=lambda text:re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),text)
   root=Path(unescape(fields[3]));mount=Path(unescape(fields[4]));v2=kind=='cgroup2'
   for _,controllers,location in groups:
    if (v2 and controllers) or (not v2 and 'memory' not in controllers.split(',')):continue
    try:relative=Path(location).relative_to(root)
    except ValueError:continue
    node=mount/relative
    while True:
     try:
      maximum=(node/('memory.max' if v2 else 'memory.limit_in_bytes')).read_text().strip()
      current=int((node/('memory.current' if v2 else 'memory.usage_in_bytes')).read_text())
      if maximum!='max' and int(maximum)<1<<60:limits.append(max(0,int(maximum)-current));sources.append(str(node))
     except (OSError,ValueError):pass
     if node==mount:break
     node=node.parent
 except (OSError,ValueError):pass
 return {'available_bytes':min(limits) if limits else None,'sources':sources}

def admit(graph,world=1,external=False,output=None):
 info=inventory()
 # Compact retention is globally <=1999 rows. Allow Python integers/lists,
 # simultaneous parsed receipts, serialization and graph definition copies.
 required=(64<<20)+2000*graph.state_elements*192+len(graph.to_json())*8+(world*(1<<20) if external else 0)
 payload=2000*graph.state_elements*22+(1<<20)
 if external and payload>64<<20:raise RuntimeError('HOST_CONTROL_PAYLOAD_EXCEEDS_LIMIT')
 available=info['available_bytes']
 if available is not None and required>available//2:raise RuntimeError('HOST_MEMORY_ADMISSION_FAILED: estimated cold control memory '+str(required)+'; available '+str(available))
 disk=[]
 if output is not None:
  # Native rank snapshots and final merged JSON coexist. Pilots live in temp.
  disk_required=(64<<20)+payload*4+len(graph.to_json())*4+world*(1<<20)
  for target in (Path(output),Path(tempfile.gettempdir())):
   while not target.exists():target=target.parent
   free=shutil.disk_usage(target).free
   if free<disk_required:raise RuntimeError('HOST_DISK_ADMISSION_FAILED: estimated compact output '+str(disk_required)+'; free '+str(free))
   disk.append({'free_bytes':free,'estimated_required_bytes':disk_required})
 return dict(info,estimated_control_bytes=required,disk=disk,scope='cold compact control/output estimate; excludes native pinned transfer allocations')
