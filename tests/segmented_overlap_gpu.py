"""Opt-in CUDA event intervals for segmented source/owner streams."""
import os,re,subprocess,sys
from pathlib import Path
env=dict(os.environ,SEG24_BENCH_TRACE="1",SEG24_BENCH_N="10",
         SEG24_BENCH_CAPACITY="2000000",MGBFS_SEGMENT_TEST_OVERLAP="1",
         MGBFS_SEGMENT_TEST_OVERLAP_MIN_DEPTH="14",MGBFS_SEGMENT_TEST_OVERLAP_MAX_DEPTH="16")
r=subprocess.run([sys.executable,"tests/segmented_speed_gpu.py"],env=env,
                 check=True,text=True,capture_output=True,timeout=240)
print(r.stdout,flush=True)
root=Path(re.findall(r"^EVIDENCE (.+)$",r.stdout,re.M)[-1])
rows=[]
for p in root.glob("case-*/rank-*.log"):
 for line in p.read_text().splitlines():
  if line.startswith("SEGMENT_OVERLAP "):
   vals=dict(x.split("=",1) for x in line.split()[1:])
   a,b,c,d,o=(float(vals[k]) for k in ("route_start_ms","route_end_ms","owner_start_ms","owner_end_ms","overlap_ms"))
   assert 0<=a<=b and 0<=c<=d and o>=0,vals
   assert 14<=int(vals["depth"])+1<=16,vals
   assert abs(o-max(0.,min(b,d)-max(a,c)))<0.001,vals
   rows.append(vals)
assert rows, "No timed pairs of CUDA stream intervals"
print("TIMED_PAIRS",len(rows),"WITH_INTERVAL_OVERLAP",sum(float(v["overlap_ms"])>0.01 for v in rows),flush=True)
