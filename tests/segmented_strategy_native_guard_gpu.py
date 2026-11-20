"""A direct native launch must reject unsupported strategy before a graph layer."""
import os, subprocess
from pathlib import Path
native=os.environ["MGBFS_EXECUTABLE"]
for setting,expected in (("invalid","SEGMENT_STRATEGY_INVALID"),("warp","SEGMENT_STRATEGY_REQUIRES_SEGMENTED_BACKEND"),("batch","SEGMENT_STRATEGY_REQUIRES_SEGMENTED_BACKEND")):
 env=dict(os.environ,MGBFS_SEGMENT_STRATEGY=setting)
 env.pop("MGBFS_EXPERIMENTAL_SEGMENTED24",None)
 result=subprocess.run([native,"graph-rank"],env=env,text=True,capture_output=True,timeout=30)
 assert result.returncode!=0 and expected in result.stderr,(setting,result.stdout,result.stderr)
 print("PASS direct native fail closed",setting,flush=True)
print("NATIVE_STRATEGY_GUARD_PASS")
