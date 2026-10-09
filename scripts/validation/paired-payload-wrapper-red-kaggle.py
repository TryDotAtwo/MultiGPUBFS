import pathlib,urllib.request,subprocess,json
SOURCE="08c6305751edc2dfdc194e4f9776dacf9936eb70"
root=pathlib.Path("/tmp/mgbfs-payload-red");root.mkdir()
out=pathlib.Path("/kaggle/working");(out/"started.json").write_text(json.dumps({"source":SOURCE}))
paths=["tests/nccl_transport_failure.cpp","tests/nccl_stubs/cuda_runtime.h","tests/nccl_stubs/nccl.h","cuda/nccl_transport.cpp","cuda/mgbfs_cuda.h","cuda/regenerate.h","cuda/state_commit.h","cuda/bounded_owner.h"]
import re,posixpath,hashlib
seen=set();manifest=[]
while paths:
 path=paths.pop(0)
 if path in seen:continue
 if path.startswith("../") or path.startswith("/"):raise RuntimeError("Invalid source path")
 seen.add(path);p=root/path;p.parent.mkdir(parents=True,exist_ok=True)
 content=urllib.request.urlopen("https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/"+SOURCE+"/"+path).read();p.write_bytes(content)
 manifest.append({"path":path,"sha256":hashlib.sha256(content).hexdigest()})
 for include in re.findall(r'^\s*#include\s+"([^"]+)"',content.decode(),re.M):
  paths.append(posixpath.normpath(posixpath.join(posixpath.dirname(path),include)))
(out/"source-manifest.json").write_text(json.dumps(manifest,indent=2))
cmd=["g++","-std=c++17","-pthread","-I"+str(root/"tests/nccl_stubs"),str(root/"tests/nccl_transport_failure.cpp"),"-o",str(root/"test")]
b=subprocess.run(cmd,capture_output=True,text=True,timeout=120);(out/"build.log").write_text(b.stdout+b.stderr)
if b.returncode:raise RuntimeError("RED must compile before execution")
r=subprocess.run([str(root/"test")],capture_output=True,text=True,timeout=30);(out/"test.log").write_text(r.stdout+r.stderr)
expected=r.returncode!=0 and "mgbfs_nccl_send_recv_pair != nullptr" in r.stderr
(out/"summary.json").write_text(json.dumps({"source":SOURCE,"status":"EXPECTED_RED" if expected else "UNEXPECTED_RESULT","build_returncode":b.returncode,"test_returncode":r.returncode,"expected_missing_pair_assertion":expected},indent=2))
if not expected:raise RuntimeError("Unexpected RED result")
