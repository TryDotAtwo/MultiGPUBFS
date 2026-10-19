"""Run the real transport with fake NCCL calls; no GPU is required."""
import os, pathlib, subprocess, tempfile
root=pathlib.Path(__file__).resolve().parents[1]
cuda=pathlib.Path(os.environ.get("CUDA_HOME", "/tmp/mgbfs-portable/cuda"))
nccl=pathlib.Path(os.environ.get("NCCL_HOME", "/tmp/mgbfs-portable/deps/nvidia/nccl"))
with tempfile.TemporaryDirectory(prefix="mgbfs-nccl-contract-") as tmp:
 binary=pathlib.Path(tmp)/"contract"
 subprocess.run(["g++","-O2","-std=c++17","-I"+str(root/"cuda"),"-I"+str(cuda/"include"),"-I"+str(nccl/"include"),str(root/"tests/nccl_bounded_groups.cpp"),"-L"+str(cuda/"lib"),"-L"+str(nccl/"lib"),"-Wl,-rpath,"+str(cuda/"lib")+":"+str(nccl/"lib"),"-lcudart","-l:libnccl.so.2","-o",str(binary)],check=True,timeout=60)
 subprocess.run([str(binary)],check=True,timeout=90)
