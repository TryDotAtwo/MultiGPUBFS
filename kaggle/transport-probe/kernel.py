"""Read-only two-T4 transport capability probe; no BFS performance claim."""
import ctypes
from pathlib import Path
import json
import subprocess


def command(*args):
    try:
        result = subprocess.run(args, text=True, capture_output=True, check=False)
    except FileNotFoundError as error:
        return {"code": None, "stdout": "", "stderr": str(error)}
    return {"code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def library(name):
    try:
        return ctypes.CDLL(name), None
    except OSError as error:
        return None, str(error)


def main():
    report = {
        "scope": "NCCL runtime/header and CUDA P2P preflight; not a device-API transfer test",
        "inventory": command("nvidia-smi", "--query-gpu=index,name,uuid,memory.total", "--format=csv,noheader,nounits"),
        "topology": command("nvidia-smi", "topo", "-m"),
        "topology_p2p": command("nvidia-smi", "topo", "-p2p", "p"),
        "nvcc": command("nvcc", "--version"),
    }
    nccl, report["nccl_load_error"] = library("libnccl.so.2")
    if nccl is not None:
        version = ctypes.c_int()
        rc = nccl.ncclGetVersion(ctypes.byref(version))
        report["nccl_version"] = version.value if rc == 0 else None
        report["nccl_version_status"] = rc
    cudart, report["cudart_load_error"] = library("libcudart.so.12")
    if cudart is not None:
        count = ctypes.c_int()
        rc = cudart.cudaGetDeviceCount(ctypes.byref(count))
        report["cuda_device_count_status"] = rc
        report["cuda_device_count"] = count.value if rc == 0 else None
        if rc == 0:
            report["can_access_peer"] = []
            for source in range(count.value):
                row = []
                for target in range(count.value):
                    flag = ctypes.c_int()
                    status = cudart.cudaDeviceCanAccessPeer(
                        ctypes.byref(flag), source, target)
                    row.append({"status": status, "value": flag.value if status == 0 else None})
                report["can_access_peer"].append(row)
    report["nccl_device_headers"] = [
        str(path) for root in (Path("/usr/include"), Path("/usr/local/cuda/include"))
        for path in root.glob("nccl*device*.h")
    ]
    output = Path("/kaggle/working/transport-probe.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    if report.get("cuda_device_count") != 2:
        raise RuntimeError("EXPECTED_TWO_PHYSICAL_GPUS")


if __name__ == "__main__":
    main()
