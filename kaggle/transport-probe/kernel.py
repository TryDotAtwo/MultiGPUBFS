"""Read-only two-T4 transport capability probe; no BFS performance claim."""
import ctypes
from pathlib import Path
import json
import os
import subprocess
import sys


def command(*args, env=None, timeout=180):
    try:
        result = subprocess.run(args, text=True, capture_output=True, check=False,
                                env=env, timeout=timeout)
    except FileNotFoundError as error:
        return {"code": None, "stdout": "", "stderr": str(error)}
    except subprocess.TimeoutExpired as error:
        return {"code": None, "stdout": str(error.stdout),
                "stderr": "TIMEOUT: " + str(error)}
    return {"code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def library(name):
    try:
        return ctypes.CDLL(name), None
    except OSError as error:
        return None, str(error)


LSA_ROUNDTRIP = r'''#include <nccl.h>
#include <nccl_device.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>

static void nc(ncclResult_t r, const char* where) {
  if (r != ncclSuccess) throw std::runtime_error(std::string(where)+": "+ncclGetErrorString(r));
}
static void cu(cudaError_t r, const char* where) {
  if (r != cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(r));
}
__global__ void exchange(ncclDevComm dev, ncclWindow_t win) {
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), dev, ncclTeamTagLsa(), 0};
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);
  if (threadIdx.x == 0) {
    auto peer = static_cast<unsigned*>(ncclGetLsaPointer(win, 0, 1-dev.lsaRank));
    peer[dev.lsaRank] = 100u + dev.lsaRank;
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}
static void worker(int rank, ncclUniqueId id, std::string& error) {
  try {
    cu(cudaSetDevice(rank), "set_device");
    ncclComm_t comm{};
    nc(ncclCommInitRank(&comm, 2, id, rank), "init_rank");
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    nc(ncclCommQueryProperties(comm, &props), "query_properties");
    std::printf("RANK_%d_DEVICE_API_%d_LSA_TEAMS_%d\n", rank,
                int(props.deviceApiSupport), props.nLsaTeams);
    std::fflush(stdout);
    if (!props.deviceApiSupport || props.nLsaTeams != 1)
      throw std::runtime_error("LSA_NOT_SUPPORTED");
    unsigned* buffer{};
    nc(ncclMemAlloc(reinterpret_cast<void**>(&buffer), 65536), "mem_alloc");
    unsigned initial[2] = {unsigned(rank+1), 0};
    cu(cudaMemcpy(buffer, initial, sizeof(initial), cudaMemcpyHostToDevice), "init_copy");
    ncclWindow_t win{};
    nc(ncclCommWindowRegister(comm, buffer, 65536, &win, NCCL_WIN_COLL_SYMMETRIC),
       "window_register");
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = 1;
    ncclDevComm dev{};
    nc(ncclDevCommCreate(comm, &reqs, &dev), "device_comm_create");
    exchange<<<1, 32>>>(dev, win);
    cu(cudaGetLastError(), "launch");
    cu(cudaDeviceSynchronize(), "sync");
    unsigned result[2]{};
    cu(cudaMemcpy(result, buffer, sizeof(result), cudaMemcpyDeviceToHost), "result_copy");
    unsigned expected[2] = {unsigned(rank == 0 ? 1 : 100),
                            unsigned(rank == 0 ? 101 : 0)};
    if (result[0] != expected[0] || result[1] != expected[1])
      throw std::runtime_error("LSA_DATA_MISMATCH");
    std::printf("RANK_%d_LSA_ROUNDTRIP_PASS\n", rank);
    std::fflush(stdout);
    nc(ncclDevCommDestroy(comm, &dev), "device_comm_destroy");
    nc(ncclCommWindowDeregister(comm, win), "window_deregister");
    nc(ncclMemFree(buffer), "mem_free");
    nc(ncclCommDestroy(comm), "comm_destroy");
  } catch (const std::exception& ex) { error = ex.what(); }
}
int main() {
  ncclUniqueId id{};
  nc(ncclGetUniqueId(&id), "unique_id");
  std::string errors[2];
  std::thread a(worker, 0, id, std::ref(errors[0]));
  std::thread b(worker, 1, id, std::ref(errors[1]));
  a.join(); b.join();
  for (int rank=0; rank<2; ++rank)
    if (!errors[rank].empty()) std::fprintf(stderr, "RANK_%d_ERROR_%s\n", rank,
                                            errors[rank].c_str());
  return errors[0].empty() && errors[1].empty() ? 0 : 1;
}
'''


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
    target = Path("/tmp/mgbfs-nccl-2.29.7")
    install = command(sys.executable, "-m", "pip", "install", "--no-deps",
                      "--target", str(target), "nvidia-nccl-cu12==2.29.7")
    report["nccl_229_install"] = install
    if install["code"] == 0:
        includes = list(target.glob("nvidia/nccl/include/nccl.h"))
        libs = list(target.glob("nvidia/nccl/lib/libnccl.so.2"))
        report["nccl_229_header"] = [str(path) for path in includes]
        report["nccl_229_device_headers"] = [str(path) for path in target.rglob("nccl*device*.h")]
        report["nccl_229_libraries"] = [str(path) for path in libs]
        if libs:
            newer = ctypes.CDLL(str(libs[0]))
            newer_version = ctypes.c_int()
            report["nccl_229_version_status"] = newer.ncclGetVersion(ctypes.byref(newer_version))
            report["nccl_229_version"] = newer_version.value
        if includes and report["nccl_229_device_headers"]:
            probe = Path("/tmp/mgbfs-nccl-device-compile.cu")
            probe.write_text("#include <nccl.h>\n#include <nccl_device.h>\n"
                             "__global__ void probe(ncclDevComm) {}\n", encoding="utf-8")
            report["nccl_229_compile"] = command(
                "nvcc", "-std=c++17", "-arch=sm_75", "-c", "-I", str(includes[0].parent),
                str(probe), "-o", "/tmp/mgbfs-nccl-device-compile.o")
            roundtrip_source = Path("/tmp/mgbfs-lsa-roundtrip.cu")
            roundtrip_source.write_text(LSA_ROUNDTRIP, encoding="utf-8")
            libdir = str(libs[0].parent) if libs else ""
            report["lsa_roundtrip_build"] = command(
                "nvcc", "-std=c++17", "-arch=sm_75", "-I", str(includes[0].parent),
                "-L", libdir, "-Xlinker", "-rpath=" + libdir,
                str(roundtrip_source), "-l:libnccl.so.2", "-o", "/tmp/mgbfs-lsa-roundtrip")
            if report["lsa_roundtrip_build"]["code"] == 0:
                environment = dict(os.environ)
                environment["LD_LIBRARY_PATH"] = libdir + ":" + environment.get("LD_LIBRARY_PATH", "")
                environment["NCCL_CUMEM_ENABLE"] = "1"
                report["lsa_roundtrip"] = command(
                    "/tmp/mgbfs-lsa-roundtrip", env=environment, timeout=120)
    output = Path("/kaggle/working/transport-probe.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    if report.get("cuda_device_count") != 2:
        raise RuntimeError("EXPECTED_TWO_PHYSICAL_GPUS")


if __name__ == "__main__":
    main()
