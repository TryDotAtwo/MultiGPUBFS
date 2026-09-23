"""Read-only two-T4 transport capability probe; no BFS performance claim."""
import ctypes
from pathlib import Path
import json
import os
import subprocess
import sys
import urllib.request

SOURCE = "ae22e29388f08ec426f0b70b867c34567a17ff7f"


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
#include <cstdlib>

static void nc(ncclResult_t r, const char* where) {
  if (r != ncclSuccess) throw std::runtime_error(std::string(where)+": "+ncclGetErrorString(r));
}
static void cu(cudaError_t r, const char* where) {
  if (r != cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(r));
}
constexpr unsigned capacity = 32, recv_offset = 64, send_offset = 128;
__global__ void exchange(ncclDevComm dev, ncclWindow_t win) {
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), dev, ncclTeamTagLsa(), 0};
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);
  auto local = static_cast<unsigned*>(ncclGetLsaPointer(win, 0, dev.lsaRank));
  auto peer = static_cast<unsigned*>(ncclGetLsaPointer(win, 0, 1-dev.lsaRank));
  if (threadIdx.x == 0) {
    unsigned n = local[2];
    if (n > capacity) local[1] = 1;
    peer[0] = n > capacity ? 0 : n;
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_acq_rel);
  bool bad = local[1] || peer[1];
  if (threadIdx.x == 0) local[3] = unsigned(bad);
  if (!bad)
    for (unsigned i = threadIdx.x; i < local[2]; i += blockDim.x)
      peer[recv_offset+i] = local[send_offset+i];
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}
static void worker(int rank, ncclUniqueId id, unsigned counts[2], std::string& error) {
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
    unsigned initial[256];
    for (auto& x : initial) x = 0xdeadbeefu;
    initial[0] = initial[1] = initial[3] = 0;
    initial[2] = counts[rank];
    for (unsigned i=0; i<=capacity; ++i)
      initial[send_offset+i] = 1000u*rank+i;
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
    unsigned result[256]{};
    cu(cudaMemcpy(result, buffer, sizeof(result), cudaMemcpyDeviceToHost), "result_copy");
    unsigned peer_count = counts[1-rank];
    bool failed = counts[0] > capacity || counts[1] > capacity;
    unsigned expected_count = peer_count > capacity ? 0 : peer_count;
    if (result[0] != expected_count || result[3] != unsigned(failed))
      throw std::runtime_error("LSA_CONTROL_MISMATCH");
    for (unsigned i=0; i<capacity; ++i) {
      unsigned expected = !failed && i<peer_count ? 1000u*(1-rank)+i : 0xdeadbeefu;
      if (result[recv_offset+i] != expected)
        throw std::runtime_error("LSA_PAYLOAD_MISMATCH");
    }
    std::printf("RANK_%d_LSA_COUNTED_PASS_%u_%u\n", rank, counts[rank], peer_count);
    std::fflush(stdout);
    nc(ncclDevCommDestroy(comm, &dev), "device_comm_destroy");
    nc(ncclCommWindowDeregister(comm, win), "window_deregister");
    nc(ncclMemFree(buffer), "mem_free");
    nc(ncclCommDestroy(comm), "comm_destroy");
  } catch (const std::exception& ex) { error = ex.what(); }
}
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  unsigned counts[2] = {unsigned(std::strtoul(argv[1], nullptr, 10)),
                        unsigned(std::strtoul(argv[2], nullptr, 10))};
  ncclUniqueId id{};
  nc(ncclGetUniqueId(&id), "unique_id");
  std::string errors[2];
  std::thread a(worker, 0, id, counts, std::ref(errors[0]));
  std::thread b(worker, 1, id, counts, std::ref(errors[1]));
  a.join(); b.join();
  for (int rank=0; rank<2; ++rank)
    if (!errors[rank].empty()) std::fprintf(stderr, "RANK_%d_ERROR_%s\n", rank,
                                            errors[rank].c_str());
  return errors[0].empty() && errors[1].empty() ? 0 : 1;
}
'''

PRODUCTION_EXCHANGE = r'''#include "mgbfs_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
static void cu(cudaError_t x,const char* where){
  if(x!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(x));
}
static void ok(int x,const char* where){
  if(x)throw std::runtime_error(std::string(where)+" rc="+std::to_string(x));
}
static void worker(unsigned rank,const unsigned char* id,unsigned counts[2],std::string& error){
 try {
  cu(cudaSetDevice(rank),"device");
  char why[256]{};void* comm{};
  ok(mgbfs_nccl_create(rank,2,rank,id,&comm,why,sizeof(why)),"create");
  int init=mgbfs_nccl_lsa_init(comm,32,16,why,sizeof(why));
  if(init)throw std::runtime_error("init rc="+std::to_string(init)+" "+why);
  uint32_t hc[2]={counts[0],counts[1]};
  uint32_t* dc{};uint4 *dh{},*ds{};
  cu(cudaMalloc(&dc,sizeof(hc)),"counts_alloc");
  cu(cudaMalloc(&dh,64*sizeof(uint4)),"hash_alloc");
  cu(cudaMalloc(&ds,64*sizeof(uint4)),"state_alloc");
  cu(cudaMemcpy(dc,hc,sizeof(hc),cudaMemcpyHostToDevice),"counts_copy");
  uint4 hashes[64]{},states[64]{};
  for(unsigned i=0;i<64;i++){
   hashes[i]={rank*1000+i,rank*1000+i+1,rank*1000+i+2,rank*1000+i+3};
   states[i]={rank*2000+i,rank*2000+i+1,rank*2000+i+2,rank*2000+i+3};
  }
  cu(cudaMemcpy(dh,hashes,sizeof(hashes),cudaMemcpyHostToDevice),"hash_copy");
  cu(cudaMemcpy(ds,states,sizeof(states),cudaMemcpyHostToDevice),"state_copy");
  const uint32_t *rc{},*fatal{};const void *rh{},*rs{};
  ok(mgbfs_nccl_lsa_view(comm,&rc,&fatal,&rh,&rs),"view");
  cu(cudaMemset(const_cast<void*>(rh),0xab,32*sizeof(uint4)),"sentinel_hash");
  cu(cudaMemset(const_cast<void*>(rs),0xab,32*sizeof(uint4)),"sentinel_state");
  ok(mgbfs_nccl_lsa_exchange(comm,dh,ds,dc,1-rank,1-rank,nullptr),"exchange");
  cu(cudaDeviceSynchronize(),"sync");
  uint32_t got_count{},got_fatal{};uint4 got_h[32]{},got_s[32]{};
  cu(cudaMemcpy(&got_count,rc,4,cudaMemcpyDeviceToHost),"read_count");
  cu(cudaMemcpy(&got_fatal,fatal,4,cudaMemcpyDeviceToHost),"read_fatal");
  cu(cudaMemcpy(got_h,rh,sizeof(got_h),cudaMemcpyDeviceToHost),"read_hash");
  cu(cudaMemcpy(got_s,rs,sizeof(got_s),cudaMemcpyDeviceToHost),"read_state");
  bool bad=counts[0]+counts[1]>32;
  unsigned incoming=counts[rank];
  if(got_fatal!=unsigned(bad)||got_count!=(bad?0:incoming))
   throw std::runtime_error("control mismatch");
  unsigned peer=1-rank,begin=rank==0?0:counts[0];
  for(unsigned i=0;i<32;i++){
   uint4 expected_h=bad||i>=incoming ? uint4{0xabababab,0xabababab,0xabababab,0xabababab}
     : uint4{peer*1000+begin+i,peer*1000+begin+i+1,peer*1000+begin+i+2,peer*1000+begin+i+3};
   uint4 expected_s=bad||i>=incoming ? uint4{0xabababab,0xabababab,0xabababab,0xabababab}
     : uint4{peer*2000+begin+i,peer*2000+begin+i+1,peer*2000+begin+i+2,peer*2000+begin+i+3};
   if(std::memcmp(&got_h[i],&expected_h,16)||std::memcmp(&got_s[i],&expected_s,16))
    throw std::runtime_error("payload mismatch row="+std::to_string(i));
  }
  std::printf("PRODUCTION_RANK_%u_PASS_%u_%u\n",rank,counts[rank],counts[1-rank]);
  cudaFree(ds);cudaFree(dh);cudaFree(dc);mgbfs_nccl_destroy(comm);
 }catch(const std::exception& ex){error=ex.what();}
}
int main(int argc,char** argv){
 if(argc!=3)return 2;
 unsigned counts[2]={unsigned(std::strtoul(argv[1],nullptr,10)),
                     unsigned(std::strtoul(argv[2],nullptr,10))};
 unsigned char id[128]{};if(mgbfs_nccl_unique_id(id))return 3;
 std::string errors[2];
 std::thread a(worker,0,id,counts,std::ref(errors[0]));
 std::thread b(worker,1,id,counts,std::ref(errors[1]));
 a.join();b.join();
 for(unsigned i=0;i<2;i++)if(!errors[i].empty())
  std::fprintf(stderr,"PRODUCTION_RANK_%u_ERROR_%s\n",i,errors[i].c_str());
 return errors[0].empty()&&errors[1].empty()?0:1;
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
            source_dir = Path("/tmp/mgbfs-production-transport")
            source_dir.mkdir(exist_ok=True)
            for name in ("mgbfs_cuda.h", "regenerate.h", "nccl_transport.cpp"):
                urllib.request.urlretrieve(
                    f"https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/{SOURCE}/cuda/{name}",
                    source_dir / name)
            report["production_transport_compile"] = command(
                "nvcc", "-std=c++17", "-arch=sm_75", "-DMGBFS_NCCL_LSA=1",
                "-I", str(includes[0].parent), "-x", "cu", "-c",
                str(source_dir / "nccl_transport.cpp"),
                "-o", "/tmp/mgbfs-production-transport.o", timeout=300)
            if report["production_transport_compile"]["code"] == 0 and libs:
                harness = source_dir / "production_exchange.cu"
                harness.write_text(PRODUCTION_EXCHANGE, encoding="utf-8")
                report["production_transport_link"] = command(
                    "nvcc", "-std=c++17", "-arch=sm_75", "-I", str(source_dir),
                    "-I", str(includes[0].parent), "-L", str(libs[0].parent),
                    "-Xlinker", "-rpath=" + str(libs[0].parent),
                    str(harness), "/tmp/mgbfs-production-transport.o",
                    "-l:libnccl.so.2", "-o", "/tmp/mgbfs-production-exchange",
                    timeout=300)
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
                if report.get("production_transport_link", {}).get("code") == 0:
                    report["production_exchange_cases"] = {}
                    for counts in ((3, 1), (0, 5), (20, 20), (33, 1)):
                        label = f"{counts[0]}_{counts[1]}"
                        report["production_exchange_cases"][label] = command(
                            "/tmp/mgbfs-production-exchange", str(counts[0]),
                            str(counts[1]), env=environment, timeout=120)
                    if all(case["code"] == 0 for case in
                           report["production_exchange_cases"].values()):
                        report["production_sanitizers_filtered"] = {}
                        for sanitizer in ("memcheck", "initcheck"):
                            report["production_sanitizers_filtered"][sanitizer] = command(
                                "compute-sanitizer", "--tool", sanitizer,
                                "--kernel-name", "kns=lsa_publish_count",
                                "--kernel-name", "kns=lsa_copy_exact",
                                "--report-api-errors", "no",
                                "--error-exitcode", "86",
                                "/tmp/mgbfs-production-exchange", "3", "1",
                                env=environment, timeout=600)
                report["lsa_counted_cases"] = {}
                for counts in ((3, 1), (0, 5), (33, 1)):
                    label = f"{counts[0]}_{counts[1]}"
                    report["lsa_counted_cases"][label] = command(
                        "/tmp/mgbfs-lsa-roundtrip", str(counts[0]), str(counts[1]),
                        env=environment, timeout=120)
    output = Path("/kaggle/working/transport-probe.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    if report.get("cuda_device_count") != 2:
        raise RuntimeError("EXPECTED_TWO_PHYSICAL_GPUS")


if __name__ == "__main__":
    main()
