"""Compile the pinned allocator entry point against controlled CUDA boundaries.

This verifies handle selection/API errors, not GPU mapping or NCCL windows.
The allocation function itself comes from the actual vendor source, not a
reimplementation. Real hardware sanitizer gates remain mandatory.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(os.environ.get('MGBFS_NCCL_POLICY_SOURCE',
    str(Path(__file__).resolve().parents[1] / 'build/nccl-source-2.29.7')))

HARNESS = r'''
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define CUDART_VERSION 12090
#define NCCL_API(...)
#define NCCL_NVTX3_FUNC_RANGE
using ncclResult_t=int;
constexpr int ncclSuccess=0;
using CUdevice=int;
using CUdeviceptr=uintptr_t;
using CUmemGenericAllocationHandle=uint64_t;
using CUmemAllocationHandleType=int;
using CUresult=int;
constexpr int CUDA_SUCCESS=0,CUDA_ERROR_NOT_PERMITTED=800,CUDA_ERROR_NOT_SUPPORTED=801;
constexpr int CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR=1,CU_MEM_HANDLE_TYPE_FABRIC=8;
constexpr int CU_MEM_ALLOCATION_TYPE_PINNED=1,CU_MEM_LOCATION_TYPE_DEVICE=1;
constexpr int CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED=128;
constexpr int CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED=129;
constexpr int CU_MEM_ALLOC_GRANULARITY_RECOMMENDED=1,CU_MEM_ACCESS_FLAGS_PROT_READWRITE=3;
struct CUmemAllocationProp {
 int type{},requestedHandleTypes{};
 struct {int type{},id{};} location;
 struct {int gpuDirectRDMACapable{};} allocFlags;
};
struct CUmemAccessDesc {struct {int type{},id{};} location;int flags{};};
int mnnvl=0,fabric_supported=1,create_error=0,create_calls=0,last_types=0;
int64_t ncclParamMNNVLEnable(){return mnnvl;}
int ncclCudaLibraryInit(){return 0;}
int ncclCuMemEnable(){return 1;}
int cudaGetDevice(int* d){*d=0;return 0;}
int cuDeviceGet(int* d,int value){*d=value;return 0;}
int cuDeviceGetAttribute(int* v,int attr,int){*v=attr==128?fabric_supported:0;return 0;}
int cuMemGetAllocationGranularity(size_t* v,const CUmemAllocationProp*,int){*v=65536;return 0;}
int cudaGetDeviceCount(int* v){*v=1;return 0;}
int cuMemCreate(uint64_t* h,size_t,const CUmemAllocationProp* p,int){
 ++create_calls;last_types=p->requestedHandleTypes;
 if(last_types&8)return 800;
 if(create_error)return create_error;
 *h=42;return 0;
}
int cuMemAddressReserve(uintptr_t* p,size_t,size_t,int,int){*p=65536;return 0;}
int cuMemMap(uintptr_t,size_t,int,uint64_t,int){return 0;}
int cudaDeviceCanAccessPeer(int* v,int,int){*v=0;return 0;}
int cuMemSetAccess(uintptr_t,size_t,const CUmemAccessDesc*,int){return 0;}
int cudaMalloc(void**,size_t){return 99;}
#define CUPFN(x) x
#define CUCHECK(x) do {if((x)!=0)return 2;} while(0)
#define CUDACHECK(x) CUCHECK(x)
#define CUDASUCCESS(x) ((x)==0)
#define CUDACHECKGOTO(x,ret,label) do {if((x)!=0){ret=2;goto label;}} while(0)
#define ALIGN_SIZE(n,g) n=((n)+(g)-1)/(g)*(g)
#define INFO(...)
// ACTUAL_VENDOR_FUNCTION
int main(int argc,char** argv){
 if(argc!=4)return 1;
 mnnvl=std::atoi(argv[1]);fabric_supported=std::atoi(argv[2]);create_error=std::atoi(argv[3]);
 void* p=nullptr;int result=ncclMemAlloc(&p,4096);
 std::printf("%d %d %d\n",result,create_calls,last_types);
}
'''


class AllocatorPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('g++')
        if not compiler or not (SOURCE / 'src/allocator.cc').exists():
            if 'MGBFS_NCCL_POLICY_SOURCE' in os.environ:
                raise RuntimeError('NCCL_POLICY_GATE_DEPENDENCY_MISSING')
            raise unittest.SkipTest('requires g++ and pinned NCCL source')
        text = (SOURCE / 'src/allocator.cc').read_text()
        begin = text.index('ncclResult_t  ncclMemAlloc(')
        end = text.index('NCCL_API(ncclResult_t, ncclMemFree', begin)
        cls.temp = tempfile.TemporaryDirectory(prefix='mgbfs-nccl-policy-')
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        # Generated compile artifact: the implementation is read from upstream.
        (root / 'probe.cpp').write_text(HARNESS.replace('// ACTUAL_VENDOR_FUNCTION',text[begin:end]))
        cls.binary = root / 'probe'
        subprocess.run([compiler,'-std=c++17','-Wall','-Werror',str(root/'probe.cpp'),
                        '-o',str(cls.binary)],check=True,capture_output=True,text=True)

    def run_policy(self,mnnvl,supported,error=0):
        return tuple(map(int,subprocess.check_output(
            [str(self.binary),str(mnnvl),str(supported),str(error)],text=True).split()))

    def test_disabled_mnnvl_never_attempts_forbidden_fabric_create(self):
        self.assertEqual(self.run_policy(0,1),(0,1,1))

    def test_enabled_mnnvl_retains_vendor_fabric_attempt(self):
        self.assertEqual(self.run_policy(1,1),(0,2,1))

    def test_auto_mnnvl_retains_vendor_fabric_attempt(self):
        self.assertEqual(self.run_policy(2,1),(0,2,1))

    def test_absent_fabric_capability_uses_posix(self):
        self.assertEqual(self.run_policy(2,0),(0,1,1))

    def test_posix_allocation_error_is_fatal_without_fallback(self):
        self.assertEqual(self.run_policy(0,1,2),(2,1,1))


class GateAdmission(unittest.TestCase):
    def test_explicit_gate_rejects_missing_vendor_source(self):
        with tempfile.TemporaryDirectory() as source:
            env = dict(os.environ, MGBFS_NCCL_POLICY_SOURCE=source)
            result = subprocess.run([sys.executable,__file__,'AllocatorPolicy'],
                                    env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)

    def test_explicit_gate_rejects_missing_compiler(self):
        env = dict(os.environ, MGBFS_NCCL_POLICY_SOURCE=str(SOURCE), PATH='')
        result = subprocess.run([sys.executable,__file__,'AllocatorPolicy'],
                                env=env,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)


if __name__=='__main__':
    unittest.main()
