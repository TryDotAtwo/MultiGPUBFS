import json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path

DRIVER=r"""
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
int cuDriverGetVersion(int*p){*p=11080;return 0;}
int cuInit(unsigned x){if(getenv("FAKE_HANG"))sleep(30);char*p=getenv("FAKE_INIT");return p?atoi(p):0;}
int cuDeviceGetCount(int*p){char*v=getenv("FAKE_COUNT");*p=v?atoi(v):2;return 0;}
int cuDeviceGetName(char*p,int n,int d){strncpy(p,"Fixture GPU",n);return 0;}
int cuDeviceGetAttribute(int*p,int a,int d){*p=a==75?7:5;return 0;}
int cuGetErrorName(int c,const char**p){*p=c==34?"CUDA_ERROR_STUB_LIBRARY":c==803?"CUDA_ERROR_SYSTEM_DRIVER_MISMATCH":"CUDA_SUCCESS";return 0;}
int cuGetErrorString(int c,const char**p){*p="controlled fixture response";return 0;}
"""
RUNTIME=r"""
#include <stdlib.h>
#include <unistd.h>
int cudaRuntimeGetVersion(int*p){*p=12090;return 0;}
int cudaDriverGetVersion(int*p){*p=11080;return 0;}
int cudaGetDeviceCount(int*p){*p=2;return 35;}
const char* cudaGetErrorString(int c){return c?"insufficient driver fixture":"success";}
"""

@unittest.skipUnless(os.name=='posix' and shutil.which('cc'),'Linux shared-library loader fixture needs C compiler')
class ActualDriverLoaderTests(unittest.TestCase):
 def test_real_ctypes_loader_and_failure_evidence(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for name,code in (('libcuda.so.1',DRIVER),('libcudart.so.12',RUNTIME)):
    source=root/(name+'.c');source.write_text(code);subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(root/name)],check=True)
   for index,(extra,expected) in enumerate((({},'DRIVER_MAJOR_TOO_OLD'),({'FAKE_INIT':'34'},'DRIVER_STUB_LOADED'),({'FAKE_INIT':'803'},'DRIVER_KERNEL_LIBRARY_MISMATCH'),({'FAKE_COUNT':'0'},'NO_VISIBLE_GPU'))):
    output=root/(str(index)+'.json');env=dict(os.environ,LD_LIBRARY_PATH=str(root));env.pop('LD_PRELOAD',None);env.update(extra)
    run=subprocess.run([sys.executable,'-m','multigpubfs.driver_diagnostics','--output',str(output)],env=env,capture_output=True,text=True,timeout=10)
    self.assertEqual(run.returncode,0,run.stderr);data=json.loads(output.read_text());self.assertEqual(data['classification'],expected)
    self.assertEqual(data['driver']['resolved_library'],str(root/'libcuda.so.1'));self.assertEqual(data['status'],'FAIL')
    self.assertEqual(data['runtime']['cudaGetDeviceCount']['code'],35)

   output=root/'partial.json';env=dict(os.environ,LD_LIBRARY_PATH=str(root),FAKE_HANG='1');env.pop('LD_PRELOAD',None)
   with self.assertRaises(subprocess.TimeoutExpired):subprocess.run([sys.executable,'-m','multigpubfs.driver_diagnostics','--output',str(output)],env=env,capture_output=True,text=True,timeout=2)
   partial=json.loads(output.read_text());self.assertEqual(partial['status'],'RUNNING');self.assertEqual(partial['driver']['api_version'],11080);self.assertEqual(partial['driver']['resolved_library'],str(root/'libcuda.so.1'))
