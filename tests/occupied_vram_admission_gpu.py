import ctypes,json,subprocess,sysconfig,os
from pathlib import Path
from multigpubfs.native_distribution import native_runtime
r=Path('/root/universal/near-vram-gate');n,e=native_runtime();runtime=Path(sysconfig.get_paths()['purelib'])/'nvidia/cu13/lib/libcudart.so.13';lib=ctypes.CDLL(str(runtime));lib.cudaMalloc.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t];lib.cudaFree.argtypes=[ctypes.c_void_p];lib.cudaSetDevice.argtypes=[ctypes.c_int]
def admit(mode,cap=None):
 env=dict(e,MGBFS_GENERIC_HISTORY=mode,MGBFS_GENERIC_OWNER_LANES='4');args=[n,'graph-info',str(r/'definition.json'),'0,1','4'];args+=[] if cap is None else [str(cap)];return subprocess.run(args,env=env,capture_output=True,text=True,timeout=30)
baseline={}
for mode in ('hash','sorted'):
 x=admit(mode);assert x.returncode==0,x.stderr;baseline[mode]=json.loads(x.stdout)
p=ctypes.c_void_p();assert lib.cudaSetDevice(0)==0;assert lib.cudaMalloc(ctypes.byref(p),2<<30)==0
checks=[]
try:
 for mode in ('hash','sorted'):
  x=admit(mode);assert x.returncode==0,x.stderr;y=json.loads(x.stdout);assert y['rank_plans'][0]['capacity']<baseline[mode]['rank_plans'][0]['capacity'],(mode,y,baseline[mode])
  reject=admit(mode,baseline[mode]['rank_plans'][0]['capacity']);assert reject.returncode!=0 and ('REQUESTED_CAPACITY_EXCEEDS_ADMISSION' if mode=='hash' else 'REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION') in reject.stderr,reject.stderr
  checks.append({'mode':mode,'occupied_device':0,'occupied_bytes':2<<30,'baseline':baseline[mode],'replanned':y,'rejected_status':reject.returncode,'rejected_stderr':reject.stderr})
finally:assert lib.cudaFree(p)==0
receipt={'status':'VERIFIED_ACTUAL_OCCUPIED_VRAM_REPLANNING','checks':checks,'scope':'Actual 2GiB competing CUDA allocation, reduced capacity and oversized request rejection on two RTX3060; no claim of hotpath throughput'};(r/'occupied-vram-verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'status':receipt['status'],'capacities':[(c['mode'],c['baseline']['rank_plans'][0]['capacity'],c['replanned']['rank_plans'][0]['capacity']) for c in checks]}))
