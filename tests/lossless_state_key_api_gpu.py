from pathlib import Path
import ctypes as C,os,json,traceback,sys
r=Path(sys.argv[2]);r.mkdir(parents=True,exist_ok=True)
try:
 u=C.c_uint32;p=C.c_void_p;size=C.c_size_t
 lib=C.CDLL(sys.argv[1]);cuda=C.CDLL('/usr/local/cuda/lib64/libcudart.so');cuda.cudaMalloc.argtypes=[C.POINTER(p),size];cuda.cudaMemcpy.argtypes=[p,p,size,C.c_int];cuda.cudaFree.argtypes=[p]
 lib.mgbfs_lossless_hash_create.argtypes=[u,u,u,C.POINTER(p),p,size];lib.mgbfs_lossless_compact_hash_create.argtypes=[u,u,u,p,u,u,C.POINTER(p),p,size];lib.mgbfs_hash_run.argtypes=[p,p,p,u,p];lib.mgbfs_compact_hash_run.argtypes=[p,p,p,u,p];lib.mgbfs_hash_destroy.argtypes=[p];lib.mgbfs_compact_hash_destroy.argtypes=[p]
 def check(x):
  if x:raise RuntimeError('CUDA_STATUS_'+str(x))
 def oracle(values,bits):
  mask=(1<<64)-1
  def f(x):
   x^=x>>30;x=x*0xbf58476d1ce4e5b9&mask;x^=x>>27;x=x*0x94d049bb133111eb&mask;return x^(x>>31)
  v=sum(x<<(j*bits) for j,x in enumerate(values));lo=v&mask;hi=v>>64;lo^=f(hi^0x9e3779b97f4a7c15);hi^=f(lo^0xd1b54a32d192ed03);return [lo&0xffffffff,lo>>32,hi&0xffffffff,hi>>32]
 results=[]
 for device in (0,1):
  check(cuda.cudaSetDevice(device))
  for n,bits in ((14,4),(15,4),(25,5),(16,8),(128,1)):
   count=17;stride=(n+15)&~15;values=[[(row*7+j*3)&((1<<bits)-1) for j in range(n)] for row in range(count)];data=[v for row in values for v in row+[0]*(stride-n)];moves=[list(range(1,n))+[0],[n-1]+list(range(n-1)),[1,0]+list(range(2,n))];table=[v for move in moves for v in move];a=(C.c_uint8*len(data))(*data);pm=(C.c_uint8*len(table))(*table);d=p();out=p();plan=p();error=C.create_string_buffer(512);check(cuda.cudaMalloc(C.byref(d),len(data)));check(cuda.cudaMalloc(C.byref(out),count*3*16));check(cuda.cudaMemcpy(d,a,len(data),1))
   try:
    check(lib.mgbfs_lossless_hash_create(n,count,bits,C.byref(plan),error,512));check(lib.mgbfs_hash_run(plan,d,out,count,None));words=(u*(count*4))();check(cuda.cudaMemcpy(words,out,C.sizeof(words),2));assert list(words)==[x for row in values for x in oracle(row,bits)];assert lib.mgbfs_hash_run(plan,d,out,count+1,None)!=0;assert lib.mgbfs_hash_run(plan,d,C.c_void_p(out.value+4),count,None)!=0;lib.mgbfs_hash_destroy(plan);plan=p()
    for move_major in (0,1):
     check(lib.mgbfs_lossless_compact_hash_create(n,3,count,pm,bits,move_major,C.byref(plan),error,512));check(lib.mgbfs_compact_hash_run(plan,d,out,count,None));words=(u*(count*3*4))();check(cuda.cudaMemcpy(words,out,C.sizeof(words),2));expected=[oracle([values[row][j] for j in moves[move]],bits) for move in range(3) for row in range(count)] if move_major else [oracle([values[row][j] for j in moves[move]],bits) for row in range(count) for move in range(3)];assert list(words)==[x for key in expected for x in key];lib.mgbfs_compact_hash_destroy(plan);plan=p()
    results.append({'device':device,'n':n,'bits':bits,'rows':count,'layouts':[0,1]})
   finally:
    check(cuda.cudaFree(d));check(cuda.cudaFree(out))
  plan=p();assert lib.mgbfs_lossless_hash_create(26,17,5,C.byref(plan),error,512)!=0 and plan.value is None
 (r/'key-api-gates.json').write_text(json.dumps({'status':'VERIFIED_CUDA_STATE_CHILD_KEYS_WIDE_BOUNDARY_LAYOUT_DOMAIN','results':results}))
except Exception:
 (r/'key-api-gates.json').write_text(json.dumps({'status':'ERROR','trace':traceback.format_exc()}));raise
