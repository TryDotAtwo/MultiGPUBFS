"""Build-time code coverage and driver evidence; never execution acceptance."""
NATIVE_ARCHITECTURES=[75,80,86,89,90,100,103,120]
PTX_ARCHITECTURES=[80]

def coverage(manifest,devices):
 native=manifest.get('architectures',[]);ptx=manifest.get('ptx_architectures')
 native=[x for x in native if type(x) is int and x>0]
 ptx=None if ptx is None else [x for x in ptx if type(x) is int and x>0]
 rows=[]
 for device in devices:
  cc=device.get('compute_capability',[])
  valid=isinstance(cc,(list,tuple)) and len(cc)==2 and all(type(x) is int and x>=0 for x in cc)
  target=10*cc[0]+cc[1] if valid else None
  compatible=[] if target is None else [x for x in native if x//10==target//10 and x%10<=target%10]
  portable=[] if target is None or ptx is None else [x for x in ptx if x<=target]
  code=('UNKNOWN_DEVICE' if target is None else 'NATIVE' if compatible else 'PTX_JIT' if portable else 'UNKNOWN_PTX' if ptx is None else 'UNSUPPORTED')
  rows.append({'logical_id':device.get('logical_id'),'name':device.get('name'),'compute_capability':cc,'sm':target,'code':code,'compatible_cubins':compatible,'compatible_ptx':portable})
 return {'schema':1,'native_architectures':native,'ptx_architectures':ptx,'devices':rows,'scope':'Ordinary non-a/non-f cubin and PTX compatibility; declarations are compiled code inventory, not successful GPU execution. PTX JIT may add first-load time.'}
