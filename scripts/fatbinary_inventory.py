"""Require every CUDA module in every advertised native and PTX target."""
import re
def modules(text,suffix):
 result={}
 for module,architecture in re.findall(r'([^\s:]+)\.sm_(\d+)\.'+suffix+r'\b',text):
  module=re.sub(r'\.\d+$','',module)
  result.setdefault(int(architecture),set()).add(module)
 return result
def verify_inventory(sass,ptx,native_architectures,ptx_architectures):
 native=modules(sass,'cubin');portable=modules(ptx,'ptx')
 expected=set().union(*(native.get(x,set()) for x in native_architectures))
 if not expected:raise RuntimeError('FATBINARY_EMPTY_NATIVE_INVENTORY')
 for kind,records,architectures in [('CUBIN',native,native_architectures),('PTX',portable,ptx_architectures)]:
  for architecture in architectures:
   missing=expected-records.get(architecture,set())
   if missing:raise RuntimeError('FATBINARY_'+kind+'_MISSING_SM_'+str(architecture)+': '+','.join(sorted(missing)))
 return {'native_module_counts':{str(x):len(native[x]) for x in native_architectures},'ptx_module_counts':{str(x):len(portable[x]) for x in ptx_architectures},'modules':sorted(expected),'scope':'cuobjdump inventory; real GPU correctness requires execution'}
