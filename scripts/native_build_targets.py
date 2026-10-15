"""Explicit code targets shared by native build and CPU regression."""
from multigpubfs.architecture_support import NATIVE_ARCHITECTURES,PTX_ARCHITECTURES
def targets(variant=None,target=None):
 if target is not None:
  if variant!='shared-sm120' or target not in [str(x) for x in NATIVE_ARCHITECTURES]:raise ValueError('UNPUBLISHED_NATIVE_TARGET')
  return [int(target)],[],target+'-real'
 if variant=='shared-legacy':return [75,86],[], '75;86'
 arches=list(NATIVE_ARCHITECTURES);ptx=list(PTX_ARCHITECTURES)
 return arches,ptx,';'.join([str(x)+'-real' for x in arches]+[str(x)+'-virtual' for x in ptx])
