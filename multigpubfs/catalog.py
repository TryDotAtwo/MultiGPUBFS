"""Optional named CayleyPy constructor adapter; imports only during startup."""
from .graph_definition import from_cayleypy
def from_catalog(name,*,args=None,kwargs=None):
 args=[] if args is None else args;kwargs={} if kwargs is None else kwargs
 if not isinstance(name,str) or name.count('.')!=1:raise ValueError('INVALID_CATALOG_CONSTRUCTOR')
 family,method=name.split('.')
 if family not in ('PermutationGroups','MatrixGroups','Puzzles') or not method.isidentifier() or method.startswith('_'):raise ValueError('INVALID_CATALOG_CONSTRUCTOR')
 if not isinstance(args,(list,tuple)) or not isinstance(kwargs,dict) or any(not isinstance(k,str) or not k.isidentifier() or k.startswith('_') for k in kwargs):raise ValueError('INVALID_CATALOG_ARGUMENTS')
 try:import cayleypy
 except ImportError as error:raise RuntimeError('CAYLEYPY_OPTIONAL_DEPENDENCY_REQUIRED: install multigpubfs[cayleypy]') from error
 constructor=getattr(getattr(cayleypy,family),method,None)
 if not callable(constructor):raise ValueError('UNKNOWN_CATALOG_CONSTRUCTOR')
 return from_cayleypy(constructor(*args,**kwargs))
