"""Lossless CayleyPy graph contract. CPU successors are verification oracles only."""
import copy,hashlib,json

I64_MIN=-(1<<63);I64_MAX=(1<<63)-1

def wrap_i64(x):
 return ((x+(1<<63))&((1<<64)-1))-(1<<63)

def _integer(x):
 if type(x) is not int or not I64_MIN<=x<=I64_MAX:raise ValueError('GRAPH_INT64_RANGE')
 return x

def _vector(v):
 if not isinstance(v,list):raise ValueError('GRAPH_VECTOR')
 return [_integer(x) for x in v]

class GraphDefinition:
 def __init__(self,value):
  self._data=copy.deepcopy(value);self.validate()
 @classmethod
 def permutation(cls,generators,start,*,name='',generator_names=None,expected_max_unique_states=None):
  return cls({'schema':2,'name':name,'generator_names':generator_names or [str(i) for i in range(len(generators))],
   'action':{'kind':'permutation','degree':len(start),'generators':generators},'start':start,'expected_max_unique_states':expected_max_unique_states})
 @classmethod
 def matrix(cls,rows,cols,generators,start,*,name='',generator_names=None,expected_max_unique_states=None):
  return cls({'schema':2,'name':name,'generator_names':generator_names or [str(i) for i in range(len(generators))],
   'action':{'kind':'matrix','rows':rows,'cols':cols,'generators':[{'matrix':list(mx),'modulo':m} for mx,m in generators]},
   'start':start,'expected_max_unique_states':expected_max_unique_states})
 @classmethod
 def from_dict(cls,value):return cls(value)
 @property
 def start(self):return copy.deepcopy(self._data['start'])
 @property
 def action(self):return copy.deepcopy(self._data['action'])
 @property
 def generator_names(self):return list(self._data['generator_names'])
 @property
 def state_elements(self):return len(self._data['start'])
 @property
 def generator_count(self):return len(self._data['action']['generators'])
 def to_dict(self):return copy.deepcopy(self._data)
 def to_json(self):return json.dumps(self._data,sort_keys=True,separators=(',',':'),ensure_ascii=True)
 def digest(self):
  semantic={k:self._data[k] for k in ('schema','action','start')}
  return hashlib.sha256(json.dumps(semantic,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 def validate(self):
  d=self._data
  if not isinstance(d,dict) or set(d)!={'schema','name','generator_names','action','start','expected_max_unique_states'} or d['schema']!=2:raise ValueError('GRAPH_SCHEMA')
  if not isinstance(d['name'],str):raise ValueError('GRAPH_NAME')
  start=_vector(d['start']);a=d['action']
  if not isinstance(a,dict) or not isinstance(a.get('generators'),list) or not a['generators']:raise ValueError('GRAPH_GENERATORS')
  if not isinstance(d['generator_names'],list) or len(d['generator_names'])!=len(a['generators']) or not all(isinstance(x,str) for x in d['generator_names']):raise ValueError('GRAPH_GENERATOR_NAMES')
  bound=d['expected_max_unique_states']
  if bound is not None and (type(bound) is not int or not 1<=bound<1<<64):raise ValueError('GRAPH_ORBIT_BOUND')
  if a.get('kind')=='permutation':
   n=a.get('degree')
   if set(a)!={'kind','degree','generators'} or type(n) is not int or not 1<=n<=0xffffffff or len(start)!=n:raise ValueError('GRAPH_PERMUTATION_SHAPE')
   expected=list(range(n))
   for g in a['generators']:
    if _vector(g)!=g or sorted(g)!=expected:raise ValueError('GRAPH_PERMUTATION_GENERATOR')
  elif a.get('kind')=='matrix':
   n,m=a.get('rows'),a.get('cols')
   if set(a)!={'kind','rows','cols','generators'} or type(n) is not int or type(m) is not int or min(n,m)<1 or n*m>0xffffffff or len(start)!=n*m:raise ValueError('GRAPH_MATRIX_SHAPE')
   for g in a['generators']:
    if not isinstance(g,dict) or set(g)!={'matrix','modulo'} or len(_vector(g['matrix']))!=n*n:raise ValueError('GRAPH_MATRIX_GENERATOR')
    mod=g['modulo']
    if type(mod) is not int or (mod!=0 and not 2<=mod<=1<<31):raise ValueError('GRAPH_MATRIX_MODULUS')
  else:raise ValueError('GRAPH_ACTION_KIND')
  return self
 def successor(self,state,generator):
  state=_vector(state)
  if len(state)!=self.state_elements or type(generator) is not int or not 0<=generator<self.generator_count:raise ValueError('GRAPH_SUCCESSOR_ARGUMENT')
  a=self._data['action'];g=a['generators'][generator]
  if a['kind']=='permutation':return [state[i] for i in g]
  return self._multiply(g['matrix'],state,a['rows'],a['cols'],g['modulo'])
 @staticmethod
 def _multiply(mx,state,n,m,mod):
  result=[]
  for i in range(n):
   for j in range(m):
    total=wrap_i64(sum(mx[i*n+k]*state[k*m+j] for k in range(n)))
    result.append(total%mod if mod else total)
  return result
 def inverse_closed(self):
  a=self._data['action'];gs=a['generators']
  if a['kind']=='permutation':
   available={tuple(g) for g in gs}
   for g in gs:
    inv=[0]*len(g)
    for i,x in enumerate(g):inv[x]=i
    if tuple(inv) not in available:return False
   return True
  n=a['rows']
  if len({g['modulo'] for g in gs})!=1:return False
  mod=gs[0]['modulo']
  if mod and mod&(mod-1):
   if any(sum(abs(v)*(mod-1) for v in g['matrix'][i*n:(i+1)*n])>I64_MAX for g in gs for i in range(n)):return False
  eye=[int(i==j) for i in range(n) for j in range(n)]
  def compose(g,h):
   right=[v%mod for v in h['matrix']] if mod else h['matrix']
   return self._multiply(g['matrix'],right,n,n,mod)
  for g in gs:
   if not any(compose(g,h)==eye and compose(h,g)==eye for h in gs):return False
  # A modular inverse is an inverse only on canonical values. Noncanonical
  # starts may enter the canonical orbit irreversibly on the first edge.
  for g in gs:
   if g['modulo'] and any(not 0<=x<g['modulo'] for x in self._data['start']):return False
  return True
 def exact_layers(self,maximum_states):
  if type(maximum_states) is not int or maximum_states<1:raise ValueError('ORACLE_CAPACITY')
  seen={tuple(self.start)};frontier={tuple(self.start)};layers=[]
  while frontier:
   layers.append([list(x) for x in sorted(frontier)])
   future={tuple(self.successor(list(x),g)) for x in frontier for g in range(self.generator_count)}-seen
   if len(seen)+len(future)>maximum_states:raise ValueError('ORACLE_CAPACITY')
   seen.update(future);frontier=future
  return layers

def _flatten(value):
 if hasattr(value,'tolist'):value=value.tolist()
 if isinstance(value,(list,tuple)):
  return [x for item in value for x in _flatten(item)]
 # CayleyPy/NumPy scalar conversion must not silently change the graph.
 if isinstance(value,bool):raise ValueError('GRAPH_NONINTEGER_SCALAR')
 try:converted=int(value)
 except (TypeError,ValueError,OverflowError):raise ValueError('GRAPH_NONINTEGER_SCALAR') from None
 if converted!=value:raise ValueError('GRAPH_NONINTEGER_SCALAR')
 return [_integer(converted)]

def from_cayleypy(definition):
 """Preserve CayleyGraphDef generators and start; never substitute a family."""
 if not hasattr(definition,'generators_type'):
  if hasattr(definition,'definition'):definition=definition.definition
  elif hasattr(definition,'graph_def'):definition=definition.graph_def
 kind=definition.generators_type.name;start=_flatten(definition.central_state)
 options={'name':definition.name,'generator_names':list(definition.generator_names)}
 if kind=='PERMUTATION':return GraphDefinition.permutation([_flatten(g) for g in definition.generators_permutations],start,**options)
 if kind=='MATRIX':
  generators=definition.generators_matrices;first=generators[0].matrix
  n=int(first.shape[0]) if hasattr(first,'shape') else len(first)
  if len(start)%n:raise ValueError('GRAPH_MATRIX_SHAPE')
  return GraphDefinition.matrix(n,len(start)//n,[(_flatten(g.matrix),int(g.modulo)) for g in generators],start,**options)
 raise ValueError('GRAPH_ACTION_KIND')
