"""Certified workload bounds; no CPU successor generation in the GPU path."""
def coxeter_frontier_bound(graph):
 a=graph.action;n=graph.state_elements
 if a['kind']!='permutation' or not 2<=n<=32 or len(set(graph.start))!=n:return None
 expected=set()
 for i in range(n-1):
  g=list(range(n));g[i],g[i+1]=g[i+1],g[i];expected.add(tuple(g))
 if {tuple(g) for g in a['generators']}!=expected:return None
 # Length in adjacent transpositions is the inversion number. Coefficients
 # count each exact layer; every rank's layer is bounded by its global size.
 sizes=[1]
 for i in range(1,n+1):
  out=[];window=0
  for j in range(len(sizes)+i-1):
   if j<len(sizes):window+=sizes[j]
   if j>=i:window-=sizes[j-i]
   out.append(window)
  sizes=out
 return {'kind':'CERTIFIED_COXETER_GLOBAL_FRONTIER','maximum_frontier':max(sizes),'order':sum(sizes),'depth':len(sizes)-1,'scope':'global peak is a safe per-rank upper bound, independent of hash balance'}

def certified_frontier_bound(graph):
 bound=coxeter_frontier_bound(graph)
 if bound is not None:return bound
 a=graph.action
 if a['kind']!='matrix':return None
 moduli={g['modulo'] for g in a['generators']}
 if len(moduli)!=1:return None
 modulo=next(iter(moduli))
 if modulo<=0 or any(not 0<=v<modulo for v in graph.start):return None
 # Every canonical entry remains in [0,m). This bounds the full orbit,
 # hence every layer, without identifying a particular matrix group.
 order=1
 for _ in graph.start:
  order*=modulo
  if order>0x7ffffffe:return None
 return {'kind':'CERTIFIED_MODULAR_MATRIX_ORBIT','maximum_frontier':order,'order_upper_bound':order,'scope':'canonical same-modulus entry-space bound; not exact group order'}
