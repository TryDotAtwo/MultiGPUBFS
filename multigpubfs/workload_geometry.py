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
