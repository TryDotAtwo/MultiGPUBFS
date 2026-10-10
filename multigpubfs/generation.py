"""Exact Tensor Core domain; backend is chosen by matched BFS prefix timing."""
def gemm_supported(graph):
 a=graph.action
 return a['kind']=='matrix' and 8<=a['rows']<=64 and a['rows']%8==0 and 1<=a['cols']<=64 and 1<=len(a['generators'])<=64 and all(0<=v<=255 for v in graph.start) and all(1<=g['modulo']<=256 and all(0<=v<=255 for v in g['matrix']) for g in a['generators'])
