"""Launch a validated JSON Cayley action using the installed native runtime."""
import argparse,json,sys
from pathlib import Path
from .graph_definition import GraphDefinition
from .launch import run_graph

def main(argv=None):
 parser=argparse.ArgumentParser(description='Exact GPU BFS for CayleyPy permutation and matrix actions. Defaults: all visible GPUs, measured startup tuning and compact terminal retention.')
 parser.add_argument('definition',type=Path,help='GraphDefinition schema2 JSON file')
 parser.add_argument('output',type=Path,help='New result directory')
 parser.add_argument('--devices',help='Comma-separated local GPU IDs; omitted uses all visible GPUs')
 parser.add_argument('--seconds',type=int,default=3600,help='Bounded launch/work budget including startup tuning')
 parser.add_argument('--capacity',type=int,help='Optional per-rank capacity, checked against actual free VRAM')
 parser.add_argument('--backend',choices=['auto','generic','shard_ab_hash','shard_ab_sort_merge'],default='auto')
 parser.add_argument('--peer-transport',choices=['auto','host','lsa'],default='auto')
 parser.add_argument('--shards',type=int,help='Manual general-backend shard count; disables profile selection')
 parser.add_argument('--transport',choices=['auto','full','parent'],default='auto')
 parser.add_argument('--candidate-order',choices=['auto','none','radix'],default='auto')
 parser.add_argument('--no-autotune',action='store_true')
 args=parser.parse_args(argv)
 try:
  graph=GraphDefinition.from_dict(json.loads(args.definition.read_text(encoding='utf-8')))
  devices=None if args.devices is None else [int(v) for v in args.devices.split(',')]
  report=run_graph(graph,args.output,devices=devices,max_seconds=args.seconds,capacity=args.capacity,backend=args.backend,peer_transport=args.peer_transport,shards=args.shards,transport=args.transport,candidate_order=args.candidate_order,autotune=not args.no_autotune)
 except (OSError,ValueError,RuntimeError) as error:
  print(str(error),file=sys.stderr);return 1
 print(json.dumps(report,sort_keys=True));return 0
