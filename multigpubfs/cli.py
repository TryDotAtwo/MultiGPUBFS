"""Launch a validated JSON Cayley action using the installed native runtime."""
import argparse,json,sys
from pathlib import Path
from .graph_definition import _normalize_graph
from .launch import run_graph

def _session_main(argv):
 from .api import open_session

 parser=argparse.ArgumentParser(prog='multigpubfs run',description='Run an exact traversal or find paths on the selected CUDA GPUs.')
 parser.add_argument('definition',help='Graph JSON file or Family.constructor with --catalog')
 parser.add_argument('output',type=Path)
 parser.add_argument('--catalog',action='store_true')
 parser.add_argument('--catalog-args',type=json.loads)
 parser.add_argument('--catalog-kwargs',type=json.loads)
 parser.add_argument('--gpus',type=int)
 parser.add_argument('--devices',type=lambda s:tuple(map(int,s.split(','))))
 parser.add_argument('--start',type=json.loads)
 parser.add_argument('--targets',type=json.loads,help='JSON list of target states; omitted performs a full traversal')
 parser.add_argument('--direction',choices=('forward','bidirectional'))
 parser.add_argument('--goal-mode',choices=('nearest','window'))
 parser.add_argument('--extra-steps',type=int)
 parser.add_argument('--preset',choices=('balanced','maxmemory'),default='balanced')
 parser.add_argument('--tune',choices=('auto','off','full'),default='auto')
 parser.add_argument('--profile-registry',help='Local catalog or pinned GitHub URL')
 parser.add_argument('--profile-registry-sha256',help='Required SHA-256 for a remote catalog')
 parser.add_argument('--native-options',type=json.loads,help='JSON graph traversal settings for reproducible runs')
 parser.add_argument('--seconds',type=int)
 parser.add_argument('--max-paths',type=int)
 parser.add_argument('--max-output-bytes',type=int)
 parser.add_argument('--lineage-dir',type=Path,help='Scratch directory on the coordinator for window queries')
 parser.add_argument('--lineage-cache-bytes',type=int)
 parser.add_argument('--max-lineage-bytes',type=int)
 args=parser.parse_args(argv)
 if args.catalog:
  graph={'catalog':args.definition,'args':args.catalog_args,'kwargs':args.catalog_kwargs}
 else:
  if args.catalog_args is not None or args.catalog_kwargs is not None:parser.error('catalog arguments require --catalog')
  graph=args.definition
 path_options={'direction':args.direction,'goal_mode':args.goal_mode,'extra_steps':args.extra_steps,
               'max_paths':args.max_paths,'max_output_bytes':args.max_output_bytes,
               'lineage_dir':args.lineage_dir,'lineage_cache_bytes':args.lineage_cache_bytes,
               'max_lineage_bytes':args.max_lineage_bytes}
 path_options={name:value for name,value in path_options.items() if value is not None}
 if args.targets is None and path_options:parser.error('path options require --targets')
 if args.targets is not None and args.native_options is not None:parser.error('--native-options requires a traversal')
 options={'start':args.start,'preset':args.preset,'output':args.output,'max_seconds':args.seconds}
 if args.native_options is not None:options['native_options']=args.native_options
 try:
  profiles={}
  if args.profile_registry is not None:profiles['profile_registry']=args.profile_registry
  if args.profile_registry_sha256 is not None:profiles['profile_registry_sha256']=args.profile_registry_sha256
  with open_session(gpus=args.gpus,devices=args.devices,tune=args.tune,**profiles) as session:
   result=session.traverse(graph,**options) if args.targets is None else session.solve(graph,targets=args.targets,**options,**path_options)
 except (OSError,ValueError,RuntimeError,TypeError,NotImplementedError) as error:
  print(str(error),file=sys.stderr);return 1
 print(json.dumps(result.report,sort_keys=True));return 0

def main(argv=None):
 argv=list(sys.argv[1:] if argv is None else argv)
 if argv and argv[0]=='run':
  return _session_main(argv[1:])
 if argv and argv[0]=='cluster':
  from .cluster import main as cluster_main
  return cluster_main(argv[1:])
 if argv and argv[0]=='setup':
  from .cluster_setup import main as setup_main
  return setup_main(argv[1:])
 if argv and argv[0]=='prefetch':
  from .prefetch import main as prefetch_main
  return prefetch_main(argv[1:])
 if argv and argv[0]=='doctor':
  from .diagnostics import main as doctor_main
  return doctor_main(argv[1:])
 if argv and argv[0]=='profiles':
  from .profile_registry import main as profiles_main
  return profiles_main(argv[1:])
 parser=argparse.ArgumentParser(description='Exact GPU BFS for CayleyPy permutation and matrix actions. Defaults: all visible GPUs, measured startup tuning and compact terminal retention.')
 parser.epilog='See multigpubfs run --help, multigpubfs cluster --help, multigpubfs profiles --help and multigpubfs doctor --help.'
 parser.add_argument('definition',help='Schema2 JSON file, or Family.constructor with --catalog')
 parser.add_argument('--catalog',action='store_true',help='Resolve an exact public CayleyPy constructor name without evaluated source')
 parser.add_argument('--catalog-args',help='JSON positional argument list for the constructor')
 parser.add_argument('--catalog-kwargs',help='JSON keyword argument object for the constructor')
 parser.add_argument('output',type=Path,help='New result directory')
 parser.add_argument('--devices',help='Comma-separated local GPU IDs; omitted uses all visible GPUs')
 parser.add_argument('--seconds',type=int,default=None,help='Optional work budget in seconds; omitted or 0 runs until complete or resource stop')
 parser.add_argument('--capacity',type=int,help='Optional per-rank capacity, checked against actual free VRAM')
 parser.add_argument('--backend',choices=['auto','generic','shard_ab_hash','shard_ab_sort_merge'],default='auto')
 parser.add_argument('--memory-backend',choices=['legacy','segmented','segmented32','segmented24','permutation_bitmap'],help='Select BFS state storage. Large explicit capacities use segmented storage by default.')
 parser.add_argument('--bitmap-pack4',action='store_true',default=None,help='Use the four-bit resident frontier with the atomic permutation bitmap backend')
 parser.add_argument('--segment-rows',type=int,default=1<<32,help='Maximum states per segment bank (1..4294967296; default 4294967296)')
 parser.add_argument('--memory-preset',choices=['balanced','max_capacity','maxmemory'],default='balanced',help='Choose measured speed or a memory-capacity-oriented segmented/bitmap profile')
 parser.add_argument('--peer-transport',choices=['auto','host','lsa'],default='auto')
 parser.add_argument('--shards',type=int,help='Manual general-backend shard count; disables profile selection')
 parser.add_argument('--transport',choices=['auto','full','parent'],default='auto')
 parser.add_argument('--candidate-order',choices=['auto','none','radix'],default='auto')
 parser.add_argument('--no-autotune',action='store_true')
 args=parser.parse_args(argv)
 try:
  if args.catalog:
   graph=_normalize_graph({'catalog':args.definition,
                           'args':json.loads(args.catalog_args) if args.catalog_args is not None else None,
                           'kwargs':json.loads(args.catalog_kwargs) if args.catalog_kwargs is not None else None})
  else:
   if args.catalog_args is not None or args.catalog_kwargs is not None:raise ValueError('CATALOG_ARGUMENTS_REQUIRE_CATALOG')
   graph=_normalize_graph(Path(args.definition))
  devices=None if args.devices is None else [int(v) for v in args.devices.split(',')]
  report=run_graph(graph,args.output,devices=devices,max_seconds=args.seconds,capacity=args.capacity,backend=args.backend,memory_backend=args.memory_backend,bitmap_pack4=args.bitmap_pack4,segment_rows=args.segment_rows,memory_preset=args.memory_preset,peer_transport=args.peer_transport,shards=args.shards,transport=args.transport,candidate_order=args.candidate_order,autotune=not args.no_autotune)
 except (OSError,ValueError,RuntimeError,TypeError) as error:
  print(str(error),file=sys.stderr);return 1
 print(json.dumps(report,sort_keys=True));return 0
