"""GPU-independent native cache preparation for login hosts and shared storage."""
import argparse,contextlib,hashlib,importlib.metadata,json,shlex,sys
from . import native_assets as assets

def prefetch_native(sm=None,*,all_architectures=False,cache=None,dry_run=False,timeout=600):
    """Cache selected SMs or all production builds; never query/access a GPU.

    sm accepts a number, comma-separated string or sequence. ``--all`` also
    includes universal code for mixed/future-compatible devices. Optional
    diagnostic controls are not part of this production prefetch.
    Set MGBFS_NATIVE_CACHE to the returned cache in every GPU job.
    """
    if not dry_run and (assets.platform.system()!='Linux' or assets.platform.machine() not in ('x86_64','AMD64')):
        raise assets.AssetError('NATIVE_PREFETCH_PLATFORM: use a Linux x86_64 cache host; no GPU is required')
    raw=assets.index_path().read_bytes()
    manager=assets.AssetManager(json.loads(raw),assets.cache_path() if cache is None else cache,timeout=timeout)
    result=manager.prefetch(sm,all_architectures=all_architectures,dry_run=dry_run)
    result['index_file_sha256']=hashlib.sha256(raw).hexdigest()
    try:result['package']=importlib.metadata.version('multigpubfs')
    except importlib.metadata.PackageNotFoundError:result['package']='source'
    info=assets.index_path().with_name('_package_info.json')
    if info.is_file():result['package_source_commit']=json.loads(info.read_text()).get('source_commit')
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description='Prefetch checked native builds into a shared cache without GPU access. No GPU discovery, CUDA loading or BFS is performed.')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sm',action='append',help='Published target(s), e.g. 100,120; repeatable. Use universal for mixed/future-compatible GPUs.')
    group.add_argument('--all',action='store_true',help='All production SM builds and universal; excludes optional diagnostic controls')
    parser.add_argument('--cache',help='Persistent shared cache directory; default follows MGBFS_NATIVE_CACHE/XDG_CACHE_HOME')
    parser.add_argument('--dry-run',action='store_true',help='Verify existing cache and report maximum remaining bytes; no downloads or writes')
    parser.add_argument('--timeout',type=float,default=600,help='Total preparation deadline in seconds (up to 1800)')
    parser.add_argument('--json',action='store_true',help='Write one JSON result to stdout; progress is written to stderr')
    args=parser.parse_args(argv)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result=prefetch_native(args.sm,all_architectures=args.all,cache=args.cache,dry_run=args.dry_run,timeout=args.timeout)
    except (OSError,ValueError,TypeError,RuntimeError) as error:
        if args.json:print(json.dumps({'schema':1,'status':'ERROR','error':str(error),'gpu_accessed':False},sort_keys=True))
        else:print('NATIVE PREFETCH ERROR: '+str(error),file=sys.stderr)
        return 1
    if args.json:print(json.dumps(result,sort_keys=True))
    else:
        print('NATIVE PREFETCH '+result['status'])
        print('Targets: '+','.join(result['targets']))
        print('Cache: '+result['cache'])
        print('Total component archives: '+str(result['total_component_bytes'])+' bytes')
        print('Maximum additional download: '+str(result['maximum_additional_download_bytes'])+' bytes')
        print('Downloaded in this call: '+str(result['downloaded_bytes'])+' bytes')
        print('GPU accessed: no; '+result['scope'])
        print('Use the same cache in every GPU job:')
        for key,value in result['environment'].items():print('export '+key+'='+shlex.quote(value))
    return 0

if __name__=='__main__':raise SystemExit(main())
