"""Freeze the selected owner pipeline before admission and transport probes."""
import copy

FLAGS = ('MGBFS_SHARD_AB_KEY_FIRST','MGBFS_COMPACT_DIRECT_HASH',
         'MGBFS_SHARD_AB_PEER_METADATA','MGBFS_SHARD_AB_ASYNC_MATERIALIZE',
         'MGBFS_SHARD_AB_DIRECT_INPUT','MGBFS_SORT_HISTORY_LOOKUP',
         'MGBFS_SHARD_AB_COMBINED_STATUS','MGBFS_SHARD_AB_FUSED_RESPONSE_META',
         'MGBFS_SHARD_AB_REUSE_HISTORY','MGBFS_SHARD_AB_REUSE_PREOWNER_STATUS')


def select_pipeline(base, runtime, requested=None, *, resume=False):
    result = copy.deepcopy(base)
    explicit = requested or runtime.get('MGBFS_PIPELINE_PROFILE')
    if explicit is not None and explicit not in ('baseline','key-first'):
        raise ValueError('unknown pipeline profile')
    saved = result.get('pipeline_profile')
    if resume:
        # Old ledgers retain their original contract; adding a profile is a
        # new-run operation, never an implicit rewrite of completed pairs.
        if saved is None:
            if runtime.get('MGBFS_SHARD_AB_REUSE_PREOWNER_STATUS') == '1':
                raise ValueError('status reuse cannot be added to a legacy resumed run')
            if explicit is not None:
                raise ValueError('pipeline profile cannot be added to a legacy resumed run')
            return result
        if explicit is not None and explicit != saved:
            raise ValueError('resume pipeline profile differs from saved configuration')
        # Older profiled ledgers predate this optimization and retain it off.
        result['env'].setdefault('MGBFS_SHARD_AB_REUSE_PREOWNER_STATUS','0')
        for key in FLAGS + ('MGBFS_SHARD_AB_DEDUP','MGBFS_PRE_DEDUP'):
            if key in runtime and key in result['env'] and runtime[key] != result['env'][key]:
                raise ValueError('resume pipeline setting differs: '+key)
        return result
    profile = explicit or ('key-first' if result['env'].get('MGBFS_OWNER_BACKEND') == 'SHARD_AB' and runtime.get('MGBFS_SHARD_AB_KEY_FIRST') != '0' else 'baseline')
    if result['env'].get('MGBFS_OWNER_BACKEND') != 'SHARD_AB':
        if profile != 'baseline':
            raise ValueError('key-first pipeline requires SHARD_AB')
        return result
    result['pipeline_profile'] = profile
    env = result['env']
    env.update({key:'0' for key in FLAGS})
    if profile == 'key-first':
        env.update({key:'1' for key in FLAGS[:6]})
        env['MGBFS_PRE_DEDUP'] = 'OFF'
    mode = runtime.get('MGBFS_SHARD_AB_DEDUP', env.get('MGBFS_SHARD_AB_DEDUP','HASH'))
    if mode not in ('HASH','SORT_MERGE'):
        raise ValueError('invalid shard dedup mode')
    env['MGBFS_SHARD_AB_DEDUP'] = mode
    if profile == 'key-first' and mode == 'HASH':
        env['MGBFS_SHARD_AB_REUSE_PREOWNER_STATUS'] = '1'
    if profile == 'baseline' and 'MGBFS_PRE_DEDUP' in runtime:
        env['MGBFS_PRE_DEDUP'] = runtime['MGBFS_PRE_DEDUP']
    # Experimental options that had no measured benefit stay off. Raw runtime
    # flags must agree with the frozen profile instead of changing admission.
    for key in FLAGS + ('MGBFS_PRE_DEDUP',):
        if key in runtime and runtime[key] != env[key]:
            raise ValueError('runtime pipeline flag conflicts with profile: '+key)
    return result
