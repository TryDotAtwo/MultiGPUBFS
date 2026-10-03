"""Startup-only selection from matched, completed archived calibration runs.

The caller owns GPU execution and full-state comparison. This selector rejects
incomparable or incomplete measurements instead of extrapolating a fast profile.
"""
import copy
import math
import statistics


def select_graph_profile(samples):
    """Require three matched pairs, full-state parity and actual full windows.

    Each record carries the same configuration identity (device UUIDs, source,
    binary, transport, allocation geometry and archive policy), a pair number,
    mode 0/32, completed search seconds, parity and observed full-window count.
    """
    if len(samples) != 6:
        raise ValueError('six calibration measurements required')
    identity = samples[0]['configuration_identity']
    pairs = {}
    for sample in samples:
        if sample['configuration_identity'] != identity:
            raise ValueError('calibration configurations differ')
        seconds = sample['search_seconds']
        if (type(seconds) not in (int, float) or not math.isfinite(seconds)
                or seconds <= 0 or sample['status'] not in ('COMPLETE','PREFIX_COMPLETE')
                or sample['full_state_parity'] is not True):
            raise ValueError('unverified calibration measurement')
        mode = sample['graph_batches']
        if type(mode) is not int or mode not in (0, 32):
            raise ValueError('invalid graph mode')
        if mode == 32 and sample['full_windows_per_rank']:
            windows = sample['full_windows_per_rank']
            if any(type(x) is not int or x <= 0 for x in windows):
                raise ValueError('full graph windows not observed on every rank')
        elif mode == 32:
            raise ValueError('full graph windows not observed')
        pair = pairs.setdefault(sample['pair'], {})
        if mode in pair:
            raise ValueError('duplicate calibration pair mode')
        pair[mode] = seconds
    if len(pairs) != 3 or any(set(x) != {0, 32} for x in pairs.values()):
        raise ValueError('three matched calibration pairs required')
    direct = statistics.median(x[0] for x in pairs.values())
    graph = statistics.median(x[32] for x in pairs.values())
    wins = all(x[32] < x[0] for x in pairs.values())
    selected = 32 if wins and graph <= direct * .98 else 0
    return dict(graph_batches=selected, policy='matched-complete-archive-v1',
                configuration_identity=identity, direct_median_seconds=direct,
                graph_median_seconds=graph, graph_over_direct=graph/direct,
                all_pairs_improved=wins, required_median_gain=.02,
                scope=('matched completed-layer prefix only' if any(
                    x['status']=='PREFIX_COMPLETE' for x in samples) else 'measured configuration only'),
                samples=copy.deepcopy(samples))
