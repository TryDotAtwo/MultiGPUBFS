"""Exact full-bank duplicate and genuine resource-stop regression on one GPU.

Set MGBFS_EXECUTABLE and a matching LD_LIBRARY_PATH. The small GPU oracles
are deliberately run with one two-row segment and a one-parent batch.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile

from multigpubfs import GraphDefinition

root = Path(tempfile.mkdtemp(prefix='mgbfs-full-bank-'))
native = os.environ['MGBFS_EXECUTABLE']
rotation = lambda k: [(i + k) % 6 for i in range(6)]
directed = GraphDefinition.permutation([rotation(1), rotation(4)], list(range(6)))
inverse_closed = GraphDefinition.permutation([
    [0, 1, 2, 4, 3], [1, 3, 4, 0, 2], [3, 0, 4, 1, 2],
], [0, 0, 0, 0, 2])
def swap(i):
    p = list(range(4)); p[i], p[i + 1] = p[i + 1], p[i]; return p
wide = GraphDefinition.permutation([swap(i) for i in range(3)], list(range(4)))

def run(name, graph, bitmap):
    case = root / name; case.mkdir()
    definition = case / 'definition.json'; definition.write_text(graph.to_json())
    expected = graph.exact_layers(1000)
    env = dict(os.environ, MGBFS_EXPERIMENTAL_SEGMENTED24='1',
        MGBFS_EXPERIMENTAL_BITMAP=str(int(bitmap)), MGBFS_SEGMENT_ROWS='2',
        MGBFS_SEGMENT_TEST_LIMIT='1', MGBFS_GENERIC_GENERATOR='cuda',
        MGBFS_GENERIC_HISTORY='hash')
    query = subprocess.run([native, 'graph-info', str(definition), '0', '1', '1024', '1'],
        env=env, text=True, capture_output=True, timeout=60, check=True)
    plan = json.loads(query.stdout); plan['owner_cuts'] = None
    config = case / 'launch.json'; config.write_text(json.dumps(plan))
    p = subprocess.run([native, 'graph-rank', str(definition), str(config), '0',
        str(case / 'id'), str(case / 'rank-0'), '0'], env=env, text=True,
        capture_output=True, timeout=60)
    logs = p.stdout + '\n' + p.stderr; (case / 'rank-0.log').write_text(logs)
    if p.returncode != 0: raise RuntimeError((name, p.returncode, logs[-4000:]))
    report = json.loads((case / 'rank-0' / 'report.json').read_text())
    snapshot = json.loads((case / 'rank-0' / 'states.json').read_text())
    depth = len(report['layer_sizes']) - 1
    assert report['layer_sizes'] == list(map(len, expected[:depth + 1])), (name, report, expected)
    assert sorted(snapshot['current']) == expected[depth], (name, snapshot, expected[depth])
    print(json.dumps({'case': name, 'status': report['status'], 'reason': report.get('reason'),
        'layers': report['layer_sizes'], 'expected': list(map(len, expected)), 'dir': str(case)}), flush=True)
    return report, logs

try:
    assert list(map(len, directed.exact_layers(1000))) == [1, 2, 2, 1]
    assert inverse_closed.inverse_closed()
    assert list(map(len, inverse_closed.exact_layers(1000))) == [1, 2, 2]
    for i in range(3):
        report, _ = run(f'bitmap-duplicate-{i}', directed, True)
        assert report['status'] == 'COMPLETE' and report['layer_sizes'] == [1, 2, 2, 1], report
        report, _ = run(f'segmented-duplicate-{i}', inverse_closed, False)
        assert report['status'] == 'COMPLETE' and report['layer_sizes'] == [1, 2, 2], report
    for bitmap in (True, False):
        report, logs = run('actual-resource-' + ('bitmap' if bitmap else 'segmented'), wide, bitmap)
        assert report['status'] == 'INCOMPLETE' and report['reason'] == 'RESOURCE_10' and report['layer_sizes'] == [1], report
        assert 'SEGMENTED_RESOURCE rank=0 phase=segment_allocation reason=segment_budget' in logs, logs
    print('ALL_REGRESSIONS_PASS', root, flush=True)
finally:
    print('EVIDENCE', root, flush=True)
