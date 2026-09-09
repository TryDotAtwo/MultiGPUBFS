"""Read-only reconciliation of the pinned September S11 benchmark panels."""
import json
import csv
import statistics
import sys
from pathlib import Path


def audit_raw(summary_directory, raw_directory):
    """Require every successful raw rank, memory sample and archive result."""
    summary = json.loads((Path(summary_directory) / 'summary.json').read_text())
    raw = Path(raw_directory)
    world = summary['world_size']
    for row in summary['rows']:
        label = row['label']
        log = (raw / (label + '.log')).read_text()
        saved = json.loads((raw / (label + '.json')).read_text())
        assert saved['status'] == row['status']
        if row['status'] != 'COMPLETE':
            assert row['phase'] == 'calibrate'
            print(label, 'OOM' if 'out of memory' in log.lower() else 'failure needs diagnosis')
            continue
        peaks = [None] * world
        for sample in csv.reader((raw / (label + '-smi.csv')).read_text().splitlines()):
            if len(sample) < 6:
                continue
            rank = int(sample[1].strip())
            if rank < world:
                value = float(sample[3].strip())
                peaks[rank] = value if peaks[rank] is None else max(peaks[rank], value)
        assert peaks == row['smi_peak_mib_per_rank'], (label, peaks)
        for rank, expected in enumerate(row['rank_results']):
            actual = json.loads((raw / (label + '-ranks') / f'rank-{rank}.json').read_text())
            assert actual == expected, (label, rank)
            if row['config_backend'] == 'native':
                verified = json.loads((raw / f'{label}-verify-{rank}.log').read_text())
                assert verified == row['archive_verification'][rank]
    print('Raw artifact reconciliation passed:', raw)


def audit(directory):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    environment = json.loads((directory / 'environment.json').read_text())
    assert environment['source'] == '66d82d03cb055daa08dae328978208efda7c8ede'
    assert environment['baseline'] == 'f0f2b8e5ee61173039ab9742f3a7756c9b6365e6'
    assert summary['status'] == 'COMPLETE'
    world = summary['world_size']
    assert world in (1, 2)
    expected = summary['comparisons'][0]['layer_sizes']
    assert len(expected) == 56 and sum(expected) == 39916800
    rows = summary['rows']
    assert len(rows) == 68 and len(summary['comparisons']) == 12
    groups = {}
    for row in rows:
        if row['phase'] == 'calibrate' and row['status'] != 'COMPLETE':
            assert row['config_backend'] == 'cayleypy'
            print('Calibration failure retained:', row['label'], row['status'], row['exit_code'])
            continue
        assert row['status'] == 'COMPLETE' and row['exit_code'] == 0
        assert row['world_size'] == world and row['layer_sizes'] == expected
        assert row['smi_memory_complete'] and len(row['rank_results']) == world
        if row['config_backend'] != 'native':
            continue
        assert len(row['archive_verification']) == world
        assert all(x['status'] == 'VERIFIED' for x in row['archive_verification'])
        ranks = row['rank_results']
        assert sorted(x['rank'] for x in ranks) == list(range(world))
        assert all(x['warmup_completed'] and x['archive_enabled'] for x in ranks)
        assert all(x['durable_run_commit_seconds'] >= x['search_complete_seconds'] for x in ranks)
        assert [sum(x['local_layer_sizes'][d] for x in ranks) for d in range(56)] == expected
        r = ranks[0]
        key = (r['frontier_profile'], r['hash_first_generation'], r['owner_backend'], r['pre_dedup'])
        groups.setdefault(key, []).append(row)
    assert len(groups) == 12
    print(f'\n{world} T4 | profile | generation | owner | prededup | search median | MAD | durable median | peak total MiB')
    for key, samples in groups.items():
        assert sorted(r['repetition'] for r in samples) == list(range(5))
        values = [r['search_complete_seconds'] for r in samples]
        median = statistics.median(values)
        mad = statistics.median(abs(x - median) for x in values)
        print(' | '.join(map(str, (world, *key, round(median, 6), round(mad, 6),
              round(statistics.median(r['durable_run_commit_seconds'] for r in samples), 6),
              max(r['smi_peak_mib_total'] for r in samples)))))
    baseline = [r for r in rows if r['config_backend'] == 'cayleypy' and r['phase'] == 'measure']
    assert len(baseline) == 5
    print('baseline', statistics.median(r['search_complete_seconds'] for r in baseline),
          'peak MiB', max(r['smi_peak_mib_total'] for r in baseline))


if __name__ == '__main__':
    for path in sys.argv[1:]:
        audit(path)
