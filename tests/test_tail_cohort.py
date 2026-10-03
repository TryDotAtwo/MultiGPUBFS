import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

from scripts.tail_cohort import publication_cases, merge
from scripts.publish_tail_batch import plan, publish


class CohortTests(unittest.TestCase):
    def fixture(self, root, n, r, width=8):
        key = f'n{n}-m{r}'
        saved = root/key/'saved'
        saved.mkdir(parents=True)
        files = []
        for depth, count, offset in ((4, 3, 7), (5, 6, 0)):
            data = b''.join((depth*100+i).to_bytes(width, 'little') for i in range(count))
            p = saved/f'{depth}.bin'
            p.write_bytes(data)
            files.append(dict(path=p.name, depth=depth, states=count, bytes=len(data),
                first_state_ordinal=offset, full_layer=offset == 0,
                sha256=hashlib.sha256(data).hexdigest()))
        m = dict(graph=dict(n=n, r=r), packing=dict(bytes_per_state=width),
                 files=files, status='INCOMPLETE', layers=[dict(depth=4, states=10),
                 dict(depth=5, states=6)], program_commit='fixture')
        (saved/'manifest.json').write_text(json.dumps(m))
        return key, saved

    def ledger(self, keys):
        return dict(configuration=dict(base=dict(run_id='cohort-test',
                    archive_format='parquet_cohort')),
                    cases={key: dict(attempted=True) for key in keys})

    def test_shared_payload_preserves_case_layer_offsets_and_uploads_once(self):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keys = [self.fixture(root, n, r)[0] for n, r in ((7, 2), (8, 3))]
            ledger = self.ledger(keys)
            payloads, manifests, size = plan(root, ledger=ledger)
            self.assertEqual(len(payloads), 1)
            self.assertEqual(len(manifests), 2)
            self.assertEqual(size, payloads[0][0].stat().st_size)
            table = pq.read_table(payloads[0][0])
            self.assertEqual(table.num_rows, 18)
            for (path, _), (n, r) in zip(manifests, ((7, 2), (8, 3))):
                m = json.loads(path.read_text())
                self.assertEqual(m['retained_packed_bytes'], 72)
                self.assertEqual(m['layers'][0]['states'], 10)
                self.assertFalse(m['retained_layers'][0]['full_layer'])
                entry = m['files'][0]
                self.assertEqual(entry['states'], 18)
                self.assertEqual(entry['case_states'], 9)
                for span in entry['layers']:
                    rows = table.slice(span['file_row_offset'], span['states']).to_pylist()
                    self.assertTrue(all(row['n'] == n and row['r'] == r for row in rows))
                    self.assertEqual([row['ordinal'] for row in rows], list(range(
                        span['first_state_ordinal'], span['first_state_ordinal']+span['states'])))
            calls = []
            class Api:
                def create_commit(self, **kwargs):
                    calls.append([op.path_in_repo for op in kwargs['operations']])
                    return 'receipt'
            publish(root, 'fixture/data', Api(), ledger=ledger)
            self.assertEqual(calls[0], [payloads[0][1]])
            self.assertEqual(len(calls[1]), 3)
            # The actual automatic-run verifier must read the shared payload
            # once, while comparing both independent per-case manifests.
            (root/'sweep.json').write_text(json.dumps(ledger))
            contents = {remote: path.read_bytes() for path, remote in payloads+manifests}
            contents[f'tail-sweeps/{root.name}/sweep.json'] = (root/'sweep.json').read_bytes()
            readback = []
            class Response:
                status_code = 200
                def __init__(self, data): self.data = data
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def raise_for_status(self): pass
                def json(self): return json.loads(self.data)
                def iter_content(self, chunk_size): yield self.data
            class Session:
                def __init__(self): self.headers = {}
                def get(self, url, **kwargs):
                    remote = url.split('/resolve/fixture/', 1)[1]
                    readback.append(remote)
                    return Response(contents[remote])
            # This script's imports also support direct execution on Linux.
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
            from scripts import run_auto_tail
            api = SimpleNamespace(repo_info=lambda *a, **k: SimpleNamespace(sha='fixture'))
            with patch('requests.Session', Session):
                verified = run_auto_tail.verify_hf(root, 'fixture/data', api, 'fixture-token')
            self.assertEqual(verified['files'], 1)
            self.assertEqual(verified['manifests_verified'], 2)
            self.assertEqual(readback.count(payloads[0][1]), 1)

    def test_widths_are_separate_and_complete_groups_are_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keys = [self.fixture(root, 7, 2)[0], self.fixture(root, 24, 10, 16)[0]]
            first = publication_cases(root, self.ledger(keys), group_size=2)
            before = {key: (p, p.stat().st_mtime_ns) for key, (_, p) in first.items()}
            keys.append(self.fixture(root, 8, 3)[0])
            second = publication_cases(root, self.ledger(keys), group_size=2)
            for key, (path, modified) in before.items():
                self.assertEqual(second[key][1], path)
                self.assertEqual(path.stat().st_mtime_ns, modified)
            self.assertNotEqual(first[keys[0]][0], first[keys[1]][0])

    def test_twenty_small_cases_share_one_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = [(9, r) for r in range(1, 10)] + [(10, r) for r in range(1, 11)] + [(11, 1)]
            keys = [self.fixture(root, n, r)[0] for n, r in pairs]
            payloads, manifests, _ = plan(root, ledger=self.ledger(keys))
            self.assertEqual(len(payloads), 1)
            self.assertEqual(len(manifests), 20)
            for path, _ in manifests:
                m = json.loads(path.read_text())
                self.assertEqual(m['files'][0]['states'], 180)
                self.assertEqual(m['files'][0]['case_states'], 9)

    def test_byte_bound_closes_stable_group_before_twenty_cases(self):
        from scripts.tail_cohort import case_groups
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keys = [self.fixture(root, n, 2)[0] for n in (7, 8)]
            ledger = self.ledger(keys)
            groups = list(case_groups(root, ledger, group_bytes=100))
            self.assertEqual([(len(m), closed) for m, closed in groups], [(2, True)])
            first = publication_cases(root, ledger, group_bytes=100)
            keys.append(self.fixture(root, 9, 2)[0])
            second = publication_cases(root, self.ledger(keys), group_bytes=100)
            self.assertEqual(first[keys[0]], second[keys[0]])
            self.assertNotEqual(second[keys[0]][0], second[keys[2]][0])

    def test_verified_release_reuses_deleted_payload_and_preserves_provenance(self):
        from scripts.release_tail_cohorts import release
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keys = [self.fixture(root, n, 2)[0] for n in (7, 8)]
            ledger = self.ledger(keys)
            ledger['configuration']['base']['cohort_group_size'] = 2
            publications = publication_cases(root, ledger)
            contents = {}
            for key, (base, manifest_path) in publications.items():
                contents[f'tail-runs/cohort-test-{key}/manifest.json'] = manifest_path.read_bytes()
                for entry in json.loads(manifest_path.read_text())['files']:
                    contents[entry['repo_path']] = (base/entry['path']).read_bytes()
            class Response:
                def __init__(self, data): self.data = data
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def raise_for_status(self): pass
                def json(self): return json.loads(self.data)
                def iter_content(self, chunk_size): yield self.data
            class Session:
                def __init__(self): self.headers = {}
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def get(self, url, **kwargs):
                    return Response(contents[url.split('/resolve/fixture/',1)[1]])
            api = SimpleNamespace(repo_info=lambda *a, **k: SimpleNamespace(sha='fixture'))
            with patch('requests.Session', Session):
                result = release(root, ledger, 'fixture/data', api, 'fixture', group_size=2)
            self.assertEqual(set(result['released_cases']), set(keys))
            self.assertFalse(list(root.rglob('*.bin')))
            self.assertFalse(list(root.rglob('*.parquet')))
            for key in keys:
                self.assertTrue((root/key/'saved/manifest.json').exists())
                self.assertTrue(publications[key][1].exists())
            # Planning for the same repository uses the verified receipt and
            # never attempts conversion from deleted packed states.
            payloads, manifests, size = plan(root, ledger=ledger, released_repo='fixture/data')
            self.assertEqual((payloads, size), ([], 0))
            self.assertEqual(len(manifests), 2)
            with self.assertRaisesRegex(ValueError, 'verified HF release receipt'):
                plan(root, ledger=ledger, released_repo='different/data')

    def test_corrupt_remote_readback_does_not_release_any_local_states(self):
        from scripts.release_tail_cohorts import release
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key, saved = self.fixture(root, 7, 2)
            ledger = self.ledger([key])
            publications = publication_cases(root, ledger, group_size=1)
            m = json.loads(publications[key][1].read_text())
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def raise_for_status(self): pass
                def json(self): return m
                def iter_content(self, chunk_size): yield b'corrupt remote data'
            class Session:
                def __init__(self): self.headers = {}
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def get(self, *a, **k): return Response()
            api = SimpleNamespace(repo_info=lambda *a, **k: SimpleNamespace(sha='fixture'))
            with patch('requests.Session', Session):
                with self.assertRaises(ValueError):
                    release(root, ledger, 'fixture/data', api, 'fixture', group_size=1)
            self.assertEqual(len(list(saved.rglob('*.bin'))), 2)
            self.assertTrue(list(publications[key][0].glob('*.parquet')))
            self.assertFalse((publications[key][0]/'release-receipt.json').exists())

    def test_shard_bound_and_corrupt_input_never_commit_cohort(self):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            members = [self.fixture(root, 7, 2), self.fixture(root, 8, 3)]
            destination = root/'merged'
            merge(members, destination, 'tail-shards/fixture/', shard_bytes=22*4)
            summary = json.loads((destination/'cohort.json').read_text())
            self.assertEqual(len(summary['files']), 5)
            self.assertEqual(sum(e['states'] for e in summary['files']), 18)
            for entry in summary['files']:
                self.assertLessEqual(entry['states'], 4)
                self.assertEqual(pq.read_metadata(destination/entry['path']).num_rows,
                                 entry['states'])
            saved = members[0][1]
            entry = json.loads((saved/'manifest.json').read_text())['files'][0]
            payload = saved/entry['path']
            payload.write_bytes(b'!' * payload.stat().st_size)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                merge(members, root/'bad', 'tail-shards/fixture/')
            self.assertFalse((root/'bad/cohort.json').exists())
            self.assertFalse((root/'bad/n7-m2.json').exists())


if __name__ == '__main__':
    unittest.main()
