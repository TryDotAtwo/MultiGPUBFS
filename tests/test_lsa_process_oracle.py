import json
import hashlib
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import replay_lsa_cancel_candidate as replay
from test_export_hf_dataset import frame


class ProcessOracleTests(unittest.TestCase):
    def test_warmup_gate_requires_both_authenticated_ranks_to_complete_warmup(self):
        def warmed(rank, record):
            record['warmup_completed'] = True
        self.assertEqual(self.check(result_mutation=warmed, expected_warmup=True)['unique_states'], 24)
        for bad in (False, None, 1, 'true'):
            def asymmetric(rank, record):
                record['warmup_completed'] = bad if rank == 1 else True
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, 'PROCESS_ORACLE_WARMUP'):
                    self.check(result_mutation=asymmetric, expected_warmup=True)

    def test_reference_archive_and_bootstrap_have_distinct_authenticated_digests(self):
        def record(rank, value):
            value['run_contract'] = 'reference_bench'
            value['bootstrap_digest'] = list(bytes.fromhex('cd' * 32))
        def marker(value):
            value['bootstrap_digest'] = list(bytes.fromhex('cd' * 32))
        self.assertEqual(self.check(result_mutation=record, marker_mutation=marker)['unique_states'], 24)

    def test_typed_contract_cannot_replace_archive_digest_even_with_matching_rank_commits(self):
        def record(rank, value):
            value['bootstrap_digest'] = list(bytes.fromhex('cd' * 32))
        def marker(value):
            value['bootstrap_digest'] = list(bytes.fromhex('cd' * 32))
        with self.assertRaisesRegex(ValueError, 'PROCESS_ORACLE_GROUP_COMMIT'):
            self.check(result_mutation=record, marker_mutation=marker)

    def check(self, mutation=None, result_mutation=None, expected_seed=None, expected_epoch_window=None,
              expected_config_digest=None, expected_run_contract=None, expected_route_banks=None,
              require_bank_reuse=False, marker_mutation=None, expected_owner_backend=None,
              expected_warmup=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'result').mkdir()
            # Hand-enumerated L/R/X layers, independent of the replay oracle.
            words = ['0123', '1230 3012 1023', '2301 2130 0312 0231 3102',
                     '3201 1302 0213 3120 2031 2310', '2013 1320 3021 1203 3210',
                     '0132 0321 2103', '1032']
            expected = [{bytes(int(column == int(word[row]))
                               for row in range(4) for column in range(4))
                         for word in layer.split()} for layer in words]
            rows = [[], []]
            for depth, layer in enumerate(expected):
                for i, state in enumerate(sorted(layer)):
                    rows[i % 2].append((depth, state))
            if mutation:
                mutation(rows)
            for rank in range(2):
                counts = [sum(d == depth for d, _ in rows[rank])
                          for depth in range(len(expected))]
                record = dict(status='COMPLETE', local_layer_sizes=counts,
                              hash_seed_hex='00000000000000000000000000000001', epoch_window=3,
                              run_contract='RunConfigV1', route_banks=3, route_bank_reuses=2,
                              rank=rank, world_size=2, archive_commit_scope='file_fsync', owner_backend='CUCO_RANK',
                              bootstrap_digest=list(bytes.fromhex('ab' * 32)))
                if result_mutation:
                    result_mutation(rank, record)
                (root / f'result/rank-{rank}.json').write_text(json.dumps(record))
                header = b'MGBFSAR1' + struct.pack('<Q', 16) + bytes.fromhex('ab' * 32)
                chain, sequence, pieces = hashlib.sha256(header).digest(), 0, [header]
                for depth, count in enumerate(counts):
                    if count:
                        states = [state for d, state in rows[rank] if d == depth]
                        item, chain = frame(chain, sequence, 1, depth, count,
                                            b''.join(states) + bytes(16 * count))
                        pieces.append(item)
                        sequence += 1
                    item, chain = frame(chain, sequence, 2, depth, count, b'')
                    pieces.append(item)
                    sequence += 1
                item, _ = frame(chain, sequence, 3, len(counts), sum(counts), b'')
                pieces.append(item)
                (root / f'archive-rank-{rank}.mgbfsar1').write_bytes(b''.join(pieces))
            marker = dict(schema='mgbfs-group-run-commit-v1', status='COMPLETE', world_size=2,
                          bootstrap_digest=list(bytes.fromhex('ab' * 32)),
                          archive_commit_scope='file_fsync', rank_sha256=[
                              list(hashlib.sha256((root / f'result/rank-{r}.json').read_bytes()).digest())
                              for r in range(2)])
            if marker_mutation:
                marker_mutation(marker)
            if marker is not None and marker.get('schema') != 'omit-marker':
                (root / 'result/group-complete.json').write_text(json.dumps(marker))
            return replay.verify_process_archives(root, expected_seed=expected_seed,
                expected_epoch_window=expected_epoch_window, expected_config_digest=expected_config_digest,
                expected_run_contract=expected_run_contract, expected_route_banks=expected_route_banks,
                require_bank_reuse=require_bank_reuse, expected_owner_backend=expected_owner_backend,
                expected_warmup=expected_warmup)

    def test_rank_cannot_silently_substitute_requested_owner_backend(self):
        self.check(expected_owner_backend='CUCO_RANK')
        with self.assertRaisesRegex(ValueError, 'PROCESS_ORACLE_OWNER_BACKEND'):
            self.check(expected_owner_backend='BMMA_BUCKET')

    def test_group_commit_must_authenticate_exact_rank_results_and_scope(self):
        for key, value in [('schema', 'omit-marker'), ('schema', 'unknown'), ('status', 'INCOMPLETE'),
                           ('world_size', 1), ('bootstrap_digest', [0] * 32),
                           ('archive_commit_scope', 'search_only'), ('rank_sha256', []),
                           ('rank_sha256', [[0] * 32, [0] * 32])]:
            def change(marker):
                marker[key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, 'PROCESS_ORACLE_GROUP_COMMIT'):
                    self.check(marker_mutation=change)

    def test_per_rank_identity_and_archive_scope_are_required(self):
        for key, value in [('rank', 9), ('world_size', 1), ('archive_commit_scope', 'search_only')]:
            def change(rank, record):
                if rank == 1:
                    record[key] = value
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, 'PROCESS_ORACLE_RANK_IDENTITY_OR_SCOPE'):
                    self.check(result_mutation=change)

    def test_group_commit_also_requires_each_rank_bootstrap_to_match_archive(self):
        def change(rank, record):
            if rank == 1:
                record['bootstrap_digest'] = [0] * 32
        with self.assertRaisesRegex(ValueError, 'PROCESS_ORACLE_GROUP_COMMIT'):
            self.check(result_mutation=change)

    def test_reuse_gate_requires_positive_aggregate_actual_reuse(self):
        self.assertEqual(self.check(require_bank_reuse=True)['route_bank_reuses'], [2, 2])
        for value in (None, 0, -1, True, '1'):
            def change(rank, record):
                record['route_bank_reuses'] = value
            with self.assertRaisesRegex(ValueError, 'PROCESS_ORACLE_ROUTE_BANK_REUSE'):
                self.check(result_mutation=change, require_bank_reuse=True)
        def asymmetric(rank, record):
            record['route_bank_reuses'] = 0 if rank == 0 else 3
        self.assertEqual(self.check(result_mutation=asymmetric, require_bank_reuse=True)
                         ['route_bank_reuses'], [0, 3])

    def test_route_bank_evidence_matches_both_ranks(self):
        self.assertEqual(self.check(expected_route_banks=3)['unique_states'], 24)
        for rank in (0, 1):
            for value in (None, 2):
                def change(current, record):
                    if current == rank:
                        record['route_banks'] = value
                with self.assertRaisesRegex(ValueError, 'PROCESS_ORACLE_ROUTE_BANKS'):
                    self.check(result_mutation=change, expected_route_banks=3)

    def test_typed_run_evidence_must_match_requested_contract_and_digest(self):
        self.assertEqual(self.check(expected_config_digest='ab' * 32,
            expected_run_contract='RunConfigV1')['unique_states'], 24)
        with self.assertRaisesRegex(ValueError, 'PROCESS_ORACLE_REQUESTED_CONFIG'):
            self.check(expected_config_digest='cd' * 32)
        for rank in (0, 1):
            for key, value, error in [('run_contract', 'reference_bench', 'PROCESS_ORACLE_RUN_CONTRACT'),
                                      ('bootstrap_digest', [0] * 32, 'PROCESS_ORACLE_BOOTSTRAP_CONFIG')]:
                def change(current, record):
                    if current == rank:
                        record[key] = value
                with self.assertRaisesRegex(ValueError, error):
                    self.check(result_mutation=change, expected_config_digest='ab' * 32,
                        expected_run_contract='RunConfigV1')

    def test_requested_epoch_window_matches_both_ranks(self):
        self.assertEqual(self.check(expected_epoch_window=3)['unique_states'], 24)
        for rank in (0, 1):
            for missing in (False, True):
                def change(current, record):
                    if current == rank:
                        if missing:
                            record.pop('epoch_window')
                        else:
                            record['epoch_window'] = 2
                with self.assertRaisesRegex(ValueError, 'PROCESS_ORACLE_EPOCH_WINDOW'):
                    self.check(result_mutation=change, expected_epoch_window=3)

    def test_requested_seed_must_match_each_runtime_rank(self):
        seed = '00000000000000000000000000000001'
        self.assertEqual(self.check(expected_seed=seed)['unique_states'], 24)
        for rank in (0, 1):
            def change(current, record):
                if current == rank:
                    record['hash_seed_hex'] = '00000000000000000000000000000000'
            with self.assertRaisesRegex(ValueError, 'PROCESS_ORACLE_HASH_SEED'):
                self.check(result_mutation=change, expected_seed=seed)

    def test_missing_runtime_seed_cannot_validate_requested_seed(self):
        def change(rank, record):
            record.pop('hash_seed_hex')
        with self.assertRaisesRegex(ValueError, 'PROCESS_ORACLE_HASH_SEED'):
            self.check(result_mutation=change, expected_seed='00000000000000000000000000000001')

    def test_all_layers_from_two_process_archives_match_independent_oracle(self):
        row = self.check()
        self.assertEqual(row['unique_states'], 24)
        self.assertEqual(row['layer_sizes'], [1, 3, 5, 6, 5, 3, 1])

    def test_same_total_at_wrong_depth_is_rejected(self):
        def change(rows):
            depth, state = rows[0][0]
            rows[0][0] = (depth + 1, state)
        with self.assertRaisesRegex(ValueError, 'PROCESS_ORACLE_LAYER'):
            self.check(change)

    def test_duplicate_across_rank_archives_is_rejected(self):
        def change(rows):
            rows[1].append(rows[0][0])
        with self.assertRaisesRegex(ValueError, 'PROCESS_ORACLE_DUPLICATE'):
            self.check(change)


if __name__ == '__main__':
    unittest.main()
