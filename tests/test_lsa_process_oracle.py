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
    def check(self, mutation=None, result_mutation=None, expected_seed=None, expected_epoch_window=None,
              expected_config_digest=None, expected_run_contract=None):
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
                              run_contract='RunConfigV1', bootstrap_digest=list(bytes.fromhex('ab' * 32)))
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
            return replay.verify_process_archives(root, expected_seed=expected_seed,
                expected_epoch_window=expected_epoch_window, expected_config_digest=expected_config_digest,
                expected_run_contract=expected_run_contract)

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
