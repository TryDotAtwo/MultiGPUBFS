import unittest
from pathlib import Path
import sys
import subprocess
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import replay_lsa_cancel_candidate as replay


class ReferenceSizes(unittest.TestCase):
    def test_cli_rejects_seed_before_creating_output_or_building(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'must-not-exist'
            result = subprocess.run([sys.executable, replay.__file__, temporary,
                                     str(output), '--hash-seed-hex', 'not-a-seed'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('HASH_SEED_HEX_32', result.stderr)
            self.assertFalse(output.exists())

    def test_seed_overrides_inherited_environment_and_matches_report(self):
        self.assertTrue(hasattr(replay, 'configure_hash_seed'))
        env = {'MGBFS_HASH_SEED_HEX': 'f' * 32}
        reported = replay.configure_hash_seed(env, '000000000000000000000000013527DC')
        self.assertEqual(reported, '000000000000000000000000013527dc')
        self.assertEqual(env['MGBFS_HASH_SEED_HEX'], reported)

    def test_seed_accepts_zero_and_rejects_malformed_before_launch(self):
        self.assertTrue(hasattr(replay, 'configure_hash_seed'))
        env = {}
        self.assertEqual(replay.configure_hash_seed(env, '0' * 32), '0' * 32)
        for value in ('', '1', '0x' + '0' * 32, 'g' * 32, '0' * 33):
            with self.assertRaises(ValueError):
                replay.configure_hash_seed(env, value)
        self.assertEqual(env['MGBFS_HASH_SEED_HEX'], '0' * 32)

    def test_s3_layers_and_canonical_matrix(self):
        self.assertTrue(hasattr(replay, 's_reference_layers'))
        layers = replay.s_reference_layers(3)
        self.assertEqual(list(map(len, layers)), [1, 3, 2])
        self.assertEqual(layers[0], {bytes([1, 0, 0, 0, 1, 0, 0, 0, 1])})
        self.assertEqual(len(set.union(*layers)), 6)

    def test_oracle_rejects_unbounded_sizes(self):
        self.assertTrue(hasattr(replay, 's_reference_layers'))
        for n in (0, 1, 9):
            with self.assertRaises(ValueError):
                replay.s_reference_layers(n)


if __name__ == '__main__':
    unittest.main()
