import unittest
import replay_lsa_cancel_candidate as replay


class UnitriangularOracleTests(unittest.TestCase):
    def test_inverse_closed_u3_mod2_is_eight_cycle(self):
        self.assertTrue(hasattr(replay, 'u_reference_layers'),
                        'unitriangular full-state oracle is missing')
        layers = replay.u_reference_layers(3, 2)
        self.assertEqual(list(map(len, layers)), [1, 2, 2, 2, 1])
        self.assertEqual(layers[0], {bytes([1,0,0,0,1,0,0,0,1])})
        self.assertEqual(layers[2], {bytes([1,1,1,0,1,1,0,0,1]),
                                   bytes([1,1,0,0,1,1,0,0,1])})

    def test_mod3_uses_inverse_moves_not_only_positive_generator(self):
        self.assertTrue(hasattr(replay, 'u_reference_layers'))
        self.assertEqual(replay.u_reference_layers(2, 3),
                         [{bytes([1,0,0,1])},
                          {bytes([1,1,0,1]), bytes([1,2,0,1])}])

    def test_bounded_oracle_rejects_unbounded_inputs(self):
        self.assertTrue(hasattr(replay, 'u_reference_layers'))
        for n, modulus in [(5,2), (4,7), (1,2), (4,1)]:
            with self.assertRaisesRegex(ValueError, 'BOUNDED_REFERENCE'):
                replay.u_reference_layers(n, modulus)
