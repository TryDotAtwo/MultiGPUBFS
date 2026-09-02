import unittest


try:
    from multigpubfs.wire_model import WireFormat, estimate_wire_bytes
except (ImportError, ModuleNotFoundError):
    WireFormat = None
    estimate_wire_bytes = None


class WireModelTests(unittest.TestCase):
    def test_eager_and_two_phase_account_for_exact_payload_components(self):
        """Catches omitting control bitmaps or accepted parent metadata."""
        self.assertTrue(callable(estimate_wire_bytes), "wire model API is missing")
        wire = WireFormat(key_bytes=8, parent_bytes=8, move_bytes=1)
        estimate = estimate_wire_bytes(
            remote_candidates=100,
            remote_only_accepted=10,
            accept_bitmap_bytes=16,
            wire_format=wire,
        )

        self.assertEqual(estimate.eager_bytes, 1700)
        self.assertEqual(estimate.two_phase_key_bytes, 800)
        self.assertEqual(estimate.two_phase_control_bytes, 16)
        self.assertEqual(estimate.two_phase_metadata_bytes, 90)
        self.assertEqual(estimate.two_phase_total_bytes, 906)
        self.assertAlmostEqual(estimate.two_phase_reduction_fraction, 1 - 906 / 1700)

    def test_rejects_impossible_remote_acceptance_count(self):
        """Catches a byte estimate detached from routing accounting."""
        with self.assertRaises(ValueError):
            estimate_wire_bytes(
                remote_candidates=2,
                remote_only_accepted=3,
                accept_bitmap_bytes=1,
                wire_format=WireFormat(key_bytes=2, parent_bytes=2, move_bytes=1),
            )


if __name__ == "__main__":
    unittest.main()
