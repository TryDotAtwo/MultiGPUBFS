import importlib.util
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location('capacity_replay', Path(__file__).with_name('replay_lsa_cancel_candidate.py'))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
class CapacityEvidenceTests(unittest.TestCase):
    def test_armed_ordinary_nccl_capacity_abort_is_reached(self):
        text = 'MGBFS_TEST_CAPACITY_ARMED rank=0 layer_capacity=2\nMGBFS_RUNTIME_FATAL rank=1 error=GROUP_STATE_RING_RETIRE_FATAL\n'
        self.assertTrue(replay.capacity_fault_reached(text, 0))
    def test_unarmed_retirement_failure_is_not_capacity_evidence(self):
        self.assertFalse(replay.capacity_fault_reached('MGBFS_RUNTIME_FATAL rank=0 error=GROUP_STATE_RING_RETIRE_FATAL\n', 0))
    def test_other_rank_arming_does_not_validate_requested_injection(self):
        text = 'MGBFS_TEST_CAPACITY_ARMED rank=1 layer_capacity=2\nGROUP_OWNER_OR_PRE_OWNER_FATAL\n'
        self.assertFalse(replay.capacity_fault_reached(text, 0))
    def test_cancellation_alone_is_not_capacity_evidence(self):
        text = 'MGBFS_TEST_CAPACITY_ARMED rank=0 layer_capacity=2\nREMOTE_SEARCH_CANCELLED\n'
        self.assertFalse(replay.capacity_fault_reached(text, 0))
if __name__ == '__main__': unittest.main()
