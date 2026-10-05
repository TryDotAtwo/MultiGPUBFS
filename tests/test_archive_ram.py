import unittest
from scripts.archive_ram import plan,SLOT_BYTES


class HostRingTests(unittest.TestCase):
    def test_target_matches_total_vram_across_pairs(self):
        for world,per_gpu,host in [(2,12<<30,64<<30),(8,24<<30,512<<30),(8,288<<30,4<<40)]:
            cfg=dict(archive_storage_mode="ram",host_available_bytes=host,gpu_inventory=[dict(total_bytes=per_gpu,free_bytes=per_gpu//2)]*world)
            for n in [2,17,33,128]:
                result=plan(cfg,n,world)
                self.assertEqual(result['total_pinned_bytes'],world*per_gpu)
                self.assertFalse(result['host_limited'])
                raw=n*result['rows_per_slot']
                self.assertEqual(((raw+65535)//65536)*65536,SLOT_BYTES)

    def test_hybrid_reserve(self):
        result=plan(dict(host_available_bytes=64<<30,gpu_inventory=[dict(total_bytes=12<<30)]*2),12,2)
        self.assertEqual(result['total_pinned_bytes'],6<<30)
        self.assertEqual(result['ssd_queue_target_bytes'],18<<30)
        self.assertEqual(result['initial_slots_per_rank'],64)
        self.assertGreater(result['slots_per_rank'],64)
        with self.assertRaises(ValueError):
            plan(dict(archive_ram_slots=16,archive_initial_slots=17),12,2)

    def test_host_limit_preserves_workspace_and_os_reserve(self):
        cfg=dict(host_available_bytes=8<<30,gpu_inventory=[dict(total_bytes=12<<30)]*2)
        result=plan(cfg,128,2)
        self.assertTrue(result['host_limited'])
        self.assertLessEqual(result['total_pinned_bytes']+result['workspace_reserve_bytes'],6<<30)
        self.assertGreater(result['slots_per_rank'],4)

    def test_compact_export_does_not_pin_a_vram_sized_pool(self):
        cfg=dict(host_available_bytes=4<<40,gpu_inventory=[dict(total_bytes=288<<30)]*8,
                 retention_policy='last_complete_small_1000')
        for n in [2,17,33,128]:
            result=plan(cfg,n,8)
            self.assertEqual(result['total_pinned_bytes'],256<<20)
            self.assertEqual(result['slots_per_rank'],32)
            self.assertEqual(result['slot_bytes'],1<<20)
            self.assertGreaterEqual(result['rows_per_slot'],1000)

    def test_explicit_geometry_and_invalid_values(self):
        cfg=dict(host_available_bytes=64<<30,archive_ram_slots=16,gpu_inventory=[dict(total_bytes=12<<30)]*2)
        self.assertEqual(plan(cfg,12,2)['slots_per_rank'],16)
        for value in [True,0,1,65537,16.0]:
            with self.assertRaises(ValueError):plan(dict(cfg,archive_ram_slots=value),12,2)
        with self.assertRaises(ValueError):plan(dict(cfg,host_available_bytes=128<<20),12,2)
        with self.assertRaises(ValueError):plan(dict(cfg,gpu_inventory=[dict(total_bytes=12<<30)]),12,2)


if __name__=='__main__':unittest.main()
