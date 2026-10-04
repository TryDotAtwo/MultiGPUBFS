import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from sweep_tail_bfs import execute,pairs,automatic_pairs,unsupported_reason,resource_stop,allocation_failure


class SweepTests(unittest.TestCase):
    def test_explicit_interruption_retry_uses_fresh_case_path(self):
        from paired_tail import publication_records
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);grid=[(4,1)];base={'run_id':'retry'}
            def fake(config,source,case,runtime):
                case.mkdir();path=case/'manifest.json'
                path.write_text(json.dumps(dict(status='COMPLETE',last_completed_layer=0,stop_reason='exhausted')))
                return path
            ledger=execute(base,root,root,{},grid,10,fake)
            original=dict(ledger['cases']['n4-m1'])
            ledger['publication_history']=publication_records(ledger)
            ledger['cases'].pop('n4-m1')
            ledger['retry_case_keys']={'n4-m1':'n4-m1-retry1'}
            (root/'sweep.json').write_text(json.dumps(ledger))
            result=execute(base,root,root,{},grid,10,fake)
            self.assertEqual(result['pending'],[])
            self.assertEqual(result['cases']['n4-m1']['publication_key'],'n4-m1-retry1')
            records=publication_records(result)
            self.assertEqual(records['n4-m1'],original)
            self.assertEqual(list(records),['n4-m1','n4-m1-retry1'])
    def test_controlled_stop_preserves_case_and_leaves_other_pairs_pending(self):
        reason=[None];calls=[]
        def fake(config,source,case,runtime):
            calls.append((config['n'],config['r']));case.mkdir()
            path=case/'manifest.json'
            reason[0]='requested cancellation: SIGTERM'
            path.write_text(json.dumps(dict(status='INCOMPLETE',last_completed_layer=3,
                stop_reason=reason[0])))
            return path
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            ledger=execute({},root,root,{},[(4,1),(5,1),(5,2)],10,fake,
                should_stop=lambda:reason[0])
            self.assertEqual(calls,[(4,1)])
            self.assertEqual(ledger['pending'],[[5,1],[5,2]])
            self.assertEqual(ledger['global_stop_reason'],reason[0])
            self.assertEqual(ledger['cases']['n4-m1']['last_completed_layer'],3)
            self.assertFalse(resource_stop(ledger['cases']['n4-m1']))

    def test_automatic_cli_without_range_or_gpu_config(self):
        script=Path(__file__).resolve().parents[1]/'scripts/sweep_tail_bfs.py'
        result=subprocess.run([sys.executable,str(script),'--plan-only'],capture_output=True,text=True,check=True)
        grid=json.loads(result.stdout)
        self.assertEqual({tuple(pair) for pair in grid},set(pairs(2,128)))
        self.assertEqual(len(grid),8255)
        self.assertLess(grid.index([16,1]),grid.index([17,1]))
        self.assertLess(grid.index([16,1]),grid.index([16,2]))

    def test_resource_pruning_keeps_other_r_independent_and_resumes(self):
        calls=[]
        def fake(config,source,case,runtime):
            pair=(config['n'],config['r']);calls.append(pair)
            case.mkdir();path=case/'manifest.json'
            failed=pair==(16,3)
            path.write_text(json.dumps(dict(status='INCOMPLETE' if failed else 'COMPLETE',
                last_completed_layer=3,stop_reason='CUDA_ERROR_OUT_OF_MEMORY' if failed else 'exhausted')))
            return path
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);grid=automatic_pairs(16,18,3,4);base={'run_id':'prune'}
            ledger=execute(base,root,root,{},grid,10,fake)
            self.assertEqual(calls,[(16,3),(16,4),(17,4),(18,4)])
            for n in (17,18):
                item=ledger['cases'][f'n{n}-m3']
                self.assertFalse(item['attempted']);self.assertEqual(item['pruned_by'],{'n':16,'r':3})
            execute(base,root,root,{},grid,10,fake)
            self.assertEqual(len(calls),4)

    def test_upload_and_worker_io_errors_do_not_prune(self):
        for reason in ('HF HTTP 429','ConnectionError','ARCHIVE_WORKER_FATAL WRITE','search deadline',
                'GROUP_STATE_RING_RETIRE_FATAL','ARCHIVE_PIN_RING_FATAL: receiving on an empty channel',
                'no space left on device','REMOTE_SEARCH_CANCELLED','out of memory','capacity exhausted'):
            self.assertFalse(resource_stop(dict(status='INCOMPLETE',attempted=True,reason=reason)))

    def test_ambiguous_failure_keeps_later_same_r_eligible(self):
        for reason in ('GROUP_STATE_RING_RETIRE_FATAL','GROUP_STATE_RING_RETIRE_FATAL_17',
                'ARCHIVE_PIN_RING_FATAL: receiving on an empty channel','search deadline'):
            calls=[]
            def fake(config,source,case,runtime):
                calls.append((config['n'],config['r']));case.mkdir();path=case/'manifest.json'
                path.write_text(json.dumps(dict(status='INCOMPLETE',last_completed_layer=-1,stop_reason=reason)))
                return path
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);v=execute({},root,root,{},[(4,1),(5,1),(5,2)],10,fake)
                self.assertEqual(calls,[(4,1),(5,1),(5,2)])
                self.assertFalse(any('pruned_by' in x for x in v['cases'].values()))

    def test_ssd_full_stops_globally_without_branch_pruning(self):
        calls=[]
        def fake(config,source,case,runtime):
            calls.append((config['n'],config['r']));case.mkdir();path=case/'manifest.json'
            path.write_text(json.dumps(dict(status='INCOMPLETE',last_completed_layer=-1,
                stop_reason='no space left on device')));return path
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);v=execute({},root,root,{},[(4,1),(5,1),(5,2)],10,fake)
            self.assertEqual(calls,[(4,1)]);self.assertEqual(v['pending'],[[5,1],[5,2]])
            self.assertIn('SSD full',v['global_stop_reason'])

    def test_native_layer_capacity_code_is_specific(self):
        for reason in ('native fatal: LIBRARY_RANK_DEPTH_FATAL_16_16,ARCHIVE_INCOMPLETE',
                       'LIBRARY_RANK_DEPTH_FATAL_16_0','LIBRARY_RANK_DEPTH_FATAL_0_16',
                       'GROUP_STATE_RING_RETIRE_FATAL_11','GROUP_STATE_RING_RETIRE_FATAL_12',
                       'GROUP_STATE_RING_RETIRE_FATAL_16'):
            self.assertTrue(resource_stop(dict(status='INCOMPLETE',attempted=True,reason=reason)))
        for reason in ('LIBRARY_RANK_DEPTH_FATAL_10_10','LIBRARY_RANK_DEPTH_FATAL_160_160',
                       'REMOTE_NEXT_EXTENT_FATAL','REMOTE_SEARCH_CANCELLED','CUDA_SET_DEVICE'):
            self.assertFalse(resource_stop(dict(status='INCOMPLETE',attempted=True,reason=reason)))

    def test_native_status_two_requires_allocation_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();case=root/'case';case.mkdir()
            (source/'native.rs').write_text('check(cudaMalloc(&mut ptr, bytes));\ncheck(ncclCommInitRank());\n')
            log=case/'native.log'
            log.write_text('MGBFS_NATIVE_STATUS status=2 file=native.rs line=1\n')
            self.assertTrue(allocation_failure(case,source))
            log.write_text('MGBFS_NATIVE_STATUS status=2 file=native.rs line=2\n')
            self.assertFalse(allocation_failure(case,source))
            (source/'host.rs').write_text('check(cudaHostAlloc(&mut ptr, bytes, 0));\n')
            log.write_text('MGBFS_NATIVE_STATUS status=2 file=host.rs line=1\n')
            self.assertFalse(allocation_failure(case,source))
            self.assertFalse(resource_stop(dict(status='INCOMPLETE',attempted=True,reason='CUDA_STATUS_2')))

    def test_all_supported_pairs_attempted_once(self):
        calls=[]
        def fake(config,source,case,runtime):
            calls.append((config['n'],config['r']))
            case.mkdir();path=case/'manifest.json'
            path.write_text(json.dumps(dict(status='COMPLETE',last_completed_layer=0,stop_reason='exhausted')))
            return path
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);grid=automatic_pairs(2,8)
            ledger=execute({'run_id':'auto'},root,root,{},grid,120,fake)
            expected={pair for pair in pairs(2,8) if unsupported_reason(*pair) is None}
            self.assertEqual(set(calls),expected)
            self.assertEqual(len(calls),len(expected))
            self.assertEqual(ledger['pending'],[])
            self.assertEqual(len(ledger['cases']),35)

    def test_bounds(self):
        self.assertEqual(pairs(3,4,2,3),[(3,2),(3,3),(4,2),(4,3)])
        with self.assertRaises(ValueError): pairs(4,3)

    def test_timing_separates_runner_and_inter_pair_progress(self):
        from unittest.mock import patch
        clock=[100.0]
        def runner(config,source,case,runtime):
            case.mkdir();(case/'result').mkdir()
            clock[0]+=2
            for rank in range(2):
                (case/'result'/f'rank-{rank}.json').write_text(json.dumps(dict(search_complete_seconds=.25)))
            path=case/'manifest.json'
            path.write_text(json.dumps(dict(status='COMPLETE',last_completed_layer=0,
                stop_reason='exhausted',layers=[dict(seconds=.2)])))
            return path
        def progress(ledger):clock[0]+=3
        with tempfile.TemporaryDirectory() as tmp,patch('sweep_tail_bfs.time.monotonic',side_effect=lambda:clock[0]):
            root=Path(tmp)
            ledger=execute({},root,root,{},[(3,2),(4,3)],100,runner,on_progress=progress)
            first=ledger['cases']['n3-m2']['timing'];second=ledger['cases']['n4-m3']['timing']
            self.assertIsNone(first['transition_before_seconds'])
            self.assertEqual(second['transition_before_seconds'],3)
            self.assertEqual(second['runner_wall_seconds'],2)
            self.assertEqual(second['production_search_seconds'],.25)
            self.assertEqual(second['completed_layer_seconds'],.2)

    def test_resume_and_failure(self):
        calls=[]
        def fake(config,source,case,runtime):
            calls.append((config['n'],config['r']))
            if config['r']==2: raise RuntimeError('capacity')
            case.mkdir()
            path=case/'manifest.json'
            path.write_text(json.dumps(dict(status='COMPLETE',last_completed_layer=4,stop_reason='exhausted')))
            return path
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base={'run_id':'unit'};grid=pairs(3,3,1,2)
            ledger=execute(base,root,root,{},grid,10,fake)
            self.assertEqual(ledger['cases']['n3-m2']['status'],'INCOMPLETE')
            self.assertEqual(ledger['pending'],[])
            execute(base,root,root,{},grid,10,fake)
            self.assertEqual(calls,[(3,1),(3,2)])
            with self.assertRaises(ValueError): execute({'run_id':'changed'},root,root,{},grid,10,fake)

    def test_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=execute({},Path(tmp),Path(tmp),{},[(129,1)],10,
                           lambda *args: self.fail('unsupported case executed'))
            self.assertFalse(ledger['cases']['n129-m1']['attempted'])

if __name__=='__main__': unittest.main()
