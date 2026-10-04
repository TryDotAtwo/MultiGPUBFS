import sys
import unittest
import threading
import tempfile
import json
import os
import time
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_auto_tail import device_budget,pair_config,tune_pair,SweepPublisher
import run_auto_tail


class AutomaticPlanningTests(unittest.TestCase):
    def test_publication_capacity_waits_only_between_cases_until_verified_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);saved = root/'n7-m1/saved';saved.mkdir(parents=True)
            payload = saved/'states.bin';payload.write_bytes(b'01234567')
            (saved/'manifest.json').write_text(json.dumps(dict(files=[dict(path=payload.name)])))
            ledger = dict(configuration=dict(grid=[[7,1]]),cases={'n7-m1':dict(attempted=True)})
            started = threading.Event();allow_release = threading.Event();finished = threading.Event()
            def publish(*args, **kwargs):
                started.set();allow_release.wait(3);payload.unlink();return {'released':True}
            publisher = SweepPublisher(root,'fixture',None,time.time()+3,publish)
            publisher.enqueue(ledger);self.assertTrue(started.wait(1))
            result = []
            def wait():
                result.append(publisher.wait_for_capacity(ledger,deadline=time.time()+3,max_pending_bytes=1))
                finished.set()
            waiter = threading.Thread(target=wait);waiter.start()
            self.assertFalse(finished.wait(.05))
            allow_release.set();self.assertTrue(finished.wait(2));waiter.join();publisher.finish()
            self.assertEqual(result,[True])

    def test_publication_capacity_deadline_retains_state_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);saved=root/'n7-m1/saved';saved.mkdir(parents=True)
            payload=saved/'states.bin';payload.write_bytes(b'01234567')
            (saved/'manifest.json').write_text(json.dumps(dict(files=[dict(path=payload.name)])))
            ledger=dict(cases={'n7-m1':dict(attempted=True)})
            publisher=SweepPublisher(root,'fixture',None,time.time()+3,lambda *a,**k:None)
            self.assertFalse(publisher.wait_for_capacity(ledger,deadline=time.time()-1,max_pending_bytes=1))
            self.assertEqual(payload.read_bytes(),b'01234567');publisher.finish()

    def test_startup_timeout_preserves_publishable_empty_incomplete_case(self):
        from publish_tail_batch import plan
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';binary=source/'target/release/mgbfs'
            binary.parent.mkdir(parents=True);binary.write_bytes(b'fixture binary')
            cfg=dict(n=7,r=1,world=2,env={},native_capacity_probe=True)
            with patch.object(run_auto_tail,'tune_pair',side_effect=TimeoutError('probe deadline')),\
                 patch.object(run_auto_tail.subprocess,'check_output',return_value='fixture-commit\n'),\
                 patch.object(run_auto_tail,'run') as runner:
                manifest=run_auto_tail.run_adaptive(cfg,source,root/'n7-m1',{},deadline=time.time()+60)
            runner.assert_not_called()
            data=json.loads(manifest.read_text())
            self.assertEqual((data['status'],data['last_completed_layer']),('INCOMPLETE',-1))
            self.assertEqual(data['files'],[])
            self.assertEqual(data['layers'],[])
            self.assertFalse(data['launch_config']['search_started'])
            ledger=dict(configuration=dict(base=dict(run_id='fixture')),cases={'n7-m1':dict(attempted=True)})
            payloads,manifests,size=plan(root,ledger=ledger)
            self.assertEqual((payloads,size),([],0))
            self.assertEqual(len(manifests),1)

    def test_probe_deadline_prevents_new_native_query(self):
        cfg=dict(n=15,r=4,world=2,env={},resource_plan=device_budget([{'free_bytes':12<<30}]))
        with patch('vram_autotune.native_query') as query:
            with self.assertRaises(TimeoutError):
                tune_pair(cfg,Path('source'),Path('case'),{},deadline=time.time()-1)
        query.assert_not_called()

    def test_automatic_reserve_preserves_explicit_settings_and_native_floor(self):
        self.assertEqual(run_auto_tail.automatic_reserve({},{}),str(256<<20))
        self.assertEqual(run_auto_tail.automatic_reserve({},
            {'MGBFS_VRAM_RESERVE_BYTES':str(64<<20)}),str(64<<20))
        self.assertEqual(run_auto_tail.automatic_reserve(
            {'MGBFS_VRAM_RESERVE_BYTES':str(1<<30)},{}),str(1<<30))
        for value in ('0','bad',str(2**64)):
            with self.assertRaises(ValueError):run_auto_tail.automatic_reserve(
                {'MGBFS_VRAM_RESERVE_BYTES':value},{})

    def test_adaptive_records_calibration_without_changing_buffers(self):
        config=dict(n=12,r=4,world=2,timeout_seconds=60,
            env={'MGBFS_TRANSPORT_BACKEND':'NCCL_LSA'},
            resource_plan=device_budget([{'free_bytes':12<<30}]))
        expected=pair_config(config,12,4)
        decision=dict(status='CALIBRATED',graph_batches=32,samples=['fixture'])
        with patch('calibrate_graph_profile.calibrate',return_value=decision) as calibration,\
             patch.object(run_auto_tail,'run',return_value=Path('manifest')) as runner:
            run_auto_tail.run_adaptive(config,Path('source'),Path('root/case'),{},
                                      deadline=time.time()+600)
        actual=runner.call_args.args[0]
        self.assertEqual(actual['graph_profile_selection'],decision)
        self.assertEqual(actual['env']['MGBFS_CUDA_GRAPH_BATCHES'],'32')
        for name in ('MGBFS_BENCH_CAPACITY','MGBFS_LIBRARY_POOL_BYTES','MGBFS_ARCHIVE_SLOTS'):
            self.assertEqual(actual['env'][name],expected['env'][name])
        self.assertNotIn('MGBFS_CUDA_GRAPH_BATCHES',config['env'])
        calibration.assert_called_once()

    def test_large_prefix_calibration_budget_is_bounded_outside_bfs(self):
        config=dict(n=15,r=4,world=2,timeout_seconds=60,
            env={'MGBFS_TRANSPORT_BACKEND':'NCCL_LSA'},
            resource_plan=device_budget([{'free_bytes':12<<30}]))
        with patch.object(run_auto_tail.time,'time',return_value=1000), \
             patch('calibrate_graph_profile.calibrate',return_value=dict(graph_batches=32)) as calibration, \
             patch.object(run_auto_tail,'run',return_value=Path('manifest')) as runner:
            run_auto_tail.run_adaptive(config,Path('source'),Path('root/case'),{},deadline=8200)
        self.assertEqual(calibration.call_args.kwargs['deadline'],1600)
        actual=runner.call_args.args[0]
        self.assertEqual(actual['graph_profile_selection']['startup_budget_seconds'],600)
        self.assertNotIn('MGBFS_CALIBRATION_LAYERS',actual['env'])

    def test_explicit_graph_profile_bypasses_calibration(self):
        config=dict(n=12,r=4,world=2,timeout_seconds=60,
            env={'MGBFS_TRANSPORT_BACKEND':'NCCL_LSA','MGBFS_CUDA_GRAPH_BATCHES':'0'},
            resource_plan=device_budget([{'free_bytes':12<<30}]))
        with patch('calibrate_graph_profile.calibrate') as calibration,\
             patch.object(run_auto_tail,'run',return_value=Path('manifest')):
            run_auto_tail.run_adaptive(config,Path('source'),Path('root/case'),{},
                                      deadline=time.time()+600)
        calibration.assert_not_called()

    def test_native_tuning_uses_actual_queries_and_preserves_fast_batch(self):
        base=dict(n=16,r=4,world=2,host_available_bytes=8<<30,env={},
                  resource_plan=device_budget([{'free_bytes':12<<30}]))
        original=json.loads(json.dumps(base))
        def query(cfg):
            self.assertEqual(cfg['batch'],32768)
            e=cfg['env'];rows=int(e['MGBFS_BENCH_CAPACITY'])
            self.assertEqual(int(e['MGBFS_FUTURE_CAPACITY']),2*rows)
            return [dict(rank=rank,required_bytes=rows*640+100000,
                reserve_bytes=1<<30,free_after_nccl_warmup_bytes=free)
                for rank,free in enumerate((12<<30,10<<30))]
        cfg=tune_pair(base,Path('fixture'),Path('case'),{},query=query)
        expected=((10<<30)-(1<<30)-100000)//640
        self.assertEqual(int(cfg['env']['MGBFS_BENCH_CAPACITY']),expected)
        self.assertGreater(expected,base['resource_plan']['max_rows_per_rank'])
        self.assertEqual(base,original)
        self.assertFalse(cfg['resource_plan']['maximum_hardware_capacity_proven'])

    def test_resident_admission_reuses_hardware_bound_but_not_small_orbit_bound(self):
        from types import SimpleNamespace
        session=SimpleNamespace(capacity_profiles={})
        base=dict(n=16,r=4,world=2,host_available_bytes=8<<30,env={},
                  resource_plan=device_budget([{'free_bytes':12<<30}]))
        def query(cfg,*args,**kwargs):
            rows=int(cfg['env']['MGBFS_BENCH_CAPACITY'])
            return [dict(rank=rank,required_bytes=rows*640+100000,
                reserve_bytes=1<<30,free_after_nccl_warmup_bytes=10<<30)
                for rank in range(2)]
        with patch('resident_session.active_session',return_value=session),\
             patch('vram_autotune.native_query',side_effect=query) as probe:
            first=tune_pair(base,Path('fixture'),Path('case'),{})
            calls=probe.call_count
            second=tune_pair(dict(base,r=5),Path('fixture'),Path('other'),{})
            self.assertEqual(probe.call_count,calls)
            self.assertTrue(second['resource_plan']['resident_admission_reused'])
            self.assertEqual(first['env']['MGBFS_BENCH_CAPACITY'],second['env']['MGBFS_BENCH_CAPACITY'])
            session.capacity_profiles.clear()
            tune_pair(dict(base,r=16),Path('fixture'),Path('small'),{})
            calls=probe.call_count
            tune_pair(base,Path('fixture'),Path('large'),{})
            self.assertGreater(probe.call_count,calls)

    def test_failed_sweep_or_publication_preserves_local_error_report(self):
        class FailedPublisher:
            closed=False;generations=0;error=None
            def enqueue(self,ledger):pass
            def finish(self):self.closed=True;raise RuntimeError('injected HF failure')
        for compute_failure in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                (root/'automatic-config.json').write_text(json.dumps(dict(resource_plan={})))
                (root/'runtime.json').write_text('{}')
                ledger=dict(cases={},pending=[[2,1]])
                (root/'sweep.json').write_text(json.dumps(ledger))
                args=['run_auto_tail','--source',str(root),'--runtime-env',str(root/'runtime.json'),
                      '--root',str(root),'--repo-id','fixture','--deadline-unix',str(time.time()+3600)]
                hub=SimpleNamespace(HfApi=lambda **kw:None,get_token=lambda:'fixture')
                with patch.object(sys,'argv',args),patch.dict(sys.modules,{'huggingface_hub':hub}),\
                     patch.object(run_auto_tail,'os',SimpleNamespace(name='posix',environ=os.environ)),\
                     patch.object(run_auto_tail,'SweepPublisher',return_value=FailedPublisher()),\
                     patch.object(run_auto_tail,'execute',return_value=ledger,
                        side_effect=RuntimeError('injected compute failure') if compute_failure else None):
                    with self.assertRaisesRegex(RuntimeError,'injected'):
                        run_auto_tail.main()
                report=json.loads((root/'automatic-report.json').read_text())
                self.assertEqual(report['status'],'INCOMPLETE')
                self.assertEqual(report['publication_status'],'FAILED')
                self.assertTrue(report['local_snapshots_retained'])
                self.assertTrue(report['pending'])
                self.assertEqual(json.loads((root/'sweep.json').read_text()),ledger)

    def test_pair_uses_only_global_deadline_not_legacy_sixty_second_limit(self):
        config=dict(n=12,r=4,world=2,timeout_seconds=60,graph_calibration=False,
            env={},resource_plan=device_budget([{'free_bytes':12<<30}]))
        with patch.object(run_auto_tail.time,'time',return_value=1000), \
             patch.object(run_auto_tail,'run',return_value=Path('manifest')) as runner:
            run_auto_tail.run_adaptive(config,Path('source'),Path('root/case'),{},deadline=8200)
        self.assertEqual(runner.call_args.args[0]['timeout_seconds'],7200)

    def test_end_upload_releases_then_resumes_pending_pairs_and_background_still_streams(self):
        for mode in ('end','background'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);events=[];calls=[]
                base=dict(resource_plan={},run_id='fixture',upload_mode=mode,cohort_group_size=1)
                (root/'automatic-config.json').write_text(json.dumps(base));(root/'runtime.json').write_text('{}')
                class Publisher:
                    def __init__(self,*args):self.closed=False;self.error=None;self.generations=0
                    def enqueue(self,ledger):events.append('enqueue')
                    def finish(self):
                        self.closed=True;self.generations+=1;events.append('uploaded_verified_released')
                        for path in root.glob('n*-m*/saved/states.bin'):path.unlink(missing_ok=True)
                        return {'receipt':'fixture'}
                    def wait_for_capacity(self,*args,**kwargs):return True
                def execute(base,source,runroot,runtime,grid,budget,runner,*,on_progress,should_stop):
                    calls.append(len(calls)+1);events.append('compute-start')
                    keys=['n2-m1'] if mode=='end' and len(calls)==1 else ['n2-m1','n3-m1']
                    ledger=dict(configuration=dict(base=base,grid=[[2,1],[3,1]]),
                        cases={key:dict(status='COMPLETE',attempted=True) for key in keys},
                        pending=[[3,1]] if len(keys)==1 else [])
                    if len(keys)==1:ledger['global_stop_reason']='SSD admission stopped compute fixture'
                    for key in keys:
                        saved=root/key/'saved';saved.mkdir(parents=True,exist_ok=True)
                        if not (saved/'manifest.json').exists():
                            (saved/'states.bin').write_bytes(bytes(8))
                            (saved/'manifest.json').write_text(json.dumps(dict(
                                packing={'bytes_per_state':8},layers=[{'states':1}],
                                files=[dict(path='states.bin',bytes=8)])))
                    on_progress(ledger);events.append('compute-finished')
                    (root/'sweep.json').write_text(json.dumps(ledger));return ledger
                args=['run_auto_tail','--source',str(root),'--runtime-env',str(root/'runtime.json'),
                      '--root',str(root),'--repo-id','fixture','--deadline-unix',str(time.time()+3600)]
                hub=SimpleNamespace(HfApi=lambda **kw:SimpleNamespace(upload_file=lambda **kw:None),get_token=lambda:'fixture')
                with patch.object(sys,'argv',args),patch.dict(sys.modules,{'huggingface_hub':hub}), \
                     patch.object(run_auto_tail,'os',SimpleNamespace(name='posix',environ=os.environ)), \
                     patch.object(run_auto_tail,'SweepPublisher',Publisher), \
                     patch.object(run_auto_tail,'execute',side_effect=execute), \
                     patch.object(run_auto_tail,'verify_hf',return_value={'verified':True}), \
                     patch.object(run_auto_tail.EndUploadDiskAdmission,'stop_reason',return_value=None), \
                     patch('builtins.print'):
                    run_auto_tail.main()
                if mode=='end':
                    self.assertEqual(len(calls),2)
                    self.assertGreater(events.index('enqueue'),events.index('compute-finished'))
                    self.assertLess(events.index('uploaded_verified_released'),events.index('compute-start',1))
                else:
                    self.assertEqual(len(calls),1)
                    self.assertLess(events.index('enqueue'),events.index('compute-finished'))
                report=json.loads((root/'automatic-report.json').read_text())
                self.assertEqual(report['upload_mode'],mode);self.assertEqual(report['pending'],[])

    def test_uses_smallest_available_gpu_and_retains_headroom(self):
        p=device_budget([{'free_bytes':12<<30},{'free_bytes':8<<30}])
        self.assertEqual(p['free_bytes'],8<<30)
        self.assertLess(p['usable_bytes'],8<<30)
        self.assertLessEqual(p['library_pool_bytes']+2048*p['max_rows_per_rank'],p['usable_bytes'])

    def test_pairs_resize_without_mutating_persisted_budget(self):
        base=dict(resource_plan=device_budget([{'free_bytes':12<<30}]),env={},batch=123)
        small=pair_config(base,4,3);large=pair_config(base,16,4)
        self.assertEqual(small['env']['MGBFS_BENCH_CAPACITY'],'256')
        self.assertEqual(int(large['env']['MGBFS_BENCH_CAPACITY']),base['resource_plan']['max_rows_per_rank'])
        self.assertEqual(base['env'],{})
        self.assertEqual(int(large['env']['MGBFS_FUTURE_CAPACITY']),2*base['resource_plan']['max_rows_per_rank'])

    def test_unavailable_hardware_rejected(self):
        for inventory in ([],[{'free_bytes':0}],[{'free_bytes':128<<20}]):
            with self.assertRaises(ValueError):device_budget(inventory)

    def test_archive_credits_fit_host_memory_and_scale_with_word_width(self):
        base=dict(world=2,host_available_bytes=8<<30,
            resource_plan=device_budget([{'free_bytes':12<<30}]),env={})
        for n,r in ((3,2),(17,10),(32,25)):
            cfg=pair_config(base,n,r);e=cfg['env']
            self.assertEqual(int(e['MGBFS_ARCHIVE_ROWS']),cfg['batch'])
            size=2*(n+16)*int(e['MGBFS_ARCHIVE_ROWS'])*int(e['MGBFS_ARCHIVE_SLOTS'])
            self.assertLessEqual(size,base['host_available_bytes']//4)
            self.assertGreaterEqual(int(e['MGBFS_ARCHIVE_SLOTS']),64)

    def test_background_keeps_inflight_and_freezes_latest_queued_ledger(self):
        started=threading.Event();release=threading.Event();seen=[]
        def publish(root,repo,api,deadline,*,ledger):
            seen.append(ledger)
            if len(seen)==1:
                started.set()
                if not release.wait(5):raise TimeoutError('test worker gate')
            return {'receipt':'ok'}
        base=dict(configuration=dict(grid=[[2,1],[2,2],[3,1]]),cases={})
        worker=SweepPublisher(None,None,None,None,publish)
        worker.enqueue(base);self.assertTrue(started.wait(5))
        base['cases']['n2-m1']=dict(attempted=True)
        worker.enqueue(base)
        base['cases']['n2-m2']=dict(attempted=True)
        worker.enqueue(base)
        base['cases']['n3-m1']=dict(attempted=True)
        release.set();worker.finish()
        self.assertEqual(len(seen),2)
        self.assertEqual(seen[0]['cases'],{})
        self.assertEqual(set(seen[1]['cases']),{'n2-m1','n2-m2'})
        self.assertEqual(seen[1]['pending'],[[3,1]])


if __name__=='__main__':unittest.main()
