import copy,json,tempfile,unittest
from pathlib import Path
from scripts.paired_tail import SEEDS,compare_tables,run_pair,publication_records
from scripts.tail_cohort import case_groups,seal_current_cohort
from scripts.publish_tail_batch import plan
from test_tail_cohort import CohortTests

class PairedTests(unittest.TestCase):
    def test_resource_exhaustion_skips_second_run_without_claiming_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); case=root/'n7-m2'; calls=[]
            def runner(cfg, source, path, runtime):
                calls.append(cfg['repetition'])
                (path/'saved').mkdir(parents=True)
                manifest=dict(status='INCOMPLETE', last_completed_layer=0,
                    stop_reason='cuda out of memory', layers=[dict(depth=0, states=1)])
                target=path/'saved/manifest.json'
                target.write_text(json.dumps(manifest)); return target
            result=run_pair(dict(n=7,r=2,world=1,run_id='test'),root,case,{},runner,None)
            manifest=json.loads(result.read_text())
            self.assertEqual(calls,[1])
            self.assertIsNone(manifest['comparison']['matched'])
            self.assertTrue(manifest['comparison']['second_run_skipped'])
            self.assertFalse((root/'n7-m2-rep2').exists())
    def test_comparison_excludes_timing_but_detects_counts_status_and_missing(self):
        a=dict(status='COMPLETE',last_completed_layer=0,layers=[dict(depth=0,states=2,seconds=1)])
        b=copy.deepcopy(a);b['layers'][0]['seconds']=9
        self.assertTrue(compare_tables(a,b,True)['matched'])
        b['layers'][0]['states']=3
        self.assertFalse(compare_tables(a,b,True)['matched'])
        b=copy.deepcopy(a);b['status']='INCOMPLETE'
        self.assertFalse(compare_tables(a,b,True)['matched'])
        self.assertFalse(compare_tables(a,a,False)['matched'])
        self.assertFalse(compare_tables({}, {},True)['matched'])
    def test_two_real_native_seed_reports_and_mismatch_still_get_two_publications(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);case=root/'n7-m2';fixture=CohortTests();calls=[]
            def runner(cfg,source,path,runtime):
                calls.append(cfg['env']['MGBFS_HASH_SEED_HEX'])
                key,saved=fixture.fixture(root,7,2)
                if path!=root/key:
                    import shutil
                    shutil.copytree(root/key,path)
                m=json.loads((path/'saved/manifest.json').read_text());m.update(last_completed_layer=5,stop_reason='capacity')
                if len(calls)==2:m['layers'][1]['states']=7
                (path/'saved/manifest.json').write_text(json.dumps(m))
                (path/'result').mkdir(exist_ok=True)
                (path/'result/rank-0.json').write_text(json.dumps(dict(hash_seed_hex=calls[-1],search_prefix_seconds=1)))
                return path/'saved/manifest.json'
            # fixture must not overwrite an existing first directory on second call
            count=0
            def one(cfg,source,path,runtime):
                nonlocal count
                if count:
                    import shutil
                    shutil.copytree(case,path)
                    calls.append(cfg['env']['MGBFS_HASH_SEED_HEX'])
                    m=json.loads((path/'saved/manifest.json').read_text());m['layers'][1]['states']=7
                    (path/'saved/manifest.json').write_text(json.dumps(m))
                    (path/'result/rank-0.json').write_text(json.dumps(dict(hash_seed_hex=calls[-1],search_prefix_seconds=1)))
                    return path/'saved/manifest.json'
                count+=1;return runner(cfg,source,path,runtime)
            result=run_pair(dict(n=7,r=2,world=1,run_id='test-n7-m2'),root,case,{},one,None)
            m=json.loads(result.read_text());self.assertEqual(calls,SEEDS);self.assertFalse(m['comparison']['matched'])
            ledger=fixture.ledger([case.name]);ledger['configuration']['base']['two_seeds']=True;ledger['cases'][case.name]['replicas']=m['replicas']
            self.assertEqual(len(publication_records(ledger)),2)
            seal_current_cohort(root,ledger)
            self.assertEqual(set(ledger['cohort_seal_after']),{case.name,case.name+'-rep2'})
            self.assertEqual(len(list(case_groups(root,ledger))),2)
            payloads,manifests,_=plan(root,ledger=ledger)
            self.assertEqual(len(payloads),2);self.assertEqual(len(manifests),2)
            self.assertTrue(all('comparison' in json.loads(path.read_text()) for path,_ in manifests))

if __name__=='__main__':unittest.main()

