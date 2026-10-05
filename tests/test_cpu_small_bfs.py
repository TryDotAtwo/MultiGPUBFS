import json
import tempfile
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from scripts.cpu_small_bfs import exact_search, eligible, run, transitions
from scripts.paired_tail import SEEDS, run_pair
from scripts.state_layout import state_layout


def oracle(n, r):
    start = tuple(range(n-r+1))+(n-r,)*(r-1)
    visited = {start}; current = {start}; layers = []
    while current:
        layers.append(current)
        following = set()
        for word in current:
            for child in (word[1:]+word[:1],word[-1:]+word[:-1],
                          (word[1],word[0])+word[2:]):
                if child not in visited:
                    visited.add(child); following.add(child)
        current = following
    return layers


class SmallCpuTests(unittest.TestCase):
    def test_transitions_across_uint64_and_eight_bit_symbols(self):
        for n in [17,33,64,128]:
            bits=state_layout(n,n)['bits_per_symbol']
            state=tuple(range(n)); packed=sum(x<<(i*bits) for i,x in enumerate(state))
            expected=[state[1:]+state[:1],state[-1:]+state[:-1],(state[1],state[0])+state[2:]]
            actual=[tuple((w>>(i*bits))&((1<<bits)-1) for i in range(n)) for w in transitions(packed,n,bits)]
            self.assertEqual(actual,expected)

    def test_exact_states_both_seeds_and_wide_layouts(self):
        for n,r in [(2,1),(4,1),(8,1),(17,16),(33,32),(128,127),(33,31)]:
            expected = oracle(n,r)
            bits = state_layout(n,n-r+1)['bits_per_symbol']
            for seed in SEEDS:
                actual = []
                result = exact_search(n,r,seed,lambda d,ws,*a: actual.append(
                    {tuple((w>>(i*bits))&((1<<bits)-1) for i in range(n)) for w in ws}))
                self.assertTrue(result['complete'])
                self.assertEqual(actual,expected)

    def test_cancellation_does_not_commit_partial_layer(self):
        depths=[]
        result=exact_search(8,1,SEEDS[0],lambda d,*a: depths.append(d),
                            cancelled=lambda: 'stop' if depths else None)
        self.assertFalse(result['complete']); self.assertEqual(depths,[0])

    def test_threshold_is_backend_routing(self):
        self.assertTrue(eligible(dict(n=128,r=127,small_graph_cpu_max_states=65536)))
        self.assertFalse(eligible(dict(n=17,r=1,small_graph_cpu_max_states=65536)))
        self.assertFalse(eligible(dict(n=4,r=1)))

    def test_complete_export_and_incomplete_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg=dict(n=8,r=1,world=2,run_id='cpu-test',small_graph_cpu_max_states=65536,
                     retention_policy='last_complete_small_1000',env={'MGBFS_HASH_SEED_HEX':SEEDS[0]})
            path=run(cfg,Path(__file__).parents[1],Path(temporary)/'complete',{},program_commit='test')
            manifest=json.loads(path.read_text())
            self.assertEqual(manifest['status'],'COMPLETE')
            self.assertEqual(manifest['layers'][-1]['states'],1)
            self.assertEqual(manifest['execution_backend'],'CPU_EXACT_PACKED')
            calls=[0]
            def cancel():
                calls[0]+=1
                return 'stop' if calls[0]>=20 else None
            path=run(cfg,Path(__file__).parents[1],Path(temporary)/'partial',{},
                     program_commit='test',cancelled=cancel)
            manifest=json.loads(path.read_text())
            self.assertEqual(manifest['status'],'INCOMPLETE')
            self.assertGreater(sum(f['states'] for f in manifest['files']),0)
            self.assertLessEqual(sum(f['states'] for f in manifest['files']),1000)
            expected=oracle(8,1)
            for f in manifest['files']:
                data=(Path(temporary)/'partial/saved'/f['path']).read_bytes()
                width=manifest['packing']['bytes_per_state'];bits=manifest['packing']['bits_per_symbol']
                words=[int.from_bytes(data[i:i+width],'little') for i in range(0,len(data),width)]
                states={tuple((word>>(i*bits))&((1<<bits)-1) for i in range(8)) for word in words}
                self.assertEqual(len(states),f['states'])
                self.assertLessEqual(states,expected[f['depth']])

    def test_two_seed_evidence_is_cpu_and_not_fabricated_rank_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg=dict(n=4,r=1,world=2,run_id='paired',small_graph_cpu_max_states=65536,
                     retention_policy='last_complete_small_1000')
            path=run_pair(cfg,Path(__file__).parents[1],Path(temporary)/'pair',{},
                lambda c,s,p,e:run(c,s,p,e,program_commit='test'),
                lambda *a:self.fail('unexpected failure'))
            comparison=json.loads(path.read_text())['comparison']
            self.assertTrue(comparison['complete_graph_verified'])
            self.assertTrue(all(x['seed_verified'] for x in comparison['runs']))
            self.assertTrue(all(x['native_seed_verified'] is None for x in comparison['runs']))
            self.assertTrue(all(x['production_search_seconds']>=0 for x in comparison['runs']))


if __name__=='__main__': unittest.main()
