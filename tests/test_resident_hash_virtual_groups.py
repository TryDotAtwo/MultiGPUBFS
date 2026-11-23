"""CPU contract of stable (shard, virtual exact-key bucket) mapping.

This is not a GPU synchronization or throughput test.
"""
import itertools, unittest
MASK=(1<<64)-1

def group(h,word,shards,buckets):
    assert 0<buckets<=64 and buckets&(buckets-1)==0
    x=(word+0x9e3779b97f4a7c15)&MASK
    x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK
    x=((x^(x>>27))*0x94d049bb133111eb)&MASK
    x^=x>>31
    shard=(((h&0xffffffff)*shards)>>32)
    return shard*buckets+(x&(buckets-1))

class VirtualGroups(unittest.TestCase):
    def test_same_key_stable_and_forced_collisions_stay_exact(self):
        words=[sum(symbol<<(4*i) for i,symbol in enumerate(p)) for p in itertools.permutations(range(6))]
        for shards in (1,2,3,8):
            for buckets in (1,2,4,8,64):
                h=0
                mapping={word:group(h,word,shards,buckets) for word in words}
                self.assertEqual(len(mapping),720)
                self.assertTrue(all(0<=v<shards*buckets for v in mapping.values()))
                self.assertTrue(all(mapping[w]==group(h,w,shards,buckets) for w in words))
                self.assertTrue(all(v//buckets==0 for v in mapping.values()))
                if buckets>1:self.assertGreater(len(set(mapping.values())),1)
    def test_shards_follow_original_hash_partition(self):
        for shards in (1,2,3,8):
            for h in (0,1,0x7fffffff,0x80000000,0xffffffff,0xffffffffffffffff):
                for word in (0,1,0x123456789abcdef):
                    self.assertEqual(group(h,word,shards,8)//8,((h&0xffffffff)*shards)>>32)

if __name__=="__main__":unittest.main()
