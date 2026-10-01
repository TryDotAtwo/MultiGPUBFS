import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tail_remote_suite import trace_layers


class TraceTests(unittest.TestCase):
    def test_requires_both_ranks_and_completed_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'native.log'
            path.write_text('''MGBFS_DEPTH_BEGIN rank=0 depth=0 count=1 unix=1
MGBFS_DEPTH_BEGIN rank=1 depth=0 count=0 unix=1
MGBFS_DEPTH_END rank=1 depth=0 seconds=0.3 next=2 alive=true unix=2
MGBFS_DEPTH_END rank=0 depth=0 seconds=0.2 next=1 alive=true unix=2
MGBFS_DEPTH_BEGIN rank=0 depth=1 count=1 unix=2
MGBFS_DEPTH_BEGIN rank=1 depth=1 count=2 unix=2
MGBFS_DEPTH_END rank=0 depth=1 seconds=0.4 next=3 alive=true unix=3
''')
            self.assertEqual(trace_layers(path),[dict(depth=0,states=1,seconds=.3)])


if __name__=='__main__':unittest.main()
