import io
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from local_tail_upload_relay import validate_header,read_exact


class RelayTests(unittest.TestCase):
    def test_rejects_other_repo_prefix_traversal_and_oversize(self):
        header=dict(repo_id='owner/data',repo_type='dataset',path_in_repo='tail-runs/run/manifest.json',
                    bytes=8,sha256='0'*64)
        validate_header(header,'owner/data','run',100)
        for changes in [dict(repo_id='other/data'),dict(path_in_repo='tail-runs/other/file'),
                        dict(path_in_repo='tail-runs/run/../../file'),dict(bytes=101),
                        dict(bytes=True),dict(sha256=42)]:
            with self.assertRaises(ValueError):validate_header(dict(header,**changes),'owner/data','run',100)

    def test_truncated_payload_cannot_be_uploaded(self):
        self.assertEqual(read_exact(io.BytesIO(b'12345678'),8),b'12345678')
        with self.assertRaises(EOFError):read_exact(io.BytesIO(b'123'),8)


if __name__=='__main__':unittest.main()
