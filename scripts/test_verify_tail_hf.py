import hashlib
import unittest

from verify_tail_hf import verify_payload


class Response:
    def __init__(self, chunks):
        self.chunks = chunks

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        return iter(self.chunks)


class VerifyTailTests(unittest.TestCase):
    def setUp(self):
        self.entry = dict(bytes=3, sha256=hashlib.sha256(b'abc').hexdigest())

    def test_chunk_boundaries_do_not_change_checksum(self):
        self.assertEqual(verify_payload(Response([b'a', b'', b'bc']), self.entry), 3)

    def test_truncated_payload_rejected(self):
        with self.assertRaises(ValueError):
            verify_payload(Response([b'ab']), self.entry)

    def test_same_size_corruption_rejected(self):
        with self.assertRaises(ValueError):
            verify_payload(Response([b'abd']), self.entry)

    def test_oversize_stops_stream_immediately(self):
        class OversizeResponse(Response):
            def iter_content(self, chunk_size):
                yield b'abcd'
                raise AssertionError('must not continue after exceeding size')
        with self.assertRaises(ValueError):
            verify_payload(OversizeResponse([]), self.entry)


if __name__ == '__main__':
    unittest.main()
