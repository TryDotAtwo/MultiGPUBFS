import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from vast_watchdog import step
except ImportError:
    step = None


class WatchdogTests(unittest.TestCase):
    def test_deadline_requires_matching_identity_and_later_absence(self):
        self.assertIsNotNone(step)
        target = dict(id=123, machine_id=456, label='mgbfs-lrx13-fixture')
        calls = []
        state = dict(target)
        def api(method, instance):
            self.assertEqual(instance, 123)
            calls.append(method)
            return {'instances': state} if method == 'GET' else {'success': True}
        self.assertEqual(step(target, 99, 100, api), 'WAITING')
        self.assertEqual(calls, ['GET'])
        self.assertEqual(step(target, 100, 100, api), 'DELETE_REQUESTED')
        self.assertEqual(calls[-2:], ['GET', 'DELETE'])
        state = None
        self.assertEqual(step(target, 101, 100, api), 'ABSENT')

    def test_mismatch_or_api_failure_never_deletes_another_instance(self):
        self.assertIsNotNone(step)
        target = dict(id=123, machine_id=456, label='mgbfs-lrx13-fixture')
        for wrong in [dict(target, id=124), dict(target, machine_id=457),
                      dict(target, label='another-project'), {}, {'success':False}]:
            calls = []
            def api(method, instance):
                calls.append(method)
                return {'instances':wrong}
            with self.assertRaises(ValueError):
                step(target, 101, 100, api)
            self.assertEqual(calls, ['GET'])

    def test_failed_delete_is_not_success(self):
        self.assertIsNotNone(step)
        target = dict(id=123, machine_id=456, label='mgbfs-lrx13-fixture')
        def api(method, instance):
            return {'instances':target} if method == 'GET' else {'success':False}
        with self.assertRaises(RuntimeError):
            step(target, 101, 100, api)


if __name__ == '__main__':
    unittest.main()
