import json
import sys
import unittest
import tempfile
import subprocess
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import remote_build
    from remote_build import test_executable
except ImportError:
    test_executable = None


class RemoteBuildTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux', 'Linux build preflight')
    def test_wrong_source_pin_fails_before_any_download_and_saves_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'source'; source.mkdir()
            subprocess.run(['git', 'init', '-q', str(source)], check=True)
            subprocess.run(['git', '-C', str(source), '-c', 'user.name=fixture',
                            '-c', 'user.email=fixture@example.invalid', 'commit',
                            '-q', '--allow-empty', '-m', 'fixture'], check=True)
            argv = ['build', '--source', str(source), '--commit', '0'*40,
                    '--work', str(root/'build'), '--timeout-seconds', '30']
            # Downloads/compilers must never execute in this wrong-pin fixture.
            with patch.object(sys, 'argv', argv), patch.object(remote_build.shutil, 'which', return_value='/fixture/tool'), patch.object(remote_build.shutil, 'disk_usage',
                    return_value=__import__('collections').namedtuple('Disk', 'total used free')(50<<30, 0, 50<<30)):
                with self.assertRaisesRegex(ValueError, 'SOURCE_COMMIT_MISMATCH'):
                    remote_build.main()
            report = json.loads((root/'build/build-summary.json').read_text())
            self.assertEqual(report['status'], 'INCOMPLETE')
            self.assertEqual(list((root/'build/logs').iterdir()), [root/'build/logs/source-sha.log'])

    def test_only_matching_executable_from_successful_cargo_build(self):
        self.assertIsNotNone(test_executable)
        artifact = dict(reason='compiler-artifact', target=dict(name='library_multi_gpu', kind=['test']),
                        executable='/tmp/build/library_multi_gpu-abcd')
        done = dict(reason='build-finished', success=True)
        self.assertEqual(test_executable('\n'.join(map(json.dumps, [artifact, done]))), artifact['executable'])
        for records in [[artifact], [done], [artifact, dict(reason='build-finished', success=False)],
                        [dict(artifact, executable=None), done],
                        [dict(artifact, target=dict(name='other', kind=['test'])), done],
                        [artifact, dict(artifact, executable='/tmp/other'), done]]:
            with self.assertRaises(ValueError):
                test_executable('\n'.join(map(json.dumps, records)))


if __name__ == '__main__':
    unittest.main()
