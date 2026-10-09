"""Pure production replay command contract; no GPU, launcher or ingress execution."""
import ast
from pathlib import Path
import sys
import unittest

source_path = Path(sys.argv.pop(1))
tree = ast.parse(source_path.read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'instrument_rank_command']
if len(functions) != 1:
    raise AssertionError('INSTRUMENT_COMMAND_MISSING')
namespace = {}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_path), 'exec'), namespace)
instrument = namespace['instrument_rank_command']

class ScopedProfileContract(unittest.TestCase):
    def test_search_capture_has_explicit_profiler_boundary(self):
        cmd = instrument('/tmp/mgbfs', ['bench'], 'nsys-search', '/tmp/rank-0')
        self.assertEqual(cmd[:2], ['nsys', 'profile'])
        self.assertIn('--capture-range=cudaProfilerApi', cmd)
        self.assertIn('--capture-range-end=stop', cmd)
        self.assertIn('--trace=cuda,nvtx,osrt', cmd)
        self.assertEqual(cmd[-2:], ['/tmp/mgbfs', 'bench'])
    def test_legacy_full_process_capture_stays_explicit(self):
        cmd = instrument('/tmp/mgbfs', ['bench'], 'nsys', '/tmp/rank-0')
        self.assertNotIn('--capture-range=cudaProfilerApi', cmd)
        self.assertEqual(cmd[-2:], ['/tmp/mgbfs', 'bench'])
    def test_sanitizers_and_uninstrumented_target_unchanged(self):
        self.assertEqual(instrument('/tmp/mgbfs', ['bench'], None, '/tmp/rank-0'), ['/tmp/mgbfs', 'bench'])
        for tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck'):
            cmd = instrument('/tmp/mgbfs', ['bench'], tool, '/tmp/rank-0', '/tmp/compute-sanitizer')
            self.assertEqual(cmd[:5], ['/tmp/compute-sanitizer', '--tool', tool, '--error-exitcode', '97'])
            self.assertEqual(cmd[-2:], ['/tmp/mgbfs', 'bench'])
    def test_unknown_instrumentation_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'UNKNOWN_PROCESS_INSTRUMENTATION'):
            instrument('/tmp/mgbfs', [], 'silent-fallback', '/tmp/rank-0')

if __name__ == '__main__':
    unittest.main()
