"""Synthetic evidence fixtures: never hardware results."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('native_verifier',Path(__file__).parents[1]/'scripts/verify_native_runtime.py')
MODULE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

class NativeEvidence(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        self.source='a'*40
        self.summary=dict(source_commit=self.source,gpus=[dict(index=i,uuid=f'fixture-{i}',name='Tesla T4') for i in range(2)],tests=[])
        for gpu in self.summary['gpus']:
            for tool in MODULE.TOOLS:
                self.summary['tests'].append(dict(gpu=gpu['uuid'],tool=tool,status='PASS'))
                log='\n'.join('test '+name+' ... ok' for name in ('native_archive_roundtrip','native_feedback_full_layers','layer_capacity_failure_is_terminal'))
                log+='\ntest result: ok. 3 passed; 0 failed;\n'
                if tool=='plain':
                    log+='\n'.join(f'FULL_STATE_PASS m={m} pre={p}' for m in range(5,9) for p in ('false','true'))
                elif tool=='racecheck': log+='RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n'
                else: log+='ERROR SUMMARY: 0 errors\n'
                (self.path/f"gpu{gpu['index']}-{tool}.log").write_text(log)
        self.save()
    def save(self): (self.path/'summary.json').write_text(json.dumps(self.summary))
    def test_complete_matrix(self): MODULE.verify_gate(self.path,self.source)
    def test_source_binding(self):
        with self.assertRaises(ValueError): MODULE.verify_gate(self.path,'b'*40)
    def test_duplicate_gpu(self):
        self.summary['gpus'][1]['uuid']='fixture-0'; self.save()
        with self.assertRaises(ValueError): MODULE.verify_gate(self.path,self.source)
    def test_missing_run(self):
        self.summary['tests'].pop(); self.save()
        with self.assertRaises(ValueError): MODULE.verify_gate(self.path,self.source)
    def test_warning_is_failure(self):
        path=self.path/'gpu1-racecheck.log'
        path.write_text(path.read_text().replace('0 warnings','1 warnings'))
        with self.assertRaises(ValueError): MODULE.verify_gate(self.path,self.source)
    def test_nonzero_error_even_with_final_zero(self):
        path=self.path/'gpu0-memcheck.log'
        path.write_text(path.read_text()+'ERROR SUMMARY: 1 errors\nERROR SUMMARY: 0 errors\n')
        with self.assertRaises(ValueError): MODULE.verify_gate(self.path,self.source)
    def test_missing_large_case(self):
        path=self.path/'gpu0-plain.log'
        path.write_text(path.read_text().replace('FULL_STATE_PASS m=8 pre=true',''))
        with self.assertRaises(ValueError): MODULE.verify_gate(self.path,self.source)
    def test_test_failure_is_not_gate_completion(self):
        path=self.path/'gpu0-plain.log'
        path.write_text(path.read_text().replace('0 failed','1 failed'))
        with self.assertRaises(ValueError): MODULE.verify_gate(self.path,self.source)

if __name__=='__main__': unittest.main()
