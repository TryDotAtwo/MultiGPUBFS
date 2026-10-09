import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
from multigpubfs import GraphDefinition
from multigpubfs.cli import main
class CliContract(unittest.TestCase):
 def test_json_and_explicit_devices_preserve_contract(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'g.json';g=GraphDefinition.permutation([[1,0]],[-9,1<<40]);p.write_text(g.to_json())
   with patch('multigpubfs.cli.run_graph',return_value={'status':'COMPLETE'}) as run:
    self.assertEqual(main([str(p),str(Path(d)/'out'),'--devices','1,0','--seconds','12','--backend','generic','--no-autotune']),0)
    self.assertEqual(run.call_args.args[0].digest(),g.digest());self.assertEqual(run.call_args.kwargs['devices'],[1,0]);self.assertFalse(run.call_args.kwargs['autotune'])
 def test_bad_definition_does_not_start_gpu(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'bad.json';p.write_text('{}')
   with patch('multigpubfs.cli.run_graph') as run:self.assertEqual(main([str(p),str(Path(d)/'out')]),1);run.assert_not_called()
if __name__=='__main__':unittest.main()
