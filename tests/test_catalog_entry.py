import contextlib,io,json,sys,types,unittest
from unittest.mock import patch
from multigpubfs.graph_definition import GraphDefinition
class CatalogEntry(unittest.TestCase):
 def fake(self):
  from multigpubfs.graph_definition import from_cayleypy
  class Constructor:
   @staticmethod
   def lrx(n=4):
    return types.SimpleNamespace(name='fake',generator_names=['L','R','X'],generators_type=types.SimpleNamespace(name='PERMUTATION'),central_state=list(range(n)),generators_permutations=[list(range(1,n))+[0],[n-1]+list(range(n-1)),[1,0]+list(range(2,n))])
  return types.SimpleNamespace(PermutationGroups=Constructor,MatrixGroups=Constructor,Puzzles=Constructor)
 def test_exact_named_constructor_and_parameters(self):
  from multigpubfs import from_catalog
  with patch.dict(sys.modules,cayleypy=self.fake()):
   g=from_catalog('PermutationGroups.lrx',kwargs={'n':5})
   self.assertEqual(g.start,list(range(5)))
   self.assertEqual(g.action['generators'][0],[1,2,3,4,0])
   self.assertEqual(from_catalog('PermutationGroups.lrx',args=[5]).digest(),g.digest())
 def test_private_arbitrary_and_noncallable_paths_rejected_before_call(self):
  from multigpubfs import from_catalog
  for name in ['os.system','PermutationGroups.__class__','PermutationGroups.lrx.__call__','__import__("os")','PermutationGroups.missing']:
   with patch.dict(sys.modules,cayleypy=self.fake()),self.assertRaises(ValueError):from_catalog(name)
 def test_parameter_shapes_fail_before_constructor(self):
  from multigpubfs import from_catalog
  for args,kwargs in [('5',{}),([],[]),([],{'__dict__':1})]:
   with patch.dict(sys.modules,cayleypy=self.fake()),self.assertRaises(ValueError):from_catalog('PermutationGroups.lrx',args=args,kwargs=kwargs)
 def test_public_cli_catalog_uses_same_definition(self):
  from multigpubfs.cli import main
  with patch.dict(sys.modules,cayleypy=self.fake()),patch('multigpubfs.cli.run_graph',return_value={'status':'COMPLETE'}) as run,contextlib.redirect_stdout(io.StringIO()):
   self.assertEqual(main(['PermutationGroups.lrx','/unused','--catalog','--catalog-kwargs','{"n":5}']),0)
   self.assertEqual(run.call_args.args[0].start,list(range(5)))
 def test_cli_invalid_json_no_gpu_launch(self):
  from multigpubfs.cli import main
  with patch('multigpubfs.cli.run_graph') as run,contextlib.redirect_stderr(io.StringIO()):
   self.assertEqual(main(['PermutationGroups.lrx','/unused','--catalog','--catalog-kwargs','not-json']),1);run.assert_not_called()
 def test_cli_catalog_parameters_not_silently_ignored_for_json_file(self):
  from multigpubfs.cli import main
  with patch('multigpubfs.cli.run_graph') as run,contextlib.redirect_stderr(io.StringIO()):
   self.assertEqual(main(['/unused.json','/unused','--catalog-kwargs','{"n":5}']),1);run.assert_not_called()
if __name__=='__main__':unittest.main()
