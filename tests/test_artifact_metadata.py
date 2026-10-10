import unittest,subprocess,tempfile
from pathlib import Path
from unittest.mock import patch
from multigpubfs.artifact_metadata import elf_runtime_requirements
class ElfRequirements(unittest.TestCase):
 def test_numeric_maximum_across_binary_and_library(self):
  def run(command,**kwargs):
   text='GLIBC_2.9 GLIBCXX_3.4.9 CXXABI_1.3.7' if command[-1]=='binary' else 'GLIBC_2.34 GLIBCXX_3.4.32 CXXABI_1.3.9 GLIBC_PRIVATE'
   return subprocess.CompletedProcess(command,0,text,'')
  with patch('multigpubfs.artifact_metadata.subprocess.run',side_effect=run) as calls:
   self.assertEqual(elf_runtime_requirements(['binary','library']),{'GLIBC':'2.34','GLIBCXX':'3.4.32','CXXABI':'1.3.9'})
   self.assertTrue(all(x.kwargs['check'] and x.kwargs['timeout']==30 for x in calls.call_args_list))
 def test_absent_symbols_are_unknown(self):
  with patch('multigpubfs.artifact_metadata.subprocess.run',return_value=subprocess.CompletedProcess([],0,'not a version requirement','')):
   self.assertEqual(elf_runtime_requirements(['binary']),{'GLIBC':None,'GLIBCXX':None,'CXXABI':None})
 def test_readelf_failure_cannot_create_accepted_metadata(self):
  with patch('multigpubfs.artifact_metadata.subprocess.run',side_effect=subprocess.CalledProcessError(1,['readelf'])):
   with self.assertRaises(subprocess.CalledProcessError):elf_runtime_requirements(['binary'])
if __name__=='__main__':unittest.main()
