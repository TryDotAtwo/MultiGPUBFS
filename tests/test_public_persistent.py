import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import multigpubfs

class PublicPersistentContext(unittest.TestCase):
 def test_public_context_manager_is_available_without_gpu_initialization(self):
  self.assertIn('persistent_graph_workers',multigpubfs.__all__)
  def forbidden(*a,**kw):raise AssertionError('Empty persistent context initialized GPU or network')
  with patch('subprocess.Popen',forbidden),patch('urllib.request.urlopen',forbidden):
   with multigpubfs.persistent_graph_workers() as pool:
    directory=pool.root
    self.assertTrue(directory.is_dir())
   self.assertFalse(directory.exists())
