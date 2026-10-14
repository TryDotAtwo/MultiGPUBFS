import unittest
from pathlib import Path
class StartupTrace(unittest.TestCase):
 def test_trace_disabled_without_diagnostic_flag(self):
  source=Path('crates/mgbfs-runtime/src/job_phase.rs').read_text()
  self.assertIn('if let Some(path)=std::env::var_os("MGBFS_DIAGNOSTIC_TRACE_FILE")',source)
 def test_inventory_checkpoints_precede_calls(self):
  source=Path('crates/mgbfs-runtime/src/generic_session.rs').read_text().split('Some("inventory")=>{')[1].split('Some("graph-local-plan")')[0]
  for phase,call in [('inventory-cudaSetDevice','cudaSetDevice('),('inventory-cudaMemGetInfo','cudaMemGetInfo('),('inventory-hardware-info','mgbfs_generic_hardware_info(')]:
   self.assertLess(source.index(phase),source.index(call))
 def test_thread_probe(self):
  import tempfile
  from multigpubfs.diagnostics import _component
  with tempfile.TemporaryDirectory() as folder:
   self.assertTrue(_component('thread',Path(folder),None,None,5)['thread_start_join_verified'])
