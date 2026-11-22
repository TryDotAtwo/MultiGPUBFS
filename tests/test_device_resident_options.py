"""CPU checks for the opt-in resident path and its explicit limitations."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multigpubfs import GraphDefinition,NativeOptions,run_graph

class DeviceResidentOptions(unittest.TestCase):
 def test_default_stays_host(self):
  self.assertEqual(NativeOptions().execution_engine,"host")

 def test_device_requires_supported_shape(self):
  options=dict(execution_engine="device",autotune=False,memory_backend="permutation_bitmap")
  self.assertEqual(NativeOptions(**options).execution_engine,"device")
  self.assertEqual(NativeOptions(**dict(options,memory_backend="segmented24")).execution_engine,"device")
  for override in (dict(memory_backend="legacy"),dict(segment_strategy="warp"),
                   dict(memory_preset="max_capacity"),dict(autotune=True),dict(max_seconds=1)):
   with self.subTest(override=override),self.assertRaises(ValueError):
    NativeOptions(**dict(options,**override))

 def test_launch_rejects_unsupported_options_before_runtime(self):
  graph=GraphDefinition.permutation([[1,0]],[0,1])
  for kw in (dict(execution_engine="bad"),dict(execution_engine="device"),
             dict(execution_engine="device",memory_backend="permutation_bitmap",autotune=True)):
   with self.subTest(kw=kw),patch("multigpubfs.launch.native_runtime") as native:
    with self.assertRaises(ValueError):
     run_graph(graph,Path(tempfile.gettempdir())/"unused-resident-test",**kw)
    native.assert_not_called()

if __name__=="__main__":unittest.main()
