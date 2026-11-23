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

 def test_batch_scheduler_is_explicit_and_sizes_are_validated(self):
  options=dict(execution_engine="device",autotune=False,memory_backend="permutation_bitmap",device_scheduler="batch",device_batch_edges=33,device_receive_records=7)
  self.assertEqual(NativeOptions(**options).device_scheduler,"batch")
  self.assertEqual(NativeOptions(**dict(options,memory_backend="segmented24")).device_scheduler,"batch")
  for override in (dict(execution_engine="host"),dict(memory_backend="legacy"),
                   dict(device_scheduler="unknown"),dict(device_batch_edges=0),
                   dict(device_receive_records=True),dict(device_receive_records=1<<64)):
   with self.subTest(override=override),self.assertRaises(ValueError):
    NativeOptions(**dict(options,**override))
  with self.assertRaisesRegex(ValueError,"DEVICE_BATCH_SIZE_REQUIRES_BATCH"):
   NativeOptions(device_batch_edges=1)

 def test_launch_rejects_unsupported_options_before_runtime(self):
  graph=GraphDefinition.permutation([[1,0]],[0,1])
  for kw in (dict(execution_engine="bad"),dict(execution_engine="device"),
             dict(execution_engine="device",memory_backend="permutation_bitmap",autotune=True)):
   with self.subTest(kw=kw),patch("multigpubfs.launch.native_runtime") as native:
    with self.assertRaises(ValueError):
     run_graph(graph,Path(tempfile.gettempdir())/"unused-resident-test",**kw)
    native.assert_not_called()

if __name__=="__main__":unittest.main()
