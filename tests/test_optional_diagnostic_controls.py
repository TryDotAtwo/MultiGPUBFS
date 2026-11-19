import copy, json
from unittest.mock import patch
import test_native_assets_diagnostics as fixture

class OptionalBuildControls(fixture.Assets):
 def slim(self):
  idx=copy.deepcopy(self.index)
  ids=list(idx['diagnostics'])
  idx['diagnostics']={}
  idx['retired_diagnostic_controls']=[{'id':name,'reason':'not rebuilt from current CUDA source'} for name in ids]
  return idx
 def test_retired_controls_do_not_invalidate_production_inventory(self):
  from multigpubfs import runtime_variants as v
  idx=self.slim();root=self.manager(idx).prepare(fixture.hardware(100))
  value=v.inventory(root,environ={})
  self.assertEqual(value['status'],'AVAILABLE')
  self.assertEqual({x['id'] for x in value['variants']},{'shared-sm120','shared-universal'})
  self.assertEqual(set(value['unavailable_variants']),{'static-sm120','shared-legacy','baseline-static-legacy'})
  with self.assertRaisesRegex(RuntimeError,'DIAGNOSTIC_VARIANT_UNAVAILABLE'):
   v.environment('static-sm120',value,parent={})
 def test_undeclared_missing_control_is_still_rejected(self):
  from multigpubfs import runtime_variants as v
  idx=self.slim();idx['retired_diagnostic_controls'].pop()
  root=self.manager(idx).prepare(fixture.hardware(100))
  with self.assertRaisesRegex(RuntimeError,'DIAGNOSTIC_VARIANT_MATRIX'):v.inventory(root,environ={})
