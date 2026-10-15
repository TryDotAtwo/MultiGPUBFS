import unittest
from multigpubfs.diagnostic_analysis import analyze
class ReadyInstructions(unittest.TestCase):
 def test_verified_calculation_is_not_told_to_resend_logs(self):
  r={'steps':[{'name':'cuda-discovery','status':'TIMEOUT'}],'working_configuration':{'status':'READY','selected':{'devices':[0,1]},'command':'verified-check','calculation_example':'verified-calculation'}}
  actions=analyze(r)['user_next_actions'];self.assertTrue(any('verified-calculation' in x for x in actions))
  self.assertFalse(any('Return report.json' in x for x in actions))
