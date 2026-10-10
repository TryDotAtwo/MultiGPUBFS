import unittest,socket
from multigpubfs.network_control import ControlStore
class NetworkControlTests(unittest.TestCase):
 def test_identity_token_immutable_keys_and_timeout(self):
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  server=ControlStore('127.0.0.1',port,0,'session',.2,'secret')
  try:
   client=ControlStore('127.0.0.1',port,1,'session',.2,'secret');client.put('key',{'x':1});self.assertEqual(server.get('key'),{'x':1});client.put('key',{'x':1})
   with self.assertRaisesRegex(RuntimeError,'409'):client.put('key',{'x':2})
   client.put('pilot/receipt',{'states':[[1]]});server.release_prefix('pilot/')
   with self.assertRaisesRegex(RuntimeError,'TIMEOUT'):client.get('pilot/receipt',timeout=.01)
   self.assertEqual(server.get('key'),{'x':1})
   with self.assertRaisesRegex(RuntimeError,'RANK_ZERO'):client.release_prefix('key')
   with self.assertRaisesRegex(RuntimeError,'403'):ControlStore('127.0.0.1',port,1,'session',.2,'wrong').put('key',0)
   with self.assertRaisesRegex(RuntimeError,'409'):ControlStore('127.0.0.1',port,1,'wrong',.2,'secret').put('key',0)
   with self.assertRaisesRegex(RuntimeError,'TIMEOUT_NO_RESTART'):client.get('missing',timeout=.01)
   client.put('error','HOST_MEMORY_ADMISSION_FAILED fixture')
   with self.assertRaisesRegex(RuntimeError,'DISTRIBUTED_PEER_ERROR'):server.get('peer-wait',timeout=.01)
  finally:server.close()
if __name__=='__main__':unittest.main()
