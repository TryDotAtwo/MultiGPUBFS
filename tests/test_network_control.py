import unittest,socket
from concurrent.futures import ThreadPoolExecutor
from threading import Event
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

 def test_first_error_is_preserved_until_peer_acknowledges(self):
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  server=ControlStore('127.0.0.1',port,0,'failure',2,'secret')
  client=ControlStore('127.0.0.1',port,1,'failure',2,'secret')
  try:
   with ThreadPoolExecutor(1) as pool:
    publishing=pool.submit(server.report_error,RuntimeError('coordinator failed'),2)
    self.assertEqual(client.get('error'),'coordinator failed')
    self.assertFalse(publishing.done())
    with self.assertRaisesRegex(RuntimeError,'DISTRIBUTED_PEER_ERROR: coordinator failed'):
     client.get('pending-job')
    client.report_error(RuntimeError('peer saw the failure'),2)
    publishing.result(timeout=2)
   self.assertEqual(client.get('error'),'coordinator failed')
  finally:server.close()

 def test_delayed_duplicate_cannot_restore_a_retired_job_reply(self):
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  server=ControlStore('127.0.0.1',port,0,'retirement',2)
  client=ControlStore('127.0.0.1',port,1,'retirement',2)
  delayed, resume=Event(),Event()
  handler=server.server.RequestHandlerClass
  block_request=handler.block_request
  def delay_first_send(peer):
   if peer.path=='/put-block' and not delayed.is_set():
    delayed.set()
    resume.wait(2)
   block_request(peer)
  handler.block_request=delay_first_send
  reply='job/0/window/files/1/reply'
  command='job/0/window/files/1/request/0'
  try:
   with ThreadPoolExecutor(1) as pool:
    original=pool.submit(client.put_block,reply,0,b'block')
    try:
     self.assertTrue(delayed.wait(1))
     client.put_block(reply,0,b'block')
     self.assertEqual(server.get_block(reply,0),b'block')
     server.acknowledge_block(reply,0,command_key=command)
     server.release_prefix('job/0/')
    finally:
     resume.set()
    original.result(timeout=2)
   client.put(command,['edges',0,10])
   self.assertIsNone(server.try_get(command))
   with self.assertRaisesRegex(RuntimeError,'409'):server.get_block(reply,0)
  finally:server.close()
if __name__=='__main__':unittest.main()
