"""A diagnostic classifier must never turn unknown or truncated errors into PASS."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import audit_nccl_sanitizer as audit
except ImportError:
    audit = None


PROBE = '''========= Program hit cudaErrorNoKernelImageForDevice (error 209) due to "no kernel image is available for execution on the device" on CUDA API call to cudaFuncGetAttributes.
=========     Saved host backtrace up to driver entry point at error
=========         Host Frame: ncclInitKernelsForDevice(int, int, unsigned long*) in enqueue.cc:52 [0x74f1b] in libnccl.so.2
========= 
'''
PEER = '''========= Program hit p2pMap(ncclComm*, ncclProxyConnector*, ncclPeerInfo*, ncclPeerInfo*, ncclP2pBuff*, void**, void**) [clone .isra.0] (error 704) due to "transport/p2p.cc" on CUDA API call to cudaGetLastError.
=========     Saved host backtrace up to driver entry point at error
=========         Host Frame: p2pRecvConnect(ncclComm*, ncclConnect*, int, int, ncclConnector*) in ncclTransportP2pSetup(ncclComm*, ncclTopoGraph*, int):331 [0x134a34] in libnccl.so.2
========= 
'''


class NcclAuditTests(unittest.TestCase):
    def classify(self, text):
        self.assertIsNotNone(audit, 'diagnostic classifier is not implemented')
        return audit.classify(text)

    def test_counts_verified_probes_but_does_not_claim_pass(self):
        row = self.classify('test running ... '+PROBE+PEER+'========= ERROR SUMMARY: 2 errors\n')
        self.assertEqual(row['status'], 'KNOWN_VENDOR_API_ONLY')
        self.assertEqual(row['counts'], {'kernel_availability': 1, 'peer_already_enabled': 1})
        self.assertEqual(row['raw_errors'], 2)
        self.assertIs(row['gate_pass'], False)

    def test_same_code_outside_specific_nccl_call_site_is_unknown(self):
        for bad in (PROBE.replace('ncclInitKernelsForDevice', 'mgbfs_kernel'),
                    PROBE.replace('enqueue.cc:52', 'enqueue.cc:99'),
                    PROBE.replace('cudaFuncGetAttributes.', 'cudaLaunchKernel.'),
                    PROBE.replace('libnccl.so.2', 'libmgbfs_cuda.so'),
                    PEER.replace('):331', '):999')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.classify(bad+'========= ERROR SUMMARY: 1 errors\n')

    def test_memory_error_warning_or_missing_diagnostic_is_fatal(self):
        for bad in (PROBE+'========= ERROR SUMMARY: 2 errors\n',
                    PROBE+'========= Invalid __global__ read of size 4\n========= ERROR SUMMARY: 1 errors\n',
                    PROBE+'========= Warning: tool internal error\n========= ERROR SUMMARY: 1 errors\n',
                    PROBE+'========= ERROR SUMMARY: 1 errors\n========= ERROR SUMMARY: 1 errors\n',
                    PROBE):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.classify(bad)

    def test_zero_summary_is_not_a_test_execution_proof(self):
        row = self.classify('========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n')
        self.assertEqual(row['status'], 'NO_REPORTED_FINDINGS')
        self.assertIs(row['gate_pass'], False)


if __name__ == '__main__':
    unittest.main()
