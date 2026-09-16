"""Classify raw memcheck diagnostics; never grants a correctness-gate PASS.

Signatures are source-backed only for NCCL v2.28.9-1, commit
dbc86fd06e8b0c4517b95d8958a09ccacf9520c9, with debug host frames.
No sanitizer output is suppressed or edited. Unknown/truncated output fails.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def classify(text):
    summaries = re.findall(r'========= ERROR SUMMARY: (\d+) errors\b', text)
    if len(summaries) != 1:
        raise ValueError('ONE_COMPLETE_SUMMARY_REQUIRED')
    # The test harness can prefix the first marker with its test name.
    messages = [line.split('========= ', 1)[1].strip()
                for line in text.splitlines() if '========= ' in line]
    for msg in messages:
        if (not msg or msg == 'COMPUTE-SANITIZER' or
                msg.startswith(('Program hit ', 'Host Frame: ')) or
                msg == 'Saved host backtrace up to driver entry point at error' or
                re.fullmatch(r'ERROR SUMMARY: \d+ errors', msg)):
            continue
        raise ValueError('UNCLASSIFIED_DIAGNOSTIC: ' + msg[:160])
    counts = dict(kernel_availability=0, peer_already_enabled=0)
    blocks = re.split(r'========= Program hit ', text)[1:]
    for block in blocks:
        header = block.splitlines()[0]
        frames = re.findall(r'Host Frame: ([^\n]+)', block)
        first = frames[0] if frames else ''
        category = None
        if first.endswith(' in libnccl.so.2'):
            for api, line in [('cudaFuncGetAttributes', 52), ('cudaGetLastError', 54)]:
                if (header.startswith('cudaErrorNoKernelImageForDevice (error 209) ') and
                        header.endswith('on CUDA API call to '+api+'.') and
                        first.startswith('ncclInitKernelsForDevice(') and
                        f' in enqueue.cc:{line} [' in first):
                    category = 'kernel_availability'
            if (header.startswith('p2pMap(') and '(error 704)' in header and
                    'due to "transport/p2p.cc"' in header and
                    header.endswith('on CUDA API call to cudaGetLastError.') and
                    first.startswith('p2pRecvConnect(') and
                    ' in ncclTransportP2pSetup(' in first and '):331 [' in first):
                category = 'peer_already_enabled'
        if category is None:
            raise ValueError('UNKNOWN_API_CALL_SITE: '+header[:160])
        counts[category] += 1
    raw_errors = int(summaries[0])
    if raw_errors != len(blocks):
        raise ValueError('TRUNCATED_OR_UNACCOUNTED_DIAGNOSTICS')
    return dict(status='KNOWN_VENDOR_API_ONLY' if raw_errors else 'NO_REPORTED_FINDINGS',
                gate_pass=False, raw_errors=raw_errors, counts=counts,
                log_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
                scope='diagnostic classification only; not test execution or library identity proof')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    args = parser.parse_args()
    print(json.dumps(classify(args.log.read_text(encoding='utf-8', errors='strict')), indent=2))
