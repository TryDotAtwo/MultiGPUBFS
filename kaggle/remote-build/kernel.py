"""Compile-only validation of the standalone H200 build recipe on Kaggle.

Uses the CUDA-capable Kaggle image for NCCL development libraries. Does not
execute SM90 binaries on T4 or claim any H200 runtime correctness.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE_COMMIT = 'cbc092bd980f7600e14e90719db84f82025d4aa3'
TIMEOUT_SECONDS = 3600


def main():
    root = Path(tempfile.mkdtemp(prefix='mgbfs-standalone-', dir='/tmp'))
    source, build = root/'source', root/'build'
    output = Path('/kaggle/working/standalone-build')
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(status='INCOMPLETE', source_commit=SOURCE_COMMIT,
                   scope='SM90 compile only; no GPU execution')
    try:
        subprocess.run(['git', 'clone', '--no-checkout', '--filter=blob:none',
                        'https://github.com/TryDotAtwo/MultiGPUBFS.git', str(source)],
                       check=True, timeout=180)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', SOURCE_COMMIT],
                       check=True, timeout=180)
        subprocess.run([sys.executable, '-u', str(source/'scripts/remote_build.py'),
                        '--source', str(source), '--commit', SOURCE_COMMIT,
                        '--work', str(build), '--jobs', '2',
                        '--timeout-seconds', str(TIMEOUT_SECONDS)], check=True)
        report = json.loads((build/'build-summary.json').read_text())
        if report['status'] != 'BUILT':
            raise RuntimeError('BUILD_NOT_COMPLETE')
        summary['status'] = 'BUILT'
    finally:
        # Preserve only manifests/logs; large SDK/build trees stay under /tmp.
        if (build/'logs').is_dir():
            shutil.copytree(build/'logs', output/'logs', dirs_exist_ok=True)
        if (build/'build-summary.json').is_file():
            shutil.copy2(build/'build-summary.json', output/'build-summary.json')
        (output/'kernel-summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
