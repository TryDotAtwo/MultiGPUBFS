"""Pin the ABI-3 memory-preset wheel only after its component assembly."""
import argparse
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = 'native-pip-20261120-v39'
WHEEL = 'multigpubfs-0.2.0.dev39-py3-none-any.whl'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wheel', type=Path, required=True)
    args = parser.parse_args()
    if args.wheel.name != WHEEL:
        raise RuntimeError('UNEXPECTED_WHEEL_NAME')
    sha = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
    path = ROOT / 'docs/install-native-package.md'
    text = path.read_text()
    target = f'https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/{TAG}/{WHEEL}#sha256={sha}'
    text, count = re.subn(
        r'https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-[^/]+/multigpubfs-[^#"]+#sha256=[0-9a-f]+',
        target, text)
    if count != 1:
        raise RuntimeError('INSTALL_DOC_PIN_MISSING')
    text = re.sub(r'(https://github.com/TryDotAtwo/MultiGPUBFS/releases/tag/)native-pip-[A-Za-z0-9-]+', r'\g<1>'+TAG, text)
    path.write_text(text)
    path = ROOT / 'docs/experimental-memory-backends.md'
    text = path.read_text()
    old = 'This candidate source requires `segmented_memory_presets_v1` and native ABI 3.'
    if text.count(old) != 1:
        raise RuntimeError('MEMORY_DOC_BUILD_REQUIREMENT_CHANGED')
    path.write_text(text.replace(old, 'The dev39 release requires `segmented_memory_presets_v1` and native ABI 3.'))
    print('DOCS_READY wheel_sha256='+sha, flush=True)


if __name__ == '__main__':
    main()
