import hashlib
from pathlib import Path
import pytest
from scripts import finalize_segmented32_release as release

def test_release_documents_exact_wheel(monkeypatch,tmp_path):
    docs=tmp_path/'docs';docs.mkdir()
    d=docs/'install-native-package.md'
    d.write_text('pip https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-old/multigpubfs-old.whl#sha256='+'a'*64+'\n[release](https://github.com/TryDotAtwo/MultiGPUBFS/releases/tag/native-pip-old)')
    arch=docs/'experimental-memory-backends.md'
    arch.write_text('''The 32-bit-row variant requires native ABI `segmented32_v2`: both the
executable and CUDA component must be built from the same source revision.
Older dev36 components have the earlier 24-bit layout and are not compatible.
The launcher checks the native capability before BFS and the actual backend
in the receipt.''')
    wheel=tmp_path/'multigpubfs-0.2.0.dev37-py3-none-any.whl';wheel.write_bytes(b'dummy')
    monkeypatch.setattr(release,'ROOT',tmp_path)
    monkeypatch.setattr('sys.argv',['finalize','--wheel',str(wheel)])
    release.main()
    assert hashlib.sha256(b'dummy').hexdigest() in d.read_text()
    assert release.TAG in d.read_text()
    assert 'dev37 release' in arch.read_text()

def test_release_rejects_wrong_wheel(monkeypatch,tmp_path):
    wheel=tmp_path/'wrong.whl';wheel.write_bytes(b'')
    monkeypatch.setattr('sys.argv',['finalize','--wheel',str(wheel)])
    with pytest.raises(RuntimeError,match='UNEXPECTED_WHEEL_NAME'):
        release.main()
