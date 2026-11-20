import hashlib
from pathlib import Path
import pytest
from scripts import finalize_vram_release as release


def test_release_pins_exact_wheel_and_abi(monkeypatch, tmp_path):
    docs = tmp_path / 'docs'
    docs.mkdir()
    install = docs / 'install-native-package.md'
    install.write_text('pip https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-old/multigpubfs-old.whl#sha256=' + 'a' * 64)
    backend = docs / 'experimental-memory-backends.md'
    backend.write_text('This candidate source requires `segmented_memory_presets_v1` and native ABI 3.')
    wheel = tmp_path / release.WHEEL
    wheel.write_bytes(b'signed by checksum')
    monkeypatch.setattr(release, 'ROOT', tmp_path)
    monkeypatch.setattr('sys.argv', ['finalize', '--wheel', str(wheel)])
    release.main()
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() in install.read_text()
    assert release.TAG in install.read_text()
    assert 'dev39 release' in backend.read_text()


def test_release_refuses_wrong_wheel(monkeypatch, tmp_path):
    monkeypatch.setattr('sys.argv', ['finalize', '--wheel', str(tmp_path / 'wrong.whl')])
    with pytest.raises(RuntimeError, match='UNEXPECTED_WHEEL_NAME'):
        release.main()
