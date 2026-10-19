"""Read back and install the published wheel. CPU/file acceptance, not GPU acceptance."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

BASE = "https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261019-v30/"
HASHES = {
    "multigpubfs-0.2.0.dev30-py3-none-any.whl": "39ed27a8dfc8f45f1ec36a907d5a0a74a24f91266a785610840afbef2ea111ce",
    "native-assets.json": "a468c11a90e2fe32c41b7d4e9d5c8614bdbfc2ceaa1890b5b8ef99234a6c3940",
}
root = Path(tempfile.mkdtemp(prefix="dev30-public-"))
for name, digest in HASHES.items():
    with urllib.request.urlopen(BASE + name, timeout=60) as response:
        data = response.read()
    assert hashlib.sha256(data).hexdigest() == digest, name
    (root / name).write_bytes(data)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
                str(root / "multigpubfs-0.2.0.dev30-py3-none-any.whl")], check=True)
import multigpubfs
assert importlib.metadata.version("multigpubfs") == "0.2.0.dev30"
assert "site-packages" in multigpubfs.__file__ or "dist-packages" in multigpubfs.__file__, multigpubfs.__file__
from multigpubfs.native_assets import AssetManager
installed = Path(multigpubfs.__file__).with_name("native_assets.json").read_bytes()
assert installed == (root / "native-assets.json").read_bytes()
index = json.loads(installed)
manager = AssetManager(index, root / "cache")
prefetch = manager.prefetch(["86"])
hardware = {"driver_api": 12090, "devices": [
    {"logical_id": i, "compute_capability": [8, 6]} for i in range(8)]}
bundle = manager.prepare(hardware)
library_sha = hashlib.sha256((bundle / "lib/libmgbfs_cuda.so").read_bytes()).hexdigest()
assert library_sha == "a4a36824bb69d57b63a2c30a32e289c37e0b96db32377321afad32ad9bf47818"
env = dict(os.environ, LD_LIBRARY_PATH=str(bundle / "lib"))
key = subprocess.run([str(bundle / "bin/mgbfs"), "key-info"],
                     env=env, check=True, capture_output=True, text=True, timeout=30)
assert json.loads(key.stdout)["lossless_bitpack128_feistel_v1"] is True
example = Path(__file__).resolve().parents[1] / "examples/ut47_eight_gpu.py"
spec = importlib.util.spec_from_file_location("ut47_example", example)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
from cayleypy import CayleyGraph
from multigpubfs.graph_definition import from_cayleypy
graph = from_cayleypy(CayleyGraph(module.definition(), device="cpu"))
expected = json.loads((Path(__file__).resolve().parents[1] / "docs/acceptance/cuda999-ut47-graph.json").read_text())
actual = json.loads(graph.to_json())
for name in ("action", "start"):
    assert actual[name] == expected[name], name
receipt = dict(status="PUBLIC_WHEEL_INSTALLED_CPU_CHECKED",
               wheel_sha256=HASHES["multigpubfs-0.2.0.dev30-py3-none-any.whl"],
               native_library_sha256=library_sha, graph_matches_original=True,
               gpu_executed=False, scope="Public artifact readback, installed index/cache/native loader and real CayleyPy graph construction; eight-GPU execution is covered by the identical library receipt")
Path("/tmp/dev30-installed-acceptance.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
