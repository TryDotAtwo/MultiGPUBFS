"""Private Kaggle 1xT4 A/B: native development stepper vs CayleyPy BFS."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

SOURCE_COMMIT = '7b7137a3ffcc26300799df9857a06d33ce945376'
BASELINE_COMMIT = 'f0f2b8e5ee61173039ab9742f3a7756c9b6365e6'
CUTLASS_COMMIT = 'ffa119a1255d78998536107466cc7097ecefa393'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(tempfile.mkdtemp(prefix='mgbfs-ab-', dir='/tmp'))
    logs = Path('/kaggle/working/single-gpu-bench')
    logs.mkdir()
    # Bootstrap helpers are themselves immutable, publicly readable source.
    helper = root/'bootstrap.py'
    urllib.request.urlretrieve('https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/c6b501c5e245ff15d92bfbc018c6bc25b0e68c98/kaggle/native-primitives/kernel.py', helper)
    gate = load(helper, 'gate')
    env = os.environ.copy()
    env['PATH'] = '/usr/local/cuda/bin:' + env.get('PATH', '')
    def run(command, name, cwd=root, timeout=900):
        return gate.run(command, cwd=cwd, env=env, logs=logs, name=name, timeout=timeout)
    inventory = run(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,memory.free', '--format=csv,noheader,nounits'], 'inventory')
    gpus = gate.validate_gpus(inventory)
    run(['nvidia-smi', '-q'], 'gpu-environment-before')
    source, baseline, cutlass = root/'source', root/'baseline', root/'cutlass'
    gate.checkout('https://github.com/TryDotAtwo/MultiGPUBFS.git', SOURCE_COMMIT, source, env, logs, 'native-source')
    gate.checkout('https://github.com/TryDotAtwo/cayleypy.git', BASELINE_COMMIT, baseline, env, logs, 'baseline-source')
    gate.checkout('https://github.com/NVIDIA/cutlass.git', CUTLASS_COMMIT, cutlass, env, logs, 'cutlass-source')
    env['CARGO_HOME'] = str(root/'cargo')
    env['RUSTUP_HOME'] = str(root/'rustup')
    installer = root/'rustup.sh'
    urllib.request.urlretrieve('https://sh.rustup.rs', installer)
    run(['sh', str(installer), '-y', '--no-modify-path', '--profile', 'minimal', '--default-toolchain', '1.75.0'], 'rust-install')
    env['PATH'] = str(root/'cargo/bin') + ':' + env['PATH']
    build = source/'build/native-cuda'
    env['MGBFS_CUDA_LIB_DIR'] = str(build)
    env['LD_LIBRARY_PATH'] = str(build)+':/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH', '')
    run(['nvcc', '--version'], 'cuda-version')
    run(['cmake', '-S', 'cuda', '-B', str(build), '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=75', '-DCUTLASS_ROOT='+str(cutlass)], 'cmake', source)
    run(['cmake', '--build', str(build), '--parallel', '2'], 'cuda-build', source)
    run(['cargo', 'build', '--locked', '--release', '-p', 'mgbfs-runtime', '--features', 'cuda', '--example', 'single_gpu_bench'], 'rust-build', source)
    run(['python', '-m', 'pip', 'freeze'], 'python-environment')
    env['PYTHONPATH'] = str(baseline)
    env['CUDA_VISIBLE_DEVICES'] = '0'
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE'):
        env.pop(key, None)
    (logs/'environment.json').write_text(json.dumps(dict(native_commit=SOURCE_COMMIT, baseline_commit=BASELINE_COMMIT,
        cutlass_commit=CUTLASS_COMMIT, measured_gpu=gpus[0], unused_gpu=gpus[1], rust='1.75.0',
        native_capacity='min(m**6,32000000), fixed per worker; 1 GiB reserve', archive=False,
        workloads=[16,20,24,28,32], minimum_seconds=60, hash_seed=20260828,
        sampler='nvidia-smi 50ms process lifetime; includes warmup'), indent=2))
    sys.path.insert(0, str(source/'scripts'))
    bench = load(source/'scripts/large_gpu_bench.py', 'bench')
    bench.suite(source/'target/release/examples/single_gpu_bench', logs, env)
    run(['nvidia-smi', '-q'], 'gpu-environment-after')


if __name__ == '__main__':
    main()
