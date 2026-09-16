"""Build native + cuCollections SM90 from a clean pinned Linux checkout.

No GPU work, provisioning, billing control, or HF credentials are needed.
Requires host C++ toolchain, NCCL headers/library, curl, git, tar, Python >=3.10,
pip and venv. Dependency pins come from this checkout's validated Kaggle recipe.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time


def test_executable(output):
    artifacts, finished = set(), []
    for line in output.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if record.get('reason') == 'build-finished':
            finished.append(record.get('success'))
        target = record.get('target', {})
        if (record.get('reason') == 'compiler-artifact' and
                target.get('name') == 'library_multi_gpu' and
                'test' in target.get('kind', []) and record.get('executable')):
            artifacts.add(record['executable'])
    if finished != [True] or len(artifacts) != 1:
        raise ValueError('CARGO_EIGHT_GPU_EXECUTABLE_INVENTORY')
    return artifacts.pop()


def load_recipe(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    from eight_gpu_gate import run_command
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if (sys.platform != 'linux' or sys.version_info < (3, 10) or
            not re.fullmatch('[0-9a-f]{40}', args.commit) or
            min(args.jobs, args.timeout_seconds) <= 0):
        parser.error('Linux Python >=3.10, full commit, positive jobs and timeout required')
    source, work = args.source.resolve(), args.work.resolve()
    if not source.is_dir() or work.exists() or not work.parent.is_dir():
        parser.error('existing source and new work directory with existing parent required')
    if shutil.disk_usage(work.parent).free < 15 * 1024**3:
        parser.error('at least 15 GiB free required for build; not an archive budget')
    for tool in ('git', 'curl', 'tar', 'c++'):
        if shutil.which(tool) is None:
            parser.error('missing host tool: ' + tool)
    work.mkdir()
    logs = work/'logs'; logs.mkdir()
    deadline = time.monotonic() + args.timeout_seconds
    env = {k:v for k,v in os.environ.items()
           if not k.startswith('MGBFS_') and k not in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN',
               'PYTHONHOME', 'PYTHONPATH', 'CARGO_TARGET_DIR', 'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS')}
    report = dict(status='INCOMPLETE', source=str(source), source_commit=args.commit,
                  cuda_architectures='90', scope='compile only; no hardware correctness')

    def run(command, name):
        print('BUILD_STAGE ' + name, flush=True)
        return run_command(list(map(str, command)), logs/(name+'.log'), env,
                           deadline-time.monotonic())

    def checkout(url, commit, destination, name):
        destination.mkdir()
        run(['git', '-C', destination, 'init', '-q'], name+'-init')
        run(['git', '-C', destination, 'fetch', '--depth=1', url, commit], name+'-fetch')
        run(['git', '-C', destination, 'checkout', '--detach', 'FETCH_HEAD'], name+'-checkout')
        if run(['git', '-C', destination, 'rev-parse', 'HEAD'], name+'-sha').strip() != commit:
            raise ValueError('DEPENDENCY_COMMIT_MISMATCH')

    try:
        if run(['git', '-C', source, 'rev-parse', 'HEAD'], 'source-sha').strip() != args.commit:
            raise ValueError('SOURCE_COMMIT_MISMATCH')
        if run(['git', '-C', source, 'status', '--porcelain', '--untracked-files=normal'], 'source-dirty').strip():
            raise ValueError('SOURCE_CHANGES')
        recipe = load_recipe(source/'kaggle/library-owner/kernel.py', 'library_build_recipe')
        primitives = load_recipe(source/'kaggle/native-primitives/kernel.py', 'primitive_build_recipe')
        report.update(cuda_components=recipe.CUDA_COMPONENTS, cuco_commit=recipe.CUCO_COMMIT,
                      cutlass_commit=primitives.CUTLASS_COMMIT, rust_version=primitives.RUST_VERSION)
        sdk = work/'cuda-12.9'; sdk.mkdir()
        for component, version, digest in recipe.CUDA_COMPONENTS:
            name = f'{component}-linux-x86_64-{version}-archive'
            archive = work/(name+'.tar.xz')
            url = f'https://developer.download.nvidia.com/compute/cuda/redist/{component}/linux-x86_64/{archive.name}'
            run(['curl', '--fail', '--location', '--max-time', '180', url, '--output', archive], component+'-download')
            with archive.open('rb') as package:
                checksum = hashlib.sha256()
                for chunk in iter(lambda: package.read(1 << 20), b''):
                    checksum.update(chunk)
                if checksum.hexdigest() != digest:
                    raise ValueError('CUDA_ARCHIVE_CHECKSUM: '+component)
            run(['tar', '-xf', archive, '-C', work], component+'-extract')
            shutil.copytree(work/name, sdk, dirs_exist_ok=True)
        (sdk/'lib64').symlink_to('lib', target_is_directory=True)
        if not (sdk/'lib64/libcudart_static.a').is_file():
            raise ValueError('CUDA_STATIC_RUNTIME_MISSING')
        env['CUDACXX'] = str(sdk/'bin/nvcc')
        env['PATH'] = str(sdk/'bin') + ':' + env['PATH']
        run([sdk/'bin/nvcc', '--version'], 'cuda-version')
        venv = work/'venv'
        run([sys.executable, '-m', 'venv', '--without-pip', venv], 'venv')
        python = venv/'bin/python'
        pip = [sys.executable, '-m', 'pip', '--python', python]
        requirements = source/'experiments/library_owner/requirements-linux-x86_64.lock'
        report['requirements_sha256'] = hashlib.sha256(requirements.read_bytes()).hexdigest()
        run([*pip, 'install', '--only-binary=:all:', '--no-cache-dir', '--require-hashes',
             '--report', logs/'pip-install.json', '-r', requirements], 'dependencies')
        run([*pip, 'freeze', '--all'], 'packages')
        # Explicit venv layout avoids parsing a path from merged stdout/stderr.
        sites = list((venv/'lib').glob('python*/site-packages'))
        if len(sites) != 1:
            raise ValueError('VENV_SITE_INVENTORY')
        site = sites[0]
        prefixes = recipe.cmake_prefixes(site)
        libs = sorted({str(p.parent) for p in site.rglob('*.so*') if p.is_file()})
        env['LD_LIBRARY_PATH'] = ':'.join([str(sdk/'lib'), *libs, env.get('LD_LIBRARY_PATH', '')])
        env['PATH'] = str(venv/'bin') + ':' + env['PATH']
        cuco, cutlass = work/'cuco', work/'cutlass'
        checkout('https://github.com/NVIDIA/cuCollections.git', recipe.CUCO_COMMIT, cuco, 'cuco')
        checkout('https://github.com/NVIDIA/cutlass.git', primitives.CUTLASS_COMMIT, cutlass, 'cutlass')
        library, native = work/'library-build', work/'native-build'
        cmake = venv/'bin/cmake'
        common = ['-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=90',
                  '-DCMAKE_CUDA_COMPILER='+str(sdk/'bin/nvcc')]
        run([cmake, '-S', source/'experiments/library_owner', '-B', library, *common,
             '-DCUDAToolkit_ROOT='+str(sdk), '-DCMAKE_PREFIX_PATH='+';'.join(prefixes),
             '-DCUCO_ROOT='+str(cuco)], 'library-configure')
        run([cmake, '--build', library, '--parallel', args.jobs], 'library-build')
        run([cmake, '-S', source/'cuda', '-B', native, *common,
             '-DBUILD_TESTING=OFF', '-DCUTLASS_ROOT='+str(cutlass)], 'native-configure')
        run([cmake, '--build', native, '--target', 'mgbfs_cuda', '--parallel', args.jobs], 'native-build')
        env['CARGO_HOME'], env['RUSTUP_HOME'] = str(work/'cargo'), str(work/'rustup')
        installer = work/'rustup-init.sh'
        run(['curl', '--fail', '--location', '--max-time', '180', 'https://sh.rustup.rs', '-o', installer], 'rust-download')
        run(['sh', installer, '-y', '--no-modify-path', '--profile', 'minimal',
             '--default-toolchain', primitives.RUST_VERSION], 'rust-install')
        env['PATH'] = str(work/'cargo/bin') + ':' + env['PATH']
        env['MGBFS_CUDA_LIB_DIR'] = str(native)
        env['MGBFS_LIBRARY_OWNER_LIB_DIR'] = str(library)
        env['MGBFS_CUDART_LIB_DIR'] = str(sdk/'lib')
        env['LD_LIBRARY_PATH'] = ':'.join([str(native), str(library), env['LD_LIBRARY_PATH']])
        run(['rustc', '--version', '--verbose'], 'rust-version')
        manifest = source/'Cargo.toml'
        artifacts = run(['cargo', 'test', '--manifest-path', manifest, '--locked', '--release',
                         '-p', 'mgbfs-runtime', '--features', 'cuda,library-owner', '--test',
                         'library_multi_gpu', '--no-run', '--message-format=json'], 'rust-oracle-build')
        executable = test_executable(artifacts)
        run(['cargo', 'build', '--manifest-path', manifest, '--locked', '--release',
             '-p', 'mgbfs-cli', '--features', 'library-owner'], 'cli-build')
        report.update(status='BUILT', oracle_executable=executable,
                      cli_executable=str(source/'target/release/mgbfs'))
        # Export only build/runtime paths. Never persist inherited credentials.
        exports = {k:env[k] for k in ('PATH', 'LD_LIBRARY_PATH', 'CARGO_HOME', 'RUSTUP_HOME',
                   'MGBFS_CUDA_LIB_DIR', 'MGBFS_LIBRARY_OWNER_LIB_DIR', 'MGBFS_CUDART_LIB_DIR')}
        (work/'runtime-env.json').write_text(json.dumps(exports, indent=2), encoding='utf-8')
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (work/'build-summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
