"""Install the pinned GitHub release into Kaggle with isolated native dependencies. Activation stays explicit."""
def setup():
    import hashlib, importlib.machinery, importlib.util, subprocess, sys, sysconfig
    from pathlib import Path
    import urllib.request
    wheel = Path('/tmp/multigpubfs-0.2.0.dev0-py3-none-linux_x86_64.whl')
    url = 'https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-419fa34f-cuda12.9-sm75-sm86/multigpubfs-0.2.0.dev0-py3-none-linux_x86_64.whl'
    print('MultiGPUBFS: downloading pinned GitHub release', flush=True)
    with urllib.request.urlopen(url, timeout=90) as response:
        wheel.write_bytes(response.read())
    expected = 'f85b5d3f90bb6c870bc60b3fbc76ef920284a83a443185090aa81643f3bee9aa'
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != expected:
        raise RuntimeError('MultiGPUBFS package checksum mismatch')
    logs = Path('/kaggle/working/multigpubfs-install.log')
    logs.parent.mkdir(parents=True, exist_ok=True)
    print('MultiGPUBFS: installing CayleyPy and native package', flush=True)
    with logs.open('w') as stream:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cayleypy==0.2.0'],
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=240)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', str(wheel)],
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=240)
        print('MultiGPUBFS: installing isolated CUDA/NCCL dependencies', flush=True)
        runtime = Path('/tmp/multigpubfs-runtime')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
                        '--target', str(runtime), 'nvidia-cuda-runtime-cu12==12.9.79',
                        'nvidia-nccl-cu12==2.30.7'],
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=240)
    # Private native dependency directory precedes Kaggle torch's NCCL/runtime.
    # Keep torch's installed dependencies intact.
    lib = Path(sysconfig.get_paths()['purelib'])/'multigpubfs/_native/lib'
    (lib.parent/'bin/mgbfs').chmod(0o755)
    for library in runtime.rglob('*.so*'):
        target = lib/library.name
        if library.is_file() and not target.exists(): target.symlink_to(library)
    spec = importlib.machinery.PathFinder.find_spec('multigpubfs', [sysconfig.get_paths()['purelib']])
    if spec is None: raise ImportError('Native MultiGPUBFS package installation failed')
    module = importlib.util.module_from_spec(spec)
    sys.modules['multigpubfs'] = module
    spec.loader.exec_module(module)
    print('MultiGPUBFS: native package ready; call enable_native() explicitly', flush=True)
    return module

