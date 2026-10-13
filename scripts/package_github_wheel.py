"""Bundle the accepted native binary with private CUDA/NCCL redistributables."""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, urllib.request, zipfile
from pathlib import Path

BASE_URL = 'https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-419fa34f-cuda12.9-sm75-sm86/multigpubfs-0.2.0.dev0-py3-none-linux_x86_64.whl'
BASE_SHA = 'f85b5d3f90bb6c870bc60b3fbc76ef920284a83a443185090aa81643f3bee9aa'
RUNTIMES = ('nvidia-cuda-runtime-cu12==12.9.79', 'nvidia-nccl-cu12==2.30.7')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output', type=Path, required=True);p.add_argument('--binary',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--native-manifest',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];stage=root/'multigpubfs/_native'
    if stage.exists():raise RuntimeError('Refusing to replace existing native staging')
    with tempfile.TemporaryDirectory() as temp:
        temp=Path(temp);base=temp/'accepted.whl'
        if a.binary and a.library:
            (stage/'bin').mkdir(parents=True);(stage/'lib').mkdir()
            shutil.copy2(a.binary,stage/'bin/mgbfs');shutil.copy2(a.library,stage/'lib/libmgbfs_cuda.so')
            manifest=(json.loads(a.native_manifest.read_text()) if a.native_manifest else dict(schema=1,cuda_toolkit='12.9.86',architectures=[75,86],platform='linux_x86_64',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),hardware_acceptance='Build targets; see separate GPU acceptance receipt'))
            (stage/'manifest.json').write_text(json.dumps(manifest))
        else:
            raise RuntimeError('Explicit portable rebuilt binary and library required')
        (stage/'bin/mgbfs').chmod(0o755)
        deps=temp/'deps';deps.mkdir()
        subprocess.run([sys.executable,'-m','pip','download','--no-deps','--only-binary=:all:','-d',str(deps),*RUNTIMES],check=True)
        (stage/'licenses').mkdir()
        runtime_receipts=[]
        for wheel in sorted(deps.glob('*.whl')):
            copied=[];licenses=[]
            with zipfile.ZipFile(wheel) as z:
                for name in z.namelist():
                    if name.endswith('/') :continue
                    filename=Path(name).name
                    if '/lib/' in name and ('.so' in filename):
                        target=stage/'lib'/filename
                        if target.exists():raise RuntimeError('RUNTIME_FILENAME_COLLISION '+filename)
                        target.write_bytes(z.read(name));copied.append(filename)
                    elif 'license' in filename.lower() or 'notice' in filename.lower():
                        target=stage/'licenses'/(wheel.name.split('-')[0]+'-'+filename)
                        if target.exists():raise RuntimeError('LICENSE_FILENAME_COLLISION')
                        target.write_bytes(z.read(name));licenses.append(target.name)
            if not copied or not licenses:raise RuntimeError('Missing redistributables/licenses in '+wheel.name)
            runtime_receipts.append(dict(wheel=wheel.name,sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),libraries=copied,licenses=licenses))
        # Real SONAME files, not symlinks that disappear in wheels.
        for path in list((stage/'lib').glob('*.so*')):
            dynamic=subprocess.check_output(['readelf','-d',str(path)],text=True)
            import re
            found=re.search(r'\(SONAME\).*?\[(.*?)\]',dynamic)
            if found and not (stage/'lib'/found.group(1)).exists():
                shutil.copy2(path,stage/'lib'/found.group(1))
        # Private runtime resolution also works without caller LD_LIBRARY_PATH.
        subprocess.run(['patchelf','--set-rpath','$ORIGIN/../lib',str(stage/'bin/mgbfs')],check=True)
        subprocess.run(['patchelf','--set-rpath','$ORIGIN',str(stage/'lib/libmgbfs_cuda.so')],check=True)
        manifest=json.loads((stage/'manifest.json').read_text())
        manifest.update(bundled_runtime=runtime_receipts,package_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),external_runtime='Compatible NVIDIA host driver and Linux system ABI only; CUDA runtime/NCCL are private bundled libraries')
        manifest['sha256']={str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.rglob('*')) if p.is_file() and p.name!='manifest.json'}
        sys.path.insert(0,str(root))
        from multigpubfs.artifact_metadata import elf_runtime_requirements
        manifest['elf_runtime_requirements']=elf_runtime_requirements([stage/'bin/mgbfs',*sorted((stage/'lib').glob('*.so*'))])
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2))
        a.output.mkdir(parents=True,exist_ok=True)
        subprocess.run([sys.executable,'-m','build','--wheel','--outdir',str(a.output.resolve())],cwd=root,check=True)
        wheel=next(a.output.glob('multigpubfs-0.2.0.dev2-*.whl'))
        sha=hashlib.sha256(wheel.read_bytes()).hexdigest()
        (a.output/'SHA256SUMS').write_text(sha+'  '+wheel.name+'\n')
        receipt=dict(schema=1,wheel=wheel.name,sha256=sha,size=wheel.stat().st_size,manifest=manifest,status='BUILT_NEEDS_GPU_ACCEPTANCE')
        (a.output/'manifest.json').write_text(json.dumps(receipt,indent=2))
        print(json.dumps(dict(wheel=wheel.name,size=wheel.stat().st_size,sha256=sha,requirements=manifest['elf_runtime_requirements'])))

if __name__=='__main__':main()
