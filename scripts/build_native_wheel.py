"""Package already-built Linux CUDA artifacts; no compilation or GPU launch."""
import argparse,hashlib,json,platform,shutil,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--architectures',required=True);p.add_argument('--cuda-version',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if platform.system()!='Linux' or platform.machine()!='x86_64':p.error('Linux x86_64 artifacts required')
architectures=[int(v) for v in a.architectures.split(',')]
if not architectures or any(v<70 or v>200 for v in architectures):p.error('explicit compiled SM architectures required')
source=Path(__file__).resolve().parents[1];stage=source/'multigpubfs/_native'
if stage.exists():p.error('existing native staging; do not replace unknown artifacts')
(stage/'bin').mkdir(parents=True);(stage/'lib').mkdir();shutil.copy2(a.binary,stage/'bin/mgbfs');shutil.copy2(a.library,stage/'lib/libmgbfs_cuda.so');(stage/'bin/mgbfs').chmod(0o755)
manifest={'schema':1,'cuda_toolkit':a.cuda_version,'architectures':architectures,'platform':'linux_x86_64','sha256':{name:hashlib.sha256((stage/name).read_bytes()).hexdigest() for name in ('bin/mgbfs','lib/libmgbfs_cuda.so')},'external_runtime':'CUDA runtime 12 and NCCL 2 shared libraries, compatible NVIDIA driver','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),'hardware_acceptance':'See separately published test evidence; architectures are build targets, not GPU acceptance'}
(stage/'manifest.json').write_text(json.dumps(manifest,indent=2));a.output.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,'-m','build','--wheel','--outdir',str(a.output.resolve())],cwd=source,check=True)
print(json.dumps(manifest))
