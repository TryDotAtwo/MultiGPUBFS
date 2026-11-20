"""Pin the segmented32 cache-reclamation release to its verified wheel."""
import argparse,hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TAG='native-pip-20261120-v38'

def main():
 p=argparse.ArgumentParser();p.add_argument('--wheel',type=Path,required=True);a=p.parse_args()
 if a.wheel.name!='multigpubfs-0.2.0.dev38-py3-none-any.whl':raise RuntimeError('UNEXPECTED_WHEEL_NAME')
 sha=hashlib.sha256(a.wheel.read_bytes()).hexdigest()
 doc=ROOT/'docs/install-native-package.md';text=doc.read_text()
 target=f'https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/{TAG}/{a.wheel.name}#sha256={sha}'
 text,count=re.subn(r'https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-[^/]+/multigpubfs-[^#"]+#sha256=[0-9a-f]+',target,text)
 if count!=1:raise RuntimeError('INSTALL_DOC_PIN_MISSING')
 text=re.sub(r'(https://github.com/TryDotAtwo/MultiGPUBFS/releases/tag/)native-pip-[A-Za-z0-9-]+',r'\g<1>'+TAG,text)
 doc.write_text(text)
 path=ROOT/'docs/experimental-memory-backends.md';text=path.read_text()
 old='''The dev37 release provides separately checksummed components for the
`segmented32_v2` ABI. Its executable and CUDA component are built from the
same source revision. Older dev36 components use the earlier 24-bit layout
and are not compatible. The launcher checks the native capability before BFS
and the actual backend in the receipt.'''
 new='''The dev38 release provides separately checksummed components for the
`segmented32_v2` ABI. Its executable includes the persistent-cache admission
fix; CUDA components are reused from dev37 after verifying the CUDA-source
digest is identical. Older dev36 components use the earlier 24-bit layout
and are not compatible. The launcher checks the native capability before BFS
and the actual backend in the receipt.'''
 if text.count(old)!=1:raise RuntimeError('MEMORY_DOC_BUILD_REQUIREMENT_CHANGED')
 path.write_text(text.replace(old,new))
 print('DOCS_READY wheel_sha256='+sha,flush=True)
if __name__=='__main__':main()
