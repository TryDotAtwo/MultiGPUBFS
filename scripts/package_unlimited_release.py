"""Build the CLI against pinned dev30 CUDA libraries; reuse all verified SM binaries."""
import gzip,hashlib,io,json,os,shutil,subprocess,tarfile,urllib.request
from pathlib import Path
BASE="https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/"
TAG="native-pip-20261019-v31"
INDEX_SHA="a468c11a90e2fe32c41b7d4e9d5c8614bdbfc2ceaa1890b5b8ef99234a6c3940"
def digest(data):return hashlib.sha256(data).hexdigest()
def download(url,path,sha):
 if not path.exists() or digest(path.read_bytes())!=sha:
  with urllib.request.urlopen(url,timeout=120) as src,path.open("wb") as dst:shutil.copyfileobj(src,dst)
 if digest(path.read_bytes())!=sha:raise RuntimeError("DOWNLOAD_CHECKSUM")
def unpack(component,cache,destination):
 archive=cache/(component["sha256"]+".tar.gz");download(component["url"],archive,component["sha256"])
 with tarfile.open(archive) as tar:
  for name,expected in component["files"].items():
   rel=Path(name)
   if rel.is_absolute() or ".." in rel.parts:raise ValueError("UNSAFE_COMPONENT_PATH")
   member=tar.getmember(name)
   if not member.isfile():raise ValueError("COMPONENT_NOT_REGULAR")
   data=tar.extractfile(member).read()
   if len(data)!=expected["size"] or digest(data)!=expected["sha256"]:raise ValueError("COMPONENT_CHECKSUM")
   p=destination/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);p.chmod(expected["mode"])
def main():
 root=Path(__file__).resolve().parents[1];work=root/"build-unlimited";work.mkdir(exist_ok=True)
 indexfile=work/"baseline-index.json";download(BASE+"native-pip-20261019-v30/native-assets.json",indexfile,INDEX_SHA)
 index=json.loads(indexfile.read_text());common=work/"common";link=work/"link";common.mkdir(exist_ok=True);link.mkdir(exist_ok=True)
 unpack(index["components"][index["common"]],work,common)
 unpack(index["components"][index["production"]["86"]],work,link)
 for p in (common/"lib").iterdir():shutil.copy2(p,link/p.name)
 (link/"libcudart.so").unlink(missing_ok=True);(link/"libcudart.so").symlink_to("libcudart.so.12")
 env=dict(os.environ,MGBFS_CUDA_LIB_DIR=str(link),MGBFS_CUDART_LIB_DIR=str(link),RUSTFLAGS="-C link-arg=-Wl,-rpath,$ORIGIN/../lib")
 subprocess.run(["cargo","build","--release","--locked","-p","mgbfs-cli","--features","cuda"],cwd=root,env=env,check=True)
 binary=root/"target/release/mgbfs";shutil.copy2(binary,common/"bin/mgbfs")
 (common/"bin/mgbfs").chmod(0o755)
 output=root/"release-unlimited";output.mkdir(exist_ok=False)
 component=index["components"][index["common"]]
 for name,info in component["files"].items():
  data=(common/name).read_bytes();info.update(sha256=digest(data),size=len(data))
 filename="common-dev31.tar.gz";dest=output/filename
 with dest.open("wb") as raw,gzip.GzipFile(filename="",fileobj=raw,mode="wb",mtime=0) as gz,tarfile.open(fileobj=gz,mode="w") as tar:
  for name,info in sorted(component["files"].items()):
   data=(common/name).read_bytes();entry=tarfile.TarInfo(name);entry.mode=info["mode"];entry.size=len(data);entry.mtime=0;tar.addfile(entry,io.BytesIO(data))
 component.update(url=BASE+TAG+"/"+filename,sha256=digest(dest.read_bytes()),size=dest.stat().st_size)
 source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
 component["build"]={"source_commit":source,"scope":"Rust CLI rebuilt; CUDA runtime and NCCL unchanged"}
 index.update(id="unlimited-20261019-v31",scope="Optional BFS work deadlines; bounded startup and diagnostics; dev30 CUDA components unchanged")
 from multigpubfs.native_assets import AssetManager
 AssetManager(index,work/"schema-check")
 raw=(json.dumps(index,sort_keys=True,separators=(",",":"))+"\n").encode()
 (output/"native-assets.json").write_bytes(raw);(root/"multigpubfs/native_assets.json").write_bytes(raw)
 receipt=dict(schema=1,source_commit=source,native_sha256=digest(binary.read_bytes()),common_sha256=component["sha256"],native_index_sha256=digest(raw),cuda_components_unchanged=True)
 (output/"build-receipt.json").write_text(json.dumps(receipt,indent=2))
 print(json.dumps(receipt))
if __name__=="__main__":main()
