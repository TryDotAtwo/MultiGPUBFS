import pathlib,urllib.request,subprocess,json,sys,hashlib
SOURCE="d23133e9cc0fddd6f2a3907442215ec86533f0aa"
root=pathlib.Path("/tmp/mgbfs-nsight-alias-red");root.mkdir()
out=pathlib.Path("/kaggle/working");manifest=[]
for name in ["tests/test_nsight_paired_launch_aliases.py","scripts/validation/kaggle-dense-packet-timeline-evidence.py"]:
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
 b=urllib.request.urlopen("https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/"+SOURCE+"/"+name).read();p.write_bytes(b);manifest.append({"name":name,"sha256":hashlib.sha256(b).hexdigest()})
(out/"source-manifest.json").write_text(json.dumps(manifest,indent=2))
r=subprocess.run([sys.executable,str(root/"tests/test_nsight_paired_launch_aliases.py")],capture_output=True,text=True,timeout=30)
(out/"test.log").write_text(r.stdout+r.stderr)
expected=r.returncode==0 and "OK" in r.stderr
(out/"summary.json").write_text(json.dumps({"source":SOURCE,"status":"ALIAS_GREEN" if expected else "UNEXPECTED_RESULT","returncode":r.returncode,"expected_two_alias_failures":expected},indent=2))
if not expected:raise RuntimeError("Unexpected Nsight alias GREEN outcome")
