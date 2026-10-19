"""One-shot eight-GPU UT diagnostic; isolated from the installed BFS implementation."""
import argparse
import ctypes as C
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
import zipfile

WHEEL = "https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261017-v29/multigpubfs-0.2.0.dev29-py3-none-any.whl#sha256=ab5d6452dd1588bf4c1a003ad911b4cea03d3e6e61e7a6ea3cc18d2d40bf7a51"
EXPECTED = [1, 6, 26, 110, 414]
MODES = ("baseline", "no-gdr", "no-ib")
SAFE_ENV = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_MODULE_LOADING",
            "NCCL_NET_GDR_LEVEL", "NCCL_IB_DISABLE", "NCCL_P2P_DISABLE",
            "NCCL_P2P_LEVEL", "NCCL_CUMEM_ENABLE", "NCCL_CUMEM_HOST_ENABLE",
            "MGBFS_NATIVE_CACHE", "MGBFS_NATIVE_OFFLINE", "LD_PRELOAD", "LD_LIBRARY_PATH")

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)

def mark(root, phase, **fields):
    write(root / "phase.json", dict(phase=phase, unix=time.time(), pid=os.getpid(), **fields))
    print("PHASE", phase, flush=True)

def capture(root, name, args, seconds=10):
    try:
        p = subprocess.run(args, capture_output=True, timeout=seconds)
        data = (p.stdout + b"\nSTDERR\n" + p.stderr)[-2*1024*1024:]
        (root / (name + ".log")).write_bytes(data)
        return dict(code=p.returncode, log=name+".log")
    except Exception as e:
        write(root / (name + ".json"), dict(error=str(e)))
        return dict(error=type(e).__name__)

def graph_data(modulo):
    generators = []
    for value in (1, modulo-1):
        for row in range(3):
            m = [int(i//4 == i%4) for i in range(16)]
            m[row*4+row+1] = value
            generators.append(dict(matrix=m, modulo=modulo))
    return dict(schema=2, name="unitriangular_fundamental_roots-4%"+str(modulo),
                action=dict(kind="matrix", rows=4, cols=4, generators=generators),
                start=[int(i//4 == i%4) for i in range(16)],
                generator_names=["0","2","4","1","3","5"], expected_max_unique_states=None)

def oracle(modulo):
    # Independent complete right-multiplication BFS for the 729-element UT(4,3).
    start = tuple(graph_data(modulo)["start"])
    seen = {start}
    frontier = {start}
    sizes = []
    while frontier:
        sizes.append(len(frontier))
        children = set()
        for state in frontier:
            for value in (1, modulo-1):
                for col in range(1,4):
                    x = list(state)
                    for row in range(4):
                        x[4*row+col] = (x[4*row+col] + value*state[4*row+col-1]) % modulo
                    child = tuple(x)
                    if child not in seen:
                        children.add(child)
        seen.update(children)
        frontier = children
    return sizes

def health(root):
    mark(root, "driver-and-eight-device-health")
    smi = capture(root, "nvidia-smi", ["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"])
    smi_text=(root/"nvidia-smi.log").read_text(errors="replace") if (root/"nvidia-smi.log").exists() else ""
    if smi.get("code") != 0 or "Unknown Error" in smi_text:
        raise RuntimeError("NVIDIA_SMI_UNHEALTHY; no further GPU tests")
    driver = C.CDLL("libcuda.so.1")
    def ck(code, name):
        if code: raise RuntimeError(name + " returned " + str(code))
    ck(driver.cuInit(0), "cuInit")
    count = C.c_int()
    ck(driver.cuDeviceGetCount(C.byref(count)), "cuDeviceGetCount")
    if count.value != 8:
        raise RuntimeError("EXACTLY_EIGHT_VISIBLE_GPUS_REQUIRED: got "+str(count.value))
    for index in range(8):
        device = C.c_int()
        ck(driver.cuDeviceGet(C.byref(device), index), "cuDeviceGet")
        ctx = C.c_void_p()
        ck(driver.cuDevicePrimaryCtxRetain(C.byref(ctx), device), "cuDevicePrimaryCtxRetain")
        try:
            ck(driver.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
            free, total = C.c_size_t(), C.c_size_t()
            ck(driver.cuMemGetInfo_v2(C.byref(free), C.byref(total)), "cuMemGetInfo")
        finally:
            driver.cuCtxSetCurrent(C.c_void_p())
            driver.cuDevicePrimaryCtxRelease_v2(device)
    return dict(status="PASS", visible_devices=count.value)

def stage_worker(root, spec):
    mode = spec.get("mode", "baseline")
    if mode == "no-gdr": os.environ["NCCL_NET_GDR_LEVEL"] = "LOC"
    elif mode == "no-ib": os.environ["NCCL_IB_DISABLE"] = "1"
    write(root/"environment.json", {k:os.environ.get(k) for k in SAFE_ENV})
    kind = spec["kind"]
    if kind == "health":
        return health(root)
    if kind == "prepare":
        capture(root, "topology", ["nvidia-smi", "topo", "-m"])
        capture(root, "gpu-info", ["nvidia-smi", "-q"])
        from multigpubfs.native_distribution import native_runtime, runtime_identity
        native, env = native_runtime()
        identity=runtime_identity(native, env)
        if identity.get("native_sha256") != "10e423aee53f450d5a14f064db0cb1c7784e8080e7b94045adbad378ad304fce":
            raise RuntimeError("PINNED_NATIVE_IDENTITY_MISMATCH")
        capture(root, "dependencies", ["ldd", native])
        return dict(status="PASS", identity=runtime_identity(native, env),
                    package=importlib.metadata.version("multigpubfs"))
    from multigpubfs import GraphDefinition, run_graph, persistent_graph_workers
    from multigpubfs.native_distribution import native_runtime
    if importlib.metadata.version("multigpubfs") != "0.2.0.dev29":
        raise RuntimeError("DEV29_REQUIRED")
    native, env = native_runtime()
    env.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="ALL",
               NCCL_DEBUG_FILE=str(root/"nccl-%h-%p.log"), MGBFS_TRACE_NATIVE_CALLS="1",
               MGBFS_TUNING_EVIDENCE_ROOT=str(root/"tuning-evidence"))
    os.environ["MGBFS_PROFILE_CACHE"] = str(root/"profile-cache")
    env["MGBFS_PROFILE_CACHE"] = str(root/"profile-cache")
    (root/"session-live").mkdir(exist_ok=True)
    os.environ["MGBFS_GENERIC_SESSION_LOG_ROOT"] = str(root/"session-live")
    env["MGBFS_GENERIC_SESSION_LOG_ROOT"] = str(root/"session-live")
    if kind == "nccl":
        # Existing component starts all requested ranks, verifies the all-reduce value.
        os.environ.update({k:v for k,v in env.items() if k.startswith("NCCL_")})
        from multigpubfs.diagnostics import _component
        return dict(status="PASS", result=_component("nccl_exchange", root, list(range(8)), None, 35))
    reports = []
    graph = GraphDefinition.from_dict(graph_data(3 if kind == "oracle" else 47))
    write(root/"graph.json", graph.to_dict() if hasattr(graph,"to_dict") else graph_data(3 if kind=="oracle" else 47))
    def execute(pool):
        sequence = spec.get("shards", [1])
        for index, shards in enumerate(sequence):
            mark(root, "bfs", step=index, shards=shards, mode=mode)
            kwargs = dict(devices=list(range(8)), max_seconds=spec.get("seconds",30), _native_env=env)
            if kind == "production":
                kwargs.update(autotune=True)
            else:
                kwargs.update(autotune=False, shards=shards, backend="generic",
                              transport="full", candidate_order="none",
                              capacity=3080192, _batch=174762)
                env["MGBFS_GENERIC_HISTORY"]="hash"
                env["MGBFS_GENERIC_GENERATOR"]="cuda"
                if kind != "oracle": kwargs["_profile_layers"]=4
            started = time.monotonic()
            report = run_graph(graph, root/("run-"+str(index)), **kwargs)
            layers = report.get("layer_sizes", [])
            world = report.get("world", report.get("plan",{}).get("world"))
            if world != 8: raise RuntimeError("WORLD_MISMATCH")
            if kind == "oracle":
                if report.get("status") != "COMPLETE" or layers != oracle(3):
                    raise RuntimeError("COMPLETE_CPU_ORACLE_MISMATCH")
            elif layers[:5] != EXPECTED:
                raise RuntimeError("UT47_PREFIX_MISMATCH")
            if kind not in ("production","oracle") and (report.get("status")!="INCOMPLETE" or report.get("reason")!="PROFILE_LAYER_LIMIT"):
                raise RuntimeError("UNEXPECTED_FIXED_PROBE_STOP")
            if report.get("status")=="COMPLETE" and sum(layers) != (3 if kind=="oracle" else 47)**6:
                raise RuntimeError("COMPLETE_CARDINALITY_MISMATCH")
            item=dict(step=index, shards=shards, status=report.get("status"), reason=report.get("reason"),
                      layers=layers, seconds=time.monotonic()-started, plan=report.get("plan"),
                      bfs_seconds=report.get("bfs_seconds"), tuning=report.get("autotune"),
                      pids=report.get("rank_worker_pids"))
            reports.append(item)
            write(root/"completed.json", reports)
    if spec.get("persistent", False) or kind=="production":
        with persistent_graph_workers() as pool:
            try:
                execute(pool)
            finally:
                # Snapshot before releasing temporary session resources; no job files or state payload.
                shutil.copytree(pool.root, root/"session-evidence", dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("job-*"))
                pool.close_workers()
                shutil.copytree(pool.root, root/"session-evidence", dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("job-*"))
    else:
        execute(None)
    return dict(status="PASS", reports=reports)

def walk(value):
    if isinstance(value, dict):
        yield value
        for v in value.values(): yield from walk(v)
    elif isinstance(value,list):
        for v in value: yield from walk(v)

def production_verdict(result):
    reports=result.get("reports",[])
    if not reports: return "NOT_VERIFIED"
    r=reports[-1]; tuning=r.get("tuning") or {}
    if not (tuning.get("measured") is True or (tuning.get("global_profile") or {}).get("measured") is True): return "AUTOTUNE_NOT_MEASURED"
    if any(d.get("candidate_failures") for d in walk(tuning)): return "AUTOTUNE_CANDIDATE_FAILURE"
    layers=r.get("layers",[])
    if layers[:5]!=EXPECTED: return "UT47_PREFIX_NOT_VERIFIED"
    if r["status"]=="COMPLETE":
        return "FULL_GRAPH_COMPLETE" if sum(layers)==47**6 else "COMPLETE_CARDINALITY_MISMATCH"
    if r["status"]!="INCOMPLETE": return "UNEXPECTED_BFS_STATUS"
    return "AUTOTUNE_AND_BOUNDED_BFS_VERIFIED"

def snapshot(root, pid):
    captured=[]
    for item in Path("/proc").iterdir():
        if not item.name.isdigit(): continue
        try:
            stat=(item/"stat").read_text().rsplit(")",1)[1].split()
            if int(stat[2]) != pid: continue
            entry=dict(pid=int(item.name))
            for name in ("status","wchan","stack","limits","maps"):
                try: entry[name]=(item/name).read_text()[:16000]
                except OSError as e: entry[name]=type(e).__name__
            entry["threads"]=[]
            for task in list((item/"task").iterdir())[:64]:
                try: entry["threads"].append(dict(tid=task.name,wchan=(task/"wchan").read_text()))
                except OSError: pass
            entry["fds"]={}
            for fd in list((item/"fd").iterdir())[:64]:
                try: entry["fds"][fd.name]=os.readlink(fd)
                except OSError: pass
            captured.append(entry)
        except (OSError,ValueError,IndexError): pass
    write(root/"process-snapshot.json", captured)
    capture(root, "kernel-log", ["dmesg","--ctime","--level=err,warn"], 4)
    capture(root, "nvidia-after-failure", ["nvidia-smi","-q"], 8)

def stop_group(p):
    for sig in (signal.SIGTERM,signal.SIGKILL):
        try: os.killpg(p.pid,sig)
        except ProcessLookupError: pass
        try: p.wait(timeout=2)
        except subprocess.TimeoutExpired: pass
    # Wait for asynchronous SIGKILL completion; persistent D-state is unsafe.
    until=time.monotonic()+15
    while True:
        alive=False
        for item in Path("/proc").iterdir():
            if item.name.isdigit():
                try:
                    fields=(item/"stat").read_text().rsplit(")",1)[1].split()
                    if int(fields[2])==p.pid and fields[0]!="Z": alive=True
                except (OSError,ValueError,IndexError): pass
        if not alive: return True
        if time.monotonic()>=until: return False
        time.sleep(.05)

def run_stage(base, name, spec, timeout, deadline, tracer=None):
    root=base/name;root.mkdir()
    write(root/"spec.json", spec)
    env=dict(os.environ)
    env["WORLD_SIZE"]="1"
    env.pop("MGBFS_EXECUTABLE",None)
    for key in list(env):
        if key.startswith("MGBFS_TEST_"): env.pop(key)
    if tracer and spec["kind"] not in ("health","prepare","production"):
        env["LD_PRELOAD"]=str(tracer)+(":"+env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
        env["MGBFS_TRACE_DIR"]=str(root)
    allowed=min(timeout, max(0,deadline-time.monotonic()))
    if allowed<1: return dict(name=name,status="SKIPPED_BUDGET",spec=spec)
    start=time.monotonic()
    print("START",name,"limit",round(allowed),flush=True)
    p=None
    try:
        with (root/"worker.log").open("wb") as out:
            p=subprocess.Popen([sys.executable,"-u",str(Path(__file__).resolve()),
                                "--child",str(root)],stdout=out,stderr=subprocess.STDOUT,
                               env=env,start_new_session=True)
            try:
                code=p.wait(timeout=allowed)
                status="PASS" if code==0 else "FAIL"
            except subprocess.TimeoutExpired:
                code=124;status="TIMEOUT"
            if status!="PASS": snapshot(root,p.pid)
    finally:
        clean=stop_group(p) if p else True
    result=json.loads((root/"result.json").read_text()) if (root/"result.json").exists() else {}
    entry=dict(name=name,status=status,code=code,seconds=time.monotonic()-start,
               cleanup_complete=clean,spec=spec,result=result)
    entry["trace_status"]=("OBSERVED" if list(root.glob("api-*.ndjson")) else "NOT_OBSERVED") if tracer and spec["kind"] not in ("health","prepare","production") else "NOT_REQUESTED"
    if status!="PASS": entry["failure_details"]=failure_details(root)
    write(root/"stage.json",entry)
    print(status,name,round(entry["seconds"],2),flush=True)
    return entry

def bundle(root):
    # Metadata/logs only. Never include state arrays, job files, full environment, or cache payloads.
    paths=[p for p in root.rglob("*") if p.is_file() and not p.is_symlink()
           and p.suffix in (".json",".log",".txt",".py",".c",".sh",".ndjson")
           and not p.name.startswith("job-") and "state" not in p.name.lower()]
    manifest={}
    omitted=[]
    archive=root.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):
            rel=p.relative_to(root).as_posix()
            if p.stat().st_size>16*1024*1024:
                omitted.append(dict(path=rel,reason="log over 16MiB; tail archived",bytes=p.stat().st_size))
                with p.open("rb") as f:
                    f.seek(-16*1024*1024,2);data=f.read()
                rel += ".tail.txt"
            else: data=p.read_bytes()
            manifest[rel]=hashlib.sha256(data).hexdigest()
            z.writestr(root.name+"/"+rel,data)
        z.writestr(root.name+"/manifest.json",json.dumps(manifest,indent=2))
        z.writestr(root.name+"/archive-omissions.json",json.dumps(omitted,indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip(): raise RuntimeError("ARCHIVE_CRC")
        for name,digest in manifest.items():
            if hashlib.sha256(z.read(root.name+"/"+name)).hexdigest()!=digest: raise RuntimeError("ARCHIVE_SHA")
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    print("ARCHIVE:",archive,flush=True);print("ARCHIVE_SHA256:",digest,flush=True)
    return archive

def failure_details(root):
    details=[]
    for path in root.rglob("*phase*.json"):
        try:
            d=json.loads(path.read_text())
            if "native_call" in d:
                details.append(dict(file=str(path.relative_to(root)), updated_unix=d.get("updated_unix",0), **{k:d.get(k) for k in (
                    "rank","layer","last_completed_layer","native_call","native_call_state",
                    "native_error","native_call_line","native_call_sequence")}))
        except (OSError,ValueError): pass
    api=[]
    for path in root.glob("api-*.ndjson"):
        if ".previous." in path.name: continue
        try:
            lines=path.read_text(errors="replace").splitlines()
            events=[json.loads(l) for l in lines]
            groups=[e for e in events if e.get("api") in ("ncclSendContract","ncclRecvContract")]
            last_group=max((e.get("group",0) for e in groups),default=None)
            api.append(dict(file=path.name,last_calls=events[-12:],
                            last_group_contract=[e for e in groups if e.get("group")==last_group]))
        except (OSError,ValueError): pass
    latest={}
    for d in sorted(details,key=lambda x:x.get("updated_unix",0)):
        latest[d.get("rank")]=d
    return dict(native=list(latest.values()),phase_file_count=len(details),api=api)

def analyze(stages, stopped):
    recommendations=[]
    for mode in MODES:
        entries=[s for s in stages if s.get("spec",{}).get("mode")==mode]
        production=next((s for s in entries if s["spec"]["kind"]=="production" and s["status"]=="PASS"),None)
        oracle_ok=any(s["spec"]["kind"]=="oracle" and s["status"]=="PASS" for s in entries)
        matrix_ok=all(any(s["name"]==mode+"-"+label and s["status"]=="PASS" for s in entries)
                      for label in ("collective","fresh-256","persistent-transitions"))
        if production and oracle_ok and matrix_ok:
            verdict=production_verdict(production["result"])
            if verdict in ("FULL_GRAPH_COMPLETE","AUTOTUNE_AND_BOUNDED_BFS_VERIFIED"):
                recommendations.append(dict(mode=mode,verdict=verdict,
                    limitations="Bounded UT47 run; not proof of full graph or all graph families"))
    observations=[]
    lookup={s["name"]:s for s in stages}
    for mode in MODES:
        fresh=lookup.get(mode+"-fresh-256",{})
        resident=lookup.get(mode+"-persistent-transitions",{})
        if fresh.get("status")=="PASS" and resident.get("status") in ("FAIL","TIMEOUT"):
            observations.append(mode+": isolated 256 passed, persistent transition failed; reuse/order is implicated, root cause unproven.")
        if fresh.get("status") in ("FAIL","TIMEOUT"):
            observations.append(mode+": isolated 256 failed; persistence alone cannot explain the failure.")
    for stage in stages:
        api=stage.get("failure_details",{}).get("api",[])
        if any(c.get("api")=="ncclCommGetAsyncError" and c.get("datatype")==7 for a in api for c in a.get("last_calls",[])):
            observations.append(stage["name"]+": last observed NCCL asynchronous state was InProgress (7); this is a pending operation, not itself an error code.")
    if stopped: observations.append("Preparation, GPU health or cleanup failed; subsequent tests were suppressed. Inspect health logs and kernel-log access results.")
    complete=all(s["status"] not in ("SKIPPED_BUDGET","SKIPPED_UNHEALTHY") for s in stages)
    return dict(status="HEALTH_ABORT" if stopped else ("FINISHED" if complete else "BUDGET_EXHAUSTED"),
                observations=observations,
                root_cause="UNRESOLVED; compare observed calls, health and isolated contrasts",
                recommended=recommendations, stages=stages,
                scope="Eight GPUs concurrently; no single-GPU catalog. No driver reset or automatic BFS restart.")

def main(args):
    root=Path(tempfile.mkdtemp(prefix="ut47-deep-"+time.strftime("%Y%m%d-%H%M%S")+"-",dir=".")).resolve()
    shutil.copyfile(__file__,root/"runner.py")
    deadline=time.monotonic()+args.budget-35
    stages=[];stopped=False
    write(root/"host.json",dict(host=platform.node(),python=sys.version,affinity=sorted(os.sched_getaffinity(0)),
                               environment={k:os.environ.get(k) for k in SAFE_ENV},budget=args.budget))
    print("OUTPUT:",root,flush=True)
    try:
        with (root/"install.log").open("wb") as out:
            subprocess.run([sys.executable,"-m","pip","install","--no-deps","--disable-pip-version-check",
                            "--upgrade","--force-reinstall",WHEEL],stdout=out,stderr=subprocess.STDOUT,check=True,
                           timeout=min(180,max(1,deadline-time.monotonic())))
        tracer=build_tracer(root)
        def run(name,spec,limit):
            nonlocal stopped
            if stopped:
                entry=dict(name=name,status="SKIPPED_UNHEALTHY",spec=spec);stages.append(entry);return entry
            result=run_stage(root,name,spec,limit,deadline,tracer)
            stages.append(result)
            if not result.get("cleanup_complete",True): stopped=True
            write(root/"progress.json",analyze(stages,stopped))
            return result
        h=run("initial-health",dict(kind="health"),30)
        if h["status"]!="PASS": stopped=True
        if not stopped:
            prepared=run("prepare",dict(kind="prepare"),120)
            if prepared["status"]!="PASS": stopped=True
        # Low-risk controls for every transport first, then the known failing geometries.
        tests=[("collective",dict(kind="nccl"),50),
               ("oracle",dict(kind="oracle",shards=[1]),60),
               ("fresh-256",dict(kind="fixed",shards=[256]),70),
               ("persistent-transitions",dict(kind="fixed",shards=[1,128,256,1],persistent=True),130)]
        for label,template,limit in tests:
            for mode in MODES:
                spec=dict(template,mode=mode,seconds=35)
                run(mode+"-"+label,spec,limit)
                h=run(mode+"-"+label+"-health",dict(kind="health"),25)
                if h["status"] not in ("PASS","SKIPPED_BUDGET"): stopped=True
        # Production runs use the real auto tuner and fresh profile caches, no fixed geometry.
        for mode in MODES:
            run(mode+"-production",dict(kind="production",mode=mode,seconds=180),260)
            h=run(mode+"-production-health",dict(kind="health"),25)
            if h["status"] not in ("PASS","SKIPPED_BUDGET"): stopped=True
    except KeyboardInterrupt:
        stopped=True
        (root/"supervisor-error.log").write_text(traceback.format_exc())
    except Exception:
        stopped=True
        (root/"supervisor-error.log").write_text(traceback.format_exc())
    finally:
        report=analyze(stages,stopped)
        write(root/"report.json",report)
        lines=["UT47 eight-GPU diagnostic", "Status: "+report["status"],
               "Root cause: "+report["root_cause"],""]+report["observations"]
        for stage in stages:
            lines.append(stage["name"]+": "+stage["status"])
            detail=stage.get("failure_details",{})
            for d in detail.get("native",[])[:8]:
                lines.append("  rank "+str(d["rank"])+" layer "+str(d["layer"])+": "+str(d["native_call"])+" "+str(d["native_call_state"]))
            for d in detail.get("api",[])[:8]:
                calls=d.get("last_calls",[])
                if calls:
                    c=calls[-1]
                    lines.append("  "+d["file"]+": "+c["api"]+" "+c["phase"]+" rc="+str(c["rc"])+" async/count="+str(c.get("datatype"))+"/"+str(c.get("count")))
        lines.append("Diagnostic NCCL tracing changes timing; production autotuning runs without the interposer.")

        if report["recommended"] and not stopped:
            best=report["recommended"][0]
            overrides={"baseline":{}, "no-gdr":{"NCCL_NET_GDR_LEVEL":"LOC"}, "no-ib":{"NCCL_IB_DISABLE":"1"}}[best["mode"]]
            command=" ".join(k+"="+shlex.quote(v) for k,v in overrides.items())
            command=(command+" " if command else "")+"python your_bfs.py"
            lines+=["","Verified configuration: "+best["mode"],"Launch in the same allocated job: "+command,
                    "Preserve CUDA_VISIBLE_DEVICES and the recorded environment. Different nodes require validation."]
            (root/"launch.env").write_text("\n".join("export "+k+"="+shlex.quote(v) for k,v in overrides.items())+"\n")
            # Concrete executable example, with an explicit finite work limit.
            (root/"launch-verified.sh").write_text("#!/bin/sh\nset -eu\nunset MGBFS_EXECUTABLE\nexport WORLD_SIZE=1\n"+ "\n".join("export "+k+"="+shlex.quote(v) for k,v in overrides.items())+
                "\nexec python -u - <<'PY'\nfrom multigpubfs import GraphDefinition, run_graph, persistent_graph_workers\nimport json,tempfile\nfrom pathlib import Path\ng=GraphDefinition.from_dict("+repr(graph_data(47))+")\nout=Path(tempfile.mkdtemp(prefix='ut47-run-'))/'result'\nwith persistent_graph_workers():\n r=run_graph(g,out,devices=list(range(8)),autotune=True,max_seconds=900)\nprint(json.dumps(r,indent=2))\nprint('RESULT_DIRECTORY',out)\nPY\n")
            lines+=["Ready-to-run example: sh "+str(root/"launch-verified.sh")]
        else: lines+=["","No accepted launch configuration. Do not treat skipped tests as passing."]
        (root/"REPORT.txt").write_text("\n".join(lines)+"\n")
        print("\n".join(lines),flush=True)
        bundle(root)
    return 0 if report["recommended"] and not stopped else 1

# Tracer source is supplied below. It wraps exported NCCL boundaries only, not CUDA kernels.
TRACER_SOURCE = "#define _GNU_SOURCE\n#include <dlfcn.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <unistd.h>\n#include <sys/syscall.h>\n#include <pthread.h>\n#include <time.h>\n#include <stdint.h>\n#include <stdatomic.h>\n#include <string.h>\nstatic pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;\nstatic FILE *out;\nstatic size_t bytes;\nstatic unsigned long seq;\nstatic _Thread_local int depth;\nstatic _Thread_local unsigned long group_id;\nstatic _Thread_local unsigned long long signature;\nstruct summary {size_t calls,total; unsigned long long hash; void *comm;};\nstatic _Thread_local struct summary ledger[2][256];\nstatic void account(int direction,int peer,size_t n,int type,void *comm) {\n if(peer<0||peer>=256) return;\n struct summary *v=&ledger[direction][peer];\n v->calls++;v->total+=n;v->comm=comm;\n v->hash=((v->hash ^ (unsigned long long)n)*1099511628211ULL ^ (unsigned int)type)*1099511628211ULL;\n}\n__asm__(\".symver dlsym,dlsym@GLIBC_2.2.5\");\nstatic int rank_number(const char *s){if(!s)return -1;int n=0;while(*s>='0'&&*s<='9'){n=n*10+(*s++-'0');}return n;}\nstatic void event(const char *api,const char *phase,int rc,size_t count,int type,int peer,void *comm,void *stream) {\n const char *dir=getenv(\"MGBFS_TRACE_DIR\"); if(!dir) return;\n char file[4096],old[4096],line[1024];\n snprintf(file,sizeof(file),\"%s/api-%d.ndjson\",dir,getpid());\n snprintf(old,sizeof(old),\"%s/api-%d.previous.ndjson\",dir,getpid());\n pthread_mutex_lock(&lock);\n if(!out) out=fopen(file,\"a\");\n if(out) {\n  if(bytes>4*1024*1024) {fclose(out);rename(file,old);out=fopen(file,\"w\");bytes=0;}\n  struct timespec t;clock_gettime(CLOCK_REALTIME,&t);\n  const char *rank=getenv(\"RANK\");\n  int n=snprintf(line,sizeof(line),\"{\\\"seq\\\":%lu,\\\"time\\\":%.9f,\\\"pid\\\":%d,\\\"tid\\\":%ld,\\\"rank\\\":%d,\\\"api\\\":\\\"%s\\\",\\\"phase\\\":\\\"%s\\\",\\\"rc\\\":%d,\\\"group_depth\\\":%d,\\\"group\\\":%lu,\\\"signature\\\":\\\"%016llx\\\",\\\"count\\\":%zu,\\\"datatype\\\":%d,\\\"peer\\\":%d,\\\"comm\\\":\\\"%p\\\",\\\"stream\\\":\\\"%p\\\"}\\n\",\n   ++seq,t.tv_sec+t.tv_nsec/1e9,getpid(),syscall(SYS_gettid),rank_number(rank),api,phase,rc,depth,group_id,signature,count,type,peer,comm,stream);\n  if(n>0 && n<(int)sizeof(line)) {fwrite(line,1,n,out);fflush(out);bytes+=n;}\n }\n pthread_mutex_unlock(&lock);\n}\n#define LOAD(name,T) T f=(T)dlsym(RTLD_NEXT,#name); if(!f){event(#name,\"symbol_missing\",-1,0,0,0,0,0);return 2;}\nint ncclGroupStart(void) {typedef int(*F)(void);LOAD(ncclGroupStart,F);if(depth==0){group_id++;memset(ledger,0,sizeof(ledger));}event(\"ncclGroupStart\",\"enter\",0,0,0,0,0,0);int r=f();if(!r)depth++;event(\"ncclGroupStart\",\"return\",r,0,0,0,0,0);return r;}\nint ncclGroupEnd(void) {typedef int(*F)(void);LOAD(ncclGroupEnd,F);if(depth==1){for(int d=0;d<2;d++)for(int peer=0;peer<256;peer++){struct summary *v=&ledger[d][peer];if(v->calls){signature=v->hash;event(d?\"ncclRecvContract\":\"ncclSendContract\",\"before-group-end\",0,v->total,(int)v->calls,peer,v->comm,0);}}signature=0;}event(\"ncclGroupEnd\",\"enter\",0,0,0,0,0,0);int r=f();event(\"ncclGroupEnd\",\"return\",r,0,0,0,0,0);if(depth>0)depth--;return r;}\nint ncclSend(const void *b,size_t n,int t,int peer,void *c,void *s) {typedef int(*F)(const void*,size_t,int,int,void*,void*);LOAD(ncclSend,F);account(0,peer,n,t,c);event(\"ncclSend\",\"enter\",0,n,t,peer,c,s);int r=f(b,n,t,peer,c,s);event(\"ncclSend\",\"return\",r,n,t,peer,c,s);return r;}\nint ncclRecv(void *b,size_t n,int t,int peer,void *c,void *s) {typedef int(*F)(void*,size_t,int,int,void*,void*);LOAD(ncclRecv,F);account(1,peer,n,t,c);event(\"ncclRecv\",\"enter\",0,n,t,peer,c,s);int r=f(b,n,t,peer,c,s);event(\"ncclRecv\",\"return\",r,n,t,peer,c,s);return r;}\nint ncclAllReduce(const void *a,void *b,size_t n,int t,int op,void *c,void *s) {typedef int(*F)(const void*,void*,size_t,int,int,void*,void*);LOAD(ncclAllReduce,F);event(\"ncclAllReduce\",\"enter\",0,n,t,op,c,s);int r=f(a,b,n,t,op,c,s);event(\"ncclAllReduce\",\"return\",r,n,t,op,c,s);return r;}\nint ncclAllGather(const void *a,void *b,size_t n,int t,void *c,void *s) {typedef int(*F)(const void*,void*,size_t,int,void*,void*);LOAD(ncclAllGather,F);event(\"ncclAllGather\",\"enter\",0,n,t,0,c,s);int r=f(a,b,n,t,c,s);event(\"ncclAllGather\",\"return\",r,n,t,0,c,s);return r;}\nint ncclCommGetAsyncError(void *c,int *e) {typedef int(*F)(void*,int*);LOAD(ncclCommGetAsyncError,F);int r=f(c,e);static _Thread_local int last=-999;int value=r?r:*e;if(value!=last){event(\"ncclCommGetAsyncError\",\"return\",r,0,*e,0,c,0);last=value;}return r;}\n"
TRACER_BINARY = "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AAAAAHhAAAAAAAAABwAAABQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA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AgAAAEFVi1wkXFP/dCRY/3QkWIu0JJQAAABWvgAEAABBVFC4AQAAAEiJfCR46Ab8//9Ig8RwSIt8JAiNUP+B+v4DAAAPhp4AAABIie/oZvz//0iLhCRYJAAAZEgrBCUoAAAAD4W4AAAASIHEaCQAAFtdQVxBXUFeQV/DDx9EAABMiedIjTWZCgAA6En8//9IiQUiKwAASInHSIXAdKzpTf7//w8fRAAA6Lv7//9Mie5MiefoMPz//0iNNWYKAABMiefoEfz//0jHBd4qAAAAAAAASIkF3yoAAOkj/v//Zi4PH4QAAAAAAEhj2EiLDcYqAAC+AQAAAEiJ2ugJ/P//SIs9sioAAOi9+///SAEdnioAAOkz////vv/////pXf7//+hS+///ZpDzDx76VUiNLfgJAABIx8f/////U0iJ7kiD7AjosPv//0iFwA+EpwAAAEiJw0iNPT0pAADoOPv//4u4IEAAAIX/dG5qADHJSI011QkAAEUxyWoARTHAMdJIie/oYfz////TWV6Jw4XAdC9qAInaRTHJRTHAagAxyUiNNaoJAABIie/oOvz//1haSIPECInYW13DDx+AAAAAAEiNPdEoAADozPr//4OAIEAAAAHrvA8fAEiNuBAAAAC6AEAAADH2SIOAGEAAAAHolfr//+ly////agBFMclFMcAxyWoAuv////9Iie+7AgAAAEiNNSMJAADoy/v//0FYQVnrjQ8fRAAA8w8e+kFXSMfH/////0FWQVVMjS0XCQAAQVRMie5VU0iD7CjoqPr//0iFwA+EUwEAAEmJxEiNPTUoAADoMPr//4O4IEAAAAF0c2oARTHJRTHAMclqADHSSI01xAgAAEyJ7+ha+///Qf/UagBFMclFMcBqAInCMclIjTWqCAAATInvicPoOPv//0iNPeEnAADo3Pn//0iDxCCLkCBAAACF0n4Jg+oBiZAgQAAASIPEKInYW11BXEFdQV5BX8NMiWQkGEgFEAAAADHSSI0tbAgAAIlUJAyF0kyNPYEIAABIicNIiUQkEEiNPWEIAABMD0X/RTHkDx9EAABMizNNhfZ0NUiNPWknAADoZPn//0iLUxBIi0sIRYnhRYnwSInuTIn/SImQEEAAADHSagD/cxjojfr//1haQYPEAUiDwyBBgfwAAQAAdbJIi0QkEItUJAxIBQAgAACF0nUKugEA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AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGfz//wAAAAAAAAAAAAAAAPASAAAAAAAAsBIAAAAAAAABAAAAAAAAAF8BAAAAAAAAAQAAAAAAAABpAQAAAAAAAAwAAAAAAAAAABAAAAAAAAANAAAAAAAAAKQeAAAAAAAAGQAAAAAAAADQPQAAAAAAABsAAAAAAAAACAAAAAAAAAAaAAAAAAAAANg9AAAAAAAAHAAAAAAAAAAIAAAAAAAAAPX+/28AAAAAKAMAAAAAAAAFAAAAAAAAAAgGAAAAAAAABgAAAAAAAABoAwAAAAAAAAoAAAAAAAAAtQEAAAAAAAALAAAAAAAAABgAAAAAAAAAAwAAAAAAAADoPwAAAAAAAAIAAAAAAAAAgAEAAAAAAAAUAAAAAAAAAAcAAAAAAAAAFwAAAAAAAAAoCQAAAAAAAAcAAAAAAAAAaAgAAAAAAAAIAAAAAAAAAMAAAAAAAAAACQAAAAAAAAAYAAAAAAAAAP7//28AAAAA+AcAAAAAAAD///9vAAAAAAIAAAAAAAAA8P//bwAAAAC+BwAAAAAAAPn//28AAAAAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4D0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAMBAAAAAAAABAEAAAAAAAAFAQAAAAAAAAYBAAAAAAAABwEAAAAAAAAIAQAAAAAAAAkBAAAAAAAACgEAAAAAAAALAQAAAAAAAAwBAAAAAAAADQEAAAAAAAAOAQAAAAAAAA8BAAAAAAAAAAEQAAAAAAABARAAAAAAAAIBEAAAAAAACAQAAAAAAAAEdDQzogKFVidW50dSAxMy4zLjAtNnVidW50dTJ+MjQuMDQuMSkgMTMuMy4wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAEAPH/AAAAAAAAAAAAAAAAAAAAAAwAAAACAA4AQBIAAAAAAAAAAAAAAAAAAA4AAAACAA4AcBIAAAAAAAAAAAAAAAAAACEAAAACAA4AsBIAAAAAAAAAAAAAAAAAADcAAAABABsAoEAAAAAAAAABAAAAAAAAAEMAAAABABYA2D0AAAAAAAAAAAAAAAAAAGoAAAACAA4A8BIAAAAAAAAAAAAAAAAAAHYAAAABABUA0D0AAAAAAAAAAAAAAAAAAJUAAAAEAPH/AAAAAAAAAAAAAAAAAAAAAJ0AAAACAA4AABMAAAAAAAA+AwAAAAAAAKMAAAABABsA4EAAAAAAAAAoAAAAAAAAAKgAAAABABsA0EAAAAAAAAAIAAAAAAAAAKwAAAABABsAyEAAAAAAAAAIAAAAAAAAALIAAAAGABQAEEAAAAAAAAAIAAAAAAAAALwAAAAGABQAGEAAAAAAAAAIAAAAAAAAAMUAAAAGABQAIEAAAAAAAAAEAAAAAAAAAMsAAAABABsAwEAAAAAAAAAIAAAAAAAAAM8AAAAGABQAEAAAAAAAAAAAQAAAAAAAANYAAAAGABMAAAAAAAAAAAAEAAAAAAAAAAEAAAAEAPH/AAAAAAAAAAAAAAAAAAAAAN0AAAABABIAICYAAAAAAAAAAAAAAAAAAAAAAAAEAPH/AAAAAAAAAAAAAAAAAAAAAOsAAAACAA8ApB4AAAAAAAAAAAAAAAAAAPEAAAABABoAgEAAAAAAAAAAAAAAAAAAAP4AAAABABcA4D0AAAAAAAAAAAAAAAAAAAcBAAAAABEAyCEAAAAAAAAAAAAAAAAAABoBAAABABoAiEAAAAAAAAAAAAAAAAAAACYBAAABABkA6D8AAAAAAAAAAAAAAAAAADwBAAACAAoAABAAAAAAAAAAAAAAAAAAAEIBAAASAA4A4BwAAAAAAADiAAAAAAAAAFABAAASAAAAAAAAAAAAAAAAAAAAAAAAAGMBAAASAA4A0BsAAAAAAAAGAQAAAAAAAHEBAAASAAAAAAAAAAAAAAAAAAAAAAAAAIwBAAASAA4AQBcAAAAAAACwAQAAAAAAAJkBAAAgAAAAAAAAAAAAAAAAAAAAAAAAALUBAAASAAAAAAAAAAAAAAAAAAAAAAAAAM4BAAASAAAAAAAAAAAAAAAAAAAAAAAAAOEBAAASAAAAAAAAAAAAAAAAAAAAAAAAAPQBAAASAAAAAAAAAAAAAAAAAAAAAAAAAA8CAAASAAAAAAAAAAAAAAAAAAAAAAAAACICAAASAAAAAAAAAAAAAAAAAAAAAAAAADsCAAASAA4A0B0AAAAAAADRAAAAAAAAAFECAAASAAAAAAAAAAAAAAAAAAAAAAAAAGUCAAAgAAAAAAAAAAAAAAAAAAAAAAAAAHQCAAASAAAAAAAAAAAAAAAAAAAAAAAAAJUCAAASAAAAAAAAAAAAAAAAAAAAAAAAAKgCAAASAA4AUBoAAAAAAAB5AQAAAAAAALECAAASAA4A8BgAAAAAAABWAQAAAAAAALoCAAASAAAAAAAAAAAAAAAAAAAAAAAAAMwCAAASAAAAAAAAAAAAAAAAAAAAAAAAAN8CAAASAAAAAAAAAAAAAAAAAAAAAAAAAPECAAASAAAAAAAAAAAAAAAAAAAAAAAAAAQDAAAgAAAAAAAAAAAAAAAAAAAAAAAAAB4DAAASAA4AQBYAAAAAAAD7AAAAAAAAAC0DAAAiAAAAAAAAAAAAAAAAAAAAAAAAAEgDAAASAAAAAAAAAAAAAAAAAAAAAAAAAABjcnRzdHVmZi5jAGRlcmVnaXN0ZXJfdG1fY2xvbmVzAF9fZG9fZ2xvYmFsX2R0b3JzX2F1eABjb21wbGV0ZWQuMABfX2RvX2dsb2JhbF9kdG9yc19hdXhfZmluaV9hcnJheV9lbnRyeQBmcmFtZV9kdW1teQBfX2ZyYW1lX2R1bW15X2luaXRfYXJyYXlfZW50cnkAdHJhY2UuYwBldmVudABsb2NrAG91dABieXRlcwBzaWduYXR1cmUAZ3JvdXBfaWQAZGVwdGgAc2VxAGxlZGdlcgBsYXN0LjAAX19GUkFNRV9FTkRfXwBfZmluaQBfX2Rzb19oYW5kbGUAX0RZTkFNSUMAX19HTlVfRUhfRlJBTUVfSERSAF9fVE1DX0VORF9fAF9HTE9CQUxfT0ZGU0VUX1RBQkxFXwBfaW5pdABuY2NsQWxsR2F0aGVyAGdldGVudkBHTElCQ18yLjIuNQBuY2NsQWxsUmVkdWNlAF9fc25wcmludGZfY2hrQEdMSUJDXzIuMy40AG5jY2xHcm91cEVuZABfSVRNX2RlcmVnaXN0ZXJUTUNsb25lVGFibGUAY2xvY2tfZ2V0dGltZUBHTElCQ18yLjE3AGdldHBpZEBHTElCQ18yLjIuNQBmY2xvc2VAR0xJQkNfMi4yLjUAX19zdGFja19jaGtfZmFpbEBHTElCQ18yLjQAbWVtc2V0QEdMSUJDXzIuMi41AF9fdGxzX2dldF9hZGRyQEdMSUJDXzIuMwBuY2NsQ29tbUdldEFzeW5jRXJyb3IAc3lzY2FsbEBHTElCQ18yLjIuNQBfX2dtb25fc3RhcnRfXwBwdGhyZWFkX211dGV4X3VubG9ja0BHTElCQ18yLjIuNQBmZmx1c2hAR0xJQkNfMi4yLjUAbmNjbFJlY3YAbmNjbFNlbmQAZm9wZW5AR0xJQkNfMi4yLjUAcmVuYW1lQEdMSUJDXzIuMi41AGRsc3ltQEdMSUJDXzIuMi41AGZ3cml0ZUBHTElCQ18yLjIuNQBfSVRNX3JlZ2lzdGVyVE1DbG9uZVRhYmxlAG5jY2xHcm91cFN0YXJ0AF9fY3hhX2ZpbmFsaXplQEdMSUJDXzIuMi41AHB0aHJlYWRfbXV0ZXhfbG9ja0BHTElCQ18yLjIuNQAALnN5bXRhYgAuc3RydGFiAC5zaHN0cnRhYgAubm90ZS5nbnUucHJvcGVydHkALm5vdGUuZ251LmJ1aWxkLWlkAC5nbnUuaGFzaAAuZHluc3ltAC5keW5zdHIALmdudS52ZXJzaW9uAC5nbnUudmVyc2lvbl9yAC5yZWxhLmR5bgAucmVsYS5wbHQALmluaXQALnBsdC5nb3QALnBsdC5zZWMALnRleHQALmZpbmkALnJvZGF0YQAuZWhfZnJhbWVfaGRyAC5laF9mcmFtZQAudGRhdGEALnRic3MALmluaXRfYXJyYXkALmZpbmlfYXJyYXkALmR5bmFtaWMALmdvdC5wbHQALmRhdGEALmJzcwAuY29tbWVudAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABsAAAAHAAAAAgAAAAAAAADgAgAAAAAAAOACAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAAuAAAABwAAAAIAAAAAAAAAAAMAAAAAAAAAAwAAAAAAACQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAQQAAAPb//28CAAAAAAAAACgDAAAAAAAAKAMAAAAAAABAAAAAAAAAAAQAAAAAAAAACAAAAAAAAAAAAAAAAAAAAEsAAAALAAAAAgAAAAAAAABoAwAAAAAAAGgDAAAAAAAAoAIAAAAAAAAFAAAAAQAAAAgAAAAAAAAAGAAAAAAAAABTAAAAAwAAAAIAAAAAAAAACAYAAAAAAAAIBgAAAAAAALUBAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAWwAAAP///28CAAAAAAAAAL4HAAAAAAAAvgcAAAAAAAA4AAAAAAAAAAQAAAAAAAAAAgAAAAAAAAACAAAAAAAAAGgAAAD+//9vAgAAAAAAAAD4BwAAAAAAAPgHAAAAAAAAcAAAAAAAAAAFAAAAAgAAAAgAAAAAAAAAAAAAAAAAAAB3AAAABAAAAAIAAAAAAAAAaAgAAAAAAABoCAAAAAAAAMAAAAAAAAAABAAAAAAAAAAIAAAAAAAAABgAAAAAAAAAgQAAAAQAAABCAAAAAAAAACgJAAAAAAAAKAkAAAAAAACAAQAAAAAAAAQAAAAZAAAACAAAAAAAAAAYAAAAAAAAAIsAAAABAAAABgAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAGwAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAACGAAAAAQAAAAYAAAAAAAAAIBAAAAAAAAAgEAAAAAAAABABAAAAAAAAAAAAAAAAAAAQAAAAAAAAABAAAAAAAAAAkQAAAAEAAAAGAAAAAAAAADARAAAAAAAAMBEAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAAJoAAAABAAAABgAAAAAAAABAEQAAAAAAAEARAAAAAAAAAAEAAAAAAAAAAAAAAAAAABAAAAAAAAAAEAAAAAAAAACjAAAAAQAAAAYAAAAAAAAAQBIAAAAAAABAEgAAAAAAAGEMAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAqQAAAAEAAAAGAAAAAAAAAKQeAAAAAAAApB4AAAAAAAANAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAK8AAAABAAAAAgAAAAAAAAAAIAAAAAAAAAAgAAAAAAAAyAEAAAAAAAAAAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAC3AAAAAQAAAAIAAAAAAAAAyCEAAAAAAADIIQAAAAAAAGQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAxQAAAAEAAAACAAAAAAAAADAiAAAAAAAAMCIAAAAAAAD0AwAAAAAAAAAAAAAAAAAACAAAAAAAAAAAAAAAAAAAAM8AAAABAAAAAwQAAAAAAADAPQAAAAAAAMAtAAAAAAAABAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAADWAAAACAAAAAMEAAAAAAAA0D0AAAAAAADELQAAAAAAABRAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAA3AAAAA4AAAADAAAAAAAAANA9AAAAAAAA0C0AAAAAAAAIAAAAAAAAAAAAAAAAAAAACAAAAAAAAAAIAAAAAAAAAOgAAAAPAAAAAwAAAAAAAADYPQAAAAAAANgtAAAAAAAACAAAAAAAAAAAAAAAAAAAAAgAAAAAAAAACAAAAAAAAAD0AAAABgAAAAMAAAAAAAAA4D0AAAAAAADgLQAAAAAAANABAAAAAAAABQAAAAAAAAAIAAAAAAAAABAAAAAAAAAAlQAAAAEAAAADAAAAAAAAALA/AAAAAAAAsC8AAAAAAAAwAAAAAAAAAAAAAAAAAAAACAAAAAAAAAAIAAAAAAAAAP0AAAABAAAAAwAAAAAAAADoPwAAAAAAAOgvAAAAAAAAmAAAAAAAAAAAAAAAAAAAAAgAAAAAAAAACAAAAAAAAAAGAQAAAQAAAAMAAAAAAAAAgEAAAAAAAACAMAAAAAAAAAgAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAADAEAAAgAAAADAAAAAAAAAKBAAAAAAAAAiDAAAAAAAABoAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAABEBAAABAAAAMAAAAAAAAAAAAAAAAAAAAIgwAAAAAAAALQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAABAAAAAgAAAAAAAAAAAAAAAAAAAAAAAAC4MAAAAAAAAFgFAAAAAAAAHgAAAB4AAAAIAAAAAAAAABgAAAAAAAAACQAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAEDYAAAAAAABnAwAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAABEAAAADAAAAAAAAAAAAAAAAAAAAAAAAAHc5AAAAAAAAGgEAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAA="
TRACER_SHA = "d2de013ee8170506352a2a01cf34fcbd046ccbde62586087b9ac9f9c430b4a78"
def build_tracer(root):
    source=root/"nccl-call-trace.c";source.write_text(TRACER_SOURCE)
    import base64
    output=root/"nccl-call-trace.so"
    if platform.machine().lower() in ("x86_64","amd64"):
        binary=base64.b64decode(TRACER_BINARY)
        if hashlib.sha256(binary).hexdigest()!=TRACER_SHA: raise RuntimeError("TRACER_SHA_MISMATCH")
        output.write_bytes(binary)
        check=capture(root,"tracer-loader",[sys.executable,"-c",
            "import ctypes,sys;ctypes.CDLL(sys.argv[1])",str(output)])
        if check.get("code")==0:
            write(root/"tracer.json",dict(status="PREBUILT_VERIFIED",sha256=TRACER_SHA,
                minimum_glibc="2.17",scope="NCCL boundary logging; kernels unchanged"))
            return output
    compiler=shutil.which("cc") or shutil.which("gcc")
    if not compiler:
        write(root/"tracer.json",dict(status="UNAVAILABLE",reason="C compiler absent; native phase logs still collected"))
        return None
    output=root/"nccl-call-trace.so"
    result=capture(root,"tracer-build",[compiler,"-O2","-shared","-fPIC","-pthread",str(source),"-ldl","-o",str(output)],30)
    ok=result.get("code")==0 and output.exists()
    write(root/"tracer.json",dict(status="BUILT" if ok else "UNAVAILABLE",details=result))
    return output if ok else None

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget",type=int,default=1800)
    parser.add_argument("--child",type=Path)
    args=parser.parse_args()
    if args.child:
        try:
            result=stage_worker(args.child,json.loads((args.child/"spec.json").read_text()))
            write(args.child/"result.json",result)
        except BaseException:
            write(args.child/"result.json",dict(status="FAIL",error=traceback.format_exc()))
            traceback.print_exc()
            sys.exit(1)
    else:
        if not 300<=args.budget<=1800: parser.error("budget must be 300..1800 seconds")
        sys.exit(main(args))
