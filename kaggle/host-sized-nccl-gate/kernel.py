"""Pinned-GitHub, two-T4 CUCO_RANK/HOST_SIZED_NCCL correctness gate."""
import hashlib
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile

SOURCE = os.environ.get("MGBFS_GATE_SOURCE", "c7488ca6d901afc1545edba4e25ea28f915f8206")
if len(SOURCE) != 40 or any(c not in "0123456789abcdef" for c in SOURCE):
    raise ValueError("SOURCE_REQUIRES_FULL_COMMIT_ID")
CUCO = "532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4"
MODE = os.environ.get("MGBFS_GATE_MODE", "lsa_abort_stack")
LSA_ENABLED = MODE in ("process_faults_only", "owner_capture_then_lsa_faults", "lsa_abort_stack", "provision_only")


import base64,time

DEBUGGER_PATH = "/usr/bin/gdb" if MODE == "lsa_abort_stack" else "/usr/local/cuda/bin/cuda-gdb"
SANITIZER_PATH = "/usr/local/cuda/bin/compute-sanitizer"

def debugger_process_rank(environ):
    values = [item[5:] for item in environ.split(b'\0') if item.startswith(b'RANK=')]
    if len(values) != 1 or not values[0].isdigit():
        return None
    return int(values[0])

def debugger_abort_origin(groups, transcript):
    begin = transcript.rfind('MGBFS_FAILURE_TEARDOWN rank=0 stage=nccl_abort_begin')
    end = transcript.rfind('MGBFS_FAILURE_TEARDOWN rank=0 stage=nccl_abort_end')
    if begin < 0 or end > begin:
        return None
    candidates = [group for group, row in groups.items()
                  if row.get('application') and row.get('rank') == 0
                  and 'exit_code' not in row]
    return candidates[0] if len(candidates) == 1 else None

def debugger_nccl_libraries(maps):
    paths = set()
    for line in maps.splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and Path(fields[5]).name.startswith('libnccl.so'):
            paths.add(fields[5])
    return sorted(paths)

def debugger_parent_argv(target, parent):
    def is_elf(path):
        with Path(path).open("rb") as stream:
            return stream.read(4) == b"\x7fELF"
    if is_elf(target[0]):
        return list(target)
    if parent is None or not is_elf(parent):
        raise RuntimeError("DEBUGGER_PARENT_NOT_ELF")
    # env execs the original launcher with unchanged argv; no shell parsing,
    # attaching, security changes or alternative sanitizer implementation.
    return [str(parent), *target]

def diagnostic_preflight(logs):
    tools={"mode":"normal parent launch; no attach", "debugger_path":DEBUGGER_PATH,"sanitizer_path":SANITIZER_PATH}
    for name,path in (("debugger",DEBUGGER_PATH),("sanitizer",SANITIZER_PATH)):
        p=subprocess.run([path,"--version"],capture_output=True,text=True,timeout=10)
        tools[name]={"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr,"realpath":str(Path(path).resolve())}
        (logs/"diagnostic-tools.json").write_text(json.dumps(tools,indent=2))
        if p.returncode: raise RuntimeError("DIAGNOSTIC_UNAVAILABLE: "+name)
    command=[DEBUGGER_PATH,"-q","-n","-batch","-ex","set disable-randomization off","-ex","starti","-ex","thread apply all bt","-ex","continue","--args","/bin/true"]
    p=subprocess.run(command,capture_output=True,text=True,timeout=15)
    text=p.stdout+p.stderr
    tools['parent_launch']={"command":command,"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
    tools['status']='READY' if p.returncode==0 and '#0' in text and not any(x in text.lower() for x in ('permission denied','operation not permitted')) else 'PARENT_LAUNCH_DENIED_OR_UNRESOLVED'
    (logs/"diagnostic-tools.json").write_text(json.dumps(tools,indent=2))
    if tools['status']!='READY': raise RuntimeError("DIAGNOSTIC_UNAVAILABLE: parent launch failed; no further ptrace attempts")

def diagnostic_replay(binary,source,env,logs, *, target_command=None,
                      sample_points=(90,140), supervision_seconds=180,
                      sample_origin_abort=False):
    import threading,queue,re
    if target_command is None:
        env=dict(env,NCCL_DEBUG="INFO",MGBFS_TRACE_ROUTE="1",MGBFS_TRACE_ROUTE_NO_SYNC="1",MGBFS_TRACE_NCCL_GATE="1")
        target=[SANITIZER_PATH,"--tool","memcheck","--report-api-errors","no","--error-exitcode","97",str(binary),"cuco_rank_two_gpu_dense_layers_and_archives_match_oracle","--exact","--nocapture","--test-threads=1"]
    else:
        env=dict(env,NCCL_DEBUG="INFO")
        target=list(target_command)
    launched_target = debugger_parent_argv(target, shutil.which("env"))
    command=[DEBUGGER_PATH,'-q','-n','--interpreter=mi2','--args',*launched_target]
    result={'status':'UNKNOWN','mode':'DEBUGGER_PARENT_MI_STACK_SAMPLED','command':command,'target_command':target,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'env':{k:v for k,v in env.items() if k.startswith('MGBFS_') or k.startswith('NCCL_')},'supervision_seconds':180,'stack_samples':[],'thread_groups':{},'untraced_replay':'NOT_RUN','diagnostic_changes':'MI async, child fork following, retained inferiors scheduled together; no attach or security change'}
    result['supervision_seconds'] = supervision_seconds
    result['scope'] = 'Parent-debugger diagnostic only; not graceful termination or performance acceptance'
    def save(): (logs/'diagnostic.json').write_text(json.dumps(result,indent=2))
    save()
    p=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    q=queue.Queue(); raw=logs/'debugger-mi.log'; decoded=logs/'memcheck.parent-debugger.log'
    def reader():
        for line in p.stdout: q.put(line)
        q.put(None)
    worker=threading.Thread(target=reader,daemon=True); worker.start()
    token=0; eof=False; text_lines=[]; sample_active=None
    raw_stream=raw.open('w'); decoded_stream=decoded.open('w')
    def drain():
        nonlocal eof
        while True:
            try: line=q.get_nowait()
            except queue.Empty: break
            if line is None: eof=True; break
            raw_stream.write(line); raw_stream.flush()
            content=line
            if line[:1] in ('~','@','&'):
                try: content=json.loads(line[1:].strip())
                except (ValueError,TypeError): pass
            text_lines.append(content); decoded_stream.write(content); decoded_stream.flush()
            if sample_active is not None: sample_active.append(content)
            m=re.search(r'=thread-group-started,id="([^"]+)",pid="(\d+)"',line)
            if m: result['thread_groups'][m[1]]={'pid':int(m[2])}
            m=re.search(r'=thread-group-exited,id="([^"]+)"(?:,exit-code="([^"]+)")?',line)
            if m: result['thread_groups'].setdefault(m[1],{})['exit_code']=m[2]
    def send(command):
        nonlocal token
        token+=1; tag=str(token); p.stdin.write(tag+command+'\n'); p.stdin.flush(); return tag
    def await_result(tag,seconds=4):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline and p.poll() is None:
            drain()
            transcript=raw.read_text(errors='replace')
            match=re.search(r'^'+tag+r'\^(done|running|error)(.*)$',transcript,re.M)
            if match: return match.group(1),match.group(2)
            time.sleep(.05)
        return 'unresolved',''
    started=None
    try:
        settings=('-gdb-set pagination off','-gdb-set disable-randomization off','-gdb-set mi-async on','-gdb-set follow-fork-mode child','-gdb-set detach-on-fork off','-interpreter-exec console "set schedule-multiple on"')
        result['debugger_settings']=[]
        for setting in settings:
            response=await_result(send(setting)); result['debugger_settings'].append({'command':setting,'response':response})
            if response[0]!='done': raise RuntimeError('DEBUGGER_SETUP_REJECTED: '+setting)
        started=time.monotonic(); result['start_monotonic']=started
        running=await_result(send('-exec-run'))
        if running[0]!='running': raise RuntimeError('DEBUGGER_RUN_REJECTED: '+str(running))
        sampled=set(); app_seen=False; origin_sampled=False
        while time.monotonic()-started<supervision_seconds and p.poll() is None:
            drain(); elapsed=time.monotonic()-started
            for group,row in result['thread_groups'].items():
                try:
                    process_root = Path('/proc') / str(row['pid'])
                    if Path(os.readlink(process_root/'exe')).resolve()==binary.resolve():
                        row['application']=True; app_seen=True
                        rank = debugger_process_rank((process_root/'environ').read_bytes())
                        if rank is not None: row['rank'] = rank
                        if 'nccl_libraries' not in row:
                            maps = (process_root/'maps').read_text()
                            libraries = debugger_nccl_libraries(maps)
                            if libraries:
                                (logs/('rank-maps-'+str(row['pid'])+'.txt')).write_text(maps)
                                row['nccl_libraries'] = []
                                for path in libraries:
                                    identity = {'mapped_path': path}
                                    try:
                                        identity['sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                        tool = shutil.which('readelf')
                                        if tool:
                                            notes = subprocess.run([tool,'-n',path], capture_output=True,
                                                                   text=True, timeout=2)
                                            match = re.search(r'Build ID:\s*(\S+)', notes.stdout)
                                            identity['build_id'] = match.group(1) if match else None
                                            identity['readelf_returncode'] = notes.returncode
                                    except (OSError,subprocess.TimeoutExpired) as error:
                                        identity['identity_error'] = str(error)
                                    row['nccl_libraries'].append(identity)
                                save()
                except OSError: pass
            origin = debugger_abort_origin(result['thread_groups'], ''.join(text_lines)) if sample_origin_abort else None
            points = list(sample_points)
            if origin is not None and not origin_sampled:
                points.append(elapsed)
                origin_sampled = True
            for point in points:
                if elapsed>=point and point not in sampled:
                    sampled.add(point); rec={'scheduled_seconds':point,'actual_seconds':elapsed}; result['stack_samples'].append(rec)
                    apps=[(group,row) for group,row in result['thread_groups'].items() if row.get('application') and 'exit_code' not in row]
                    # Independent snapshots: later MI exit events must not
                    # retroactively change the recorded sample membership.
                    rec['application_groups']=json.loads(json.dumps(apps))
                    if sample_origin_abort:
                        rec['trigger']='origin_rank0_abort_begin'
                        apps=[(group,row) for group,row in apps if group == origin]
                    before=len(text_lines)
                    response=await_result(send('-exec-interrupt --all'),seconds=min(4,supervision_seconds-elapsed)); rec['interrupt_response']=response
                    if response[0]!='done': rec['status']='INTERRUPT_UNRESOLVED'
                    else:
                        stop_deadline=min(started+supervision_seconds,time.monotonic()+4)
                        while time.monotonic()<stop_deadline:
                            drain()
                            if '*stopped' in ''.join(text_lines[before:]): break
                            time.sleep(.05)
                        rec['stop_observed']='*stopped' in ''.join(text_lines[before:])
                        sample_active=[]
                        if sample_origin_abort:
                            # Cancellation may have completed while interrupt
                            # was queued. Never attribute a later stack to it.
                            current = debugger_abort_origin(result['thread_groups'], ''.join(text_lines))
                            apps=[(group,row) for group,row in apps if group == current]
                            if apps:
                                process_root = Path('/proc') / str(apps[0][1]['pid'])
                                try:
                                    live = (Path(os.readlink(process_root/'exe')).resolve() == binary.resolve()
                                            and debugger_process_rank((process_root/'environ').read_bytes()) == 0)
                                except OSError: live = False
                                if not live: apps=[]
                        if apps and rec['stop_observed']:
                            rec['selected_application']=json.loads(json.dumps(apps[0]))
                            rec['inferior_response']=await_result(send('-interpreter-exec console "inferior '+apps[0][0][1:]+'"'))
                            if rec['inferior_response'][0]=='done':
                                rec['bt_response']=await_result(send('-interpreter-exec console "thread apply all bt"'))
                                rec['status']='STACK_COLLECTED' if rec['bt_response'][0]=='done' and '#0' in ''.join(sample_active) else 'STACK_UNRESOLVED'
                            else: rec['status']='INFERIOR_SELECTION_UNRESOLVED'
                        else: rec['status']='APPLICATION_INFERIOR_UNRESOLVED'
                        (logs/('host-stack-'+str(point)+'.log')).write_text(''.join(sample_active)); sample_active=None
                    rec['resume_response']=await_result(send('-exec-continue --all'))
                    save()
            exited=[row for row in result['thread_groups'].values() if 'exit_code' in row]
            if app_seen and exited and len(exited)==len(result['thread_groups']): break
            if eof: break
            time.sleep(.1)
        drain()
        elapsed=time.monotonic()-started
        if elapsed>=supervision_seconds and any('exit_code' not in row for row in result['thread_groups'].values()): result['status']='TIMEOUT'
        else: result['status']='FINISHED'
        result['elapsed_seconds']=elapsed
    except Exception as error:
        result['status']='DIAGNOSTIC_FAILURE'; result['error']=str(error)
    finally:
        if p.poll() is None:
            if result['status']=='FINISHED':
                try: send('-gdb-exit'); p.wait(timeout=2)
                except (OSError,subprocess.TimeoutExpired): pass
            if p.poll() is None:
                try: os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError: pass
                try: p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try: os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                    try: p.wait(timeout=5)
                    except subprocess.TimeoutExpired: result['reap']='UNRESOLVED'
        drain(); raw_stream.close(); decoded_stream.close()
    text=decoded.read_text(errors='replace'); result['debugger_returncode']=p.poll()
    result['oracle_passed']='test result: ok. 1 passed; 0 failed' in text
    result['clean_summary']='ERROR SUMMARY: 0 errors' in text
    apps=[row for row in result['thread_groups'].values() if row.get('application')]
    result['application_exit_codes']=[row.get('exit_code') for row in apps]
    if result['status']=='FINISHED': result['status']='PASS' if apps and all(row.get('exit_code')=='0' for row in apps) and result['oracle_passed'] and result['clean_summary'] else 'FAIL'
    result['last_route_lines']=[line for line in text.splitlines() if 'mgbfs' in line.lower() or 'stage=' in line][-40:]
    result['host_stack_acceptance']='COLLECTED' if result['stack_samples'] and all(row.get('status')=='STACK_COLLECTED' for row in result['stack_samples']) else 'NOT_PROVEN'
    if sample_origin_abort:
        result['origin_abort_trigger_observed']=origin_sampled
        result['abort_reproduction']='SAMPLED' if origin_sampled else 'NOT_REPRODUCED_OR_UNRESOLVED'
    save(); return result

def main():
    hardware = os.environ.get("MGBFS_DIAGNOSTIC_HARDWARE", "T4")
    if hardware not in ("T4", "A4000"):
        raise ValueError("Unsupported diagnostic hardware")
    architecture = "75" if hardware == "T4" else "86"
    work = Path(tempfile.mkdtemp(prefix="mgbfs-lsa-bfs-", dir="/tmp"))
    logs = Path(os.environ.get("MGBFS_DIAGNOSTIC_LOGS", "/kaggle/working/lsa-bfs-gate"))
    logs.mkdir(parents=True, exist_ok=True)
    report = {"source": SOURCE, "status": "INCOMPLETE", "scope":
              ("two physical T4; independent rank-process owner, archive-admission and archive-finish fault propagation"
               if MODE == "process_faults_only" else
               "two physical T4; one-rank archive slot exhaustion before exchange"
               if MODE == "archive_fault_gate" else
               "two physical T4; warmup/CLI admission fault propagation and archive cleanup"
               if MODE == "warmup_admission_gate" else
               "two physical T4; boundary agreement, archive integrity and independent S4 full-state oracle")}

    def save():
        report["hardware_gate"] = hardware
        report["t4_acceptance_eligible"] = hardware == "T4"
        report["cuda_architecture"] = architecture
        if hardware != "T4":
            report["scope"] = report["scope"].replace("T4", hardware)
        (logs / "summary.json").write_text(json.dumps(report, indent=2))

    save()
    source = work / "source"
    # Only committed public source enters the build. Never mix the old working
    # snapshot with a new commit label or claim it exercises current runtime.
    subprocess.run(["git", "clone", "--no-checkout",
                    "https://github.com/TryDotAtwo/MultiGPUBFS.git", str(source)],
                   check=True, timeout=180)
    subprocess.run(["git", "checkout", "--detach", SOURCE], cwd=source,
                   check=True, timeout=60)
    def git_text(*args):
        return subprocess.check_output(["git", *args], cwd=source, text=True,
                                       timeout=30).strip()
    if git_text("rev-parse", "HEAD") != SOURCE or git_text("status", "--porcelain"):
        raise RuntimeError("SOURCE_CHECKOUT_IDENTITY")
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=source,
                                    timeout=30).decode().split("\0")
    identity = {"commit": SOURCE, "tree": git_text("rev-parse", "HEAD^{tree}"),
                "file_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                for name in names if name}}
    report["snapshot"] = identity
    (logs / "source-manifest.json").write_text(json.dumps(identity, indent=2))
    if MODE == "full_bfs_memcheck_diagnostic":
        try:
            diagnostic_preflight(logs)
        except Exception as error:
            report["status"] = "DIAGNOSTIC_UNAVAILABLE"
            report["error"] = str(error)
            save()
            raise
    spec = importlib.util.spec_from_file_location("gate", source / "kaggle/native-primitives/kernel.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    spec = importlib.util.spec_from_file_location("library", source / "kaggle/library-owner/kernel.py")
    library = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(library)
    env = library.isolated_environment(os.environ)
    env["NCCL_CUMEM_ENABLE"] = "1"
    env["CARGO_PROFILE_DEV_DEBUG"] = "2"
    env["PIP_DEFAULT_TIMEOUT"] = "300"
    env["PIP_RETRIES"] = "5"

    def run(command, name, cwd=source, timeout=900):
        return gate.run(command, cwd=cwd, env=env, logs=logs, name=name, timeout=timeout)

    try:
        inventory_spec = importlib.util.spec_from_file_location(
            "hardware_inventory", Path(__file__).parents[1] / "native-primitives/kernel.py")
        inventory_gate = importlib.util.module_from_spec(inventory_spec)
        inventory_spec.loader.exec_module(inventory_gate)
        report["gpus"] = inventory_gate.validate_gpus(run([
            "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free",
            "--format=csv,noheader,nounits"], "inventory"), hardware=hardware)
        cudart = ctypes.CDLL("libcudart.so.12")
        p2p = []
        for source_gpu, target_gpu in ((0, 1), (1, 0)):
            allowed = ctypes.c_int()
            rc = cudart.cudaDeviceCanAccessPeer(ctypes.byref(allowed), source_gpu, target_gpu)
            p2p.append({"source": source_gpu, "target": target_gpu,
                        "cuda_status": rc, "allowed": allowed.value})
        report["p2p"] = p2p
        report["selected_transport"] = "NCCL_LSA" if LSA_ENABLED else "HOST_SIZED_NCCL"
        report["p2p_policy"] = ("required for explicit LSA mode" if LSA_ENABLED else
                                "inventory only; ordinary NCCL selects supported transport")
        save()
        lsa_host_supported = all(row["cuda_status"] == 0 and row["allowed"] == 1 for row in p2p)
        if LSA_ENABLED and not lsa_host_supported and MODE != "owner_capture_then_lsa_faults":
            report["status"] = "UNSUPPORTED_HOST"
            save()
            return
        if MODE == "lsa_abort_stack":
            # Host stacks only: CUDA-GDB enables driver debug-agent forks before
            # torchrun launches ranks, diverting follow-fork-mode child away
            # from the actual BFS. Do not attach or change ptrace policy.
            if not Path(DEBUGGER_PATH).is_file():
                run(["apt-get", "update"], "host-gdb-package-index", timeout=180)
                run(["apt-get", "install", "-y", "gdb"], "host-gdb-install", timeout=180)
            try:
                diagnostic_preflight(logs)
            except Exception as error:
                report["status"] = "DIAGNOSTIC_UNAVAILABLE"
                report["error"] = str(error)
                save()
                raise
        sdk = work / "cuda-12.9"
        sdk.mkdir()
        for component, version, digest in library.CUDA_COMPONENTS:
            name = f"{component}-linux-x86_64-{version}-archive"
            archive = work / (name + ".tar.xz")
            url = ("https://developer.download.nvidia.com/compute/cuda/redist/"
                   f"{component}/linux-x86_64/{archive.name}")
            run(["curl", "--fail", "--location", "--silent", "--show-error",
                 "--retry", "8", "--retry-all-errors", "--retry-delay", "5",
                 "--continue-at", "-", "--connect-timeout", "30", "--max-time", "600",
                 url, "--output", str(archive)], component + "-download", timeout=5400)
            with archive.open("rb") as package:
                if hashlib.file_digest(package, "sha256").hexdigest() != digest:
                    raise RuntimeError("CUDA_SDK_CHECKSUM")
            run(["tar", "-xf", str(archive), "-C", str(work)], component + "-extract")
            shutil.copytree(work / name, sdk, dirs_exist_ok=True)
        (sdk / "lib64").symlink_to("lib", target_is_directory=True)
        env["PATH"] = str(sdk / "bin") + ":" + env.get("PATH", "")
        env["CUDACXX"] = str(sdk / "bin/nvcc")
        venv = work / "venv"
        run([sys.executable, "-m", "venv", "--without-pip", str(venv)], "venv")
        python = str(venv / "bin/python")
        run([sys.executable, "-m", "pip", "--python", python, "install",
             "--only-binary=:all:", "--no-cache-dir", "--require-hashes", "-r",
             str(source / "experiments/library_owner/requirements-linux-x86_64.lock")],
            "dependencies", timeout=1200)
        site = subprocess.check_output([python, "-c", "import site; print(site.getsitepackages()[0])"],
                                       text=True, env=env).strip()
        site = Path(site)
        prefixes = library.cmake_prefixes(site)
        libdirs = sorted({str(p.parent) for p in site.rglob("*.so*") if p.is_file()})
        nccl_target = work / "nccl"
        run([sys.executable, "-m", "pip", "install", "--no-deps", "--target",
             str(nccl_target), "nvidia-nccl-cu12==2.29.7"], "nccl-install")
        nccl = nccl_target / "nvidia/nccl"
        if not (nccl / "include/nccl_device.h").is_file():
            raise RuntimeError("PINNED_NCCL_DEVICE_HEADER")
        env["PATH"] = str(venv / "bin") + ":" + env["PATH"]
        env["LD_LIBRARY_PATH"] = ":".join([str(nccl / "lib"), str(sdk / "lib"),
                                            *libdirs, env.get("LD_LIBRARY_PATH", "")])
        env["MGBFS_CUDART_LIB_DIR"] = str(sdk / "lib")
        if MODE == "nccl_abort_isolation":
            binary = work / "nccl-nonblocking-abort-isolation"
            run(["g++", "-std=c++17", "-pthread", "-x", "c++",
                 "-I" + str(nccl / "include"), "-I" + str(sdk / "include"),
                 str(source / "experiments/nccl_nonblocking_abort_isolation.cpp"),
                 "-x", "none", str(nccl / "lib/libnccl.so.2"),
                 "-L" + str(sdk / "lib"), "-lcudart",
                 "-Wl,-rpath," + str(nccl / "lib"),
                 "-o", str(binary)], "abort-isolation-build", timeout=600)
            try:
                completed = subprocess.run([str(binary)], cwd=source, env=env,
                                           capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired as error:
                (logs / "abort-isolation.log").write_text(
                    str(error.stdout) + str(error.stderr))
                report["scope"] = "independent NCCL nonblocking abort; no BFS code"
                report["abort_isolation"] = {"status": "TIMEOUT"}
                report["status"] = "DIAGNOSTIC_COMPLETE"
                return
            output = completed.stdout + completed.stderr
            (logs / "abort-isolation.log").write_text(output)
            report["scope"] = ("independent NCCL 2.29.7 nonblocking asymmetric "
                               "abort on two physical T4s; no BFS code")
            report["abort_isolation"] = {
                "returncode": completed.returncode,
                "passed_ranks": output.count("stage=abort result=PASS"),
            }
            if completed.returncode != 0 or report["abort_isolation"]["passed_ranks"] != 2:
                raise RuntimeError("NCCL_NONBLOCKING_ABORT_GATE")
            report["status"] = "COMPLETE"
            return
        if MODE in ("nccl_window_isolation", "nccl_window_nonblocking"):
            binary = work / "nccl-window-isolation"
            run(["g++", "-std=c++17", "-pthread", "-x", "c++",
                 "-I" + str(nccl / "include"), "-I" + str(sdk / "include"),
                 str(source / "experiments/nccl_window_isolation.cu"),
                 "-x", "none", str(nccl / "lib/libnccl.so.2"),
                 "-L" + str(sdk / "lib"), "-lcudart",
                 "-Wl,-rpath," + str(nccl / "lib"),
                 "-o", str(binary)], "window-isolation-build", timeout=600)
            report["scope"] = ("independent NCCL ncclMemAlloc and "
                               "ncclCommWindowRegister on two physical T4s; no BFS code")
            report["window_runs"] = {}
            cases = (("blocking", [str(binary)]),
                     ("nonblocking", [str(binary), "nonblocking"])) if MODE == "nccl_window_nonblocking" else (
                     ("plain", [str(binary)]),
                     ("initcheck", ["compute-sanitizer", "--tool", "initcheck",
                                    "--report-api-errors", "no", "--error-exitcode", "97",
                                    str(binary)]))
            for label, command in cases:
                try:
                    completed = subprocess.run(command, cwd=source, env=env,
                                               capture_output=True, text=True,
                                               timeout=180)
                    output = completed.stdout + completed.stderr
                    (logs / ("window-" + label + ".log")).write_text(output)
                    report["window_runs"][label] = {
                        "returncode": completed.returncode,
                        "registered_ranks": output.count("stage=window_register result=PASS"),
                        "zero_sanitizer_errors": "ERROR SUMMARY: 0 errors" in output,
                    }
                except subprocess.TimeoutExpired as error:
                    (logs / ("window-" + label + ".log")).write_text(
                        str(error.stdout) + str(error.stderr))
                    report["window_runs"][label] = {"status": "TIMEOUT"}
                save()
            report["status"] = ("COMPLETE" if MODE == "nccl_window_nonblocking" and
                                all(row.get("returncode") == 0 and row.get("registered_ranks") == 2
                                    for row in report["window_runs"].values()) else
                                "DIAGNOSTIC_COMPLETE")
            return
        env["CARGO_HOME"] = str(work / "cargo")
        env["RUSTUP_HOME"] = str(work / "rustup")
        installer = work / "rustup-init.sh"
        run(["curl", "--fail", "--location", "--max-time", "180",
             "https://sh.rustup.rs", "-o", str(installer)], "rust-download")
        run(["sh", str(installer), "-y", "--no-modify-path", "--profile", "minimal",
             "--default-toolchain", gate.RUST_VERSION], "rust-install")
        env["PATH"] = str(work / "cargo/bin") + ":" + env["PATH"]
        cuco = work / "cuco"
        gate.checkout("https://github.com/NVIDIA/cuCollections.git", CUCO,
                      cuco, env, logs, "cuco")
        build = work / "library-build"
        run(["cmake", "-S", str(source / "experiments/library_owner"), "-B", str(build),
             "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=" + architecture,
             "-DCMAKE_CUDA_COMPILER=" + str(sdk / "bin/nvcc"),
             "-DCUDAToolkit_ROOT=" + str(sdk),
             "-DCMAKE_PREFIX_PATH=" + ";".join(prefixes),
             "-DCUCO_ROOT=" + str(cuco)], "library-configure")
        run(["cmake", "--build", str(build), "--target", "mgbfs_library_owner", "-j2"],
            "library-build", timeout=1800)
        env["MGBFS_LIBRARY_OWNER_LIB_DIR"] = str(build)
        env["LD_LIBRARY_PATH"] = str(build) + ":" + env["LD_LIBRARY_PATH"]
        cutlass = work / "cutlass"
        gate.checkout("https://github.com/NVIDIA/cutlass.git", gate.CUTLASS_COMMIT,
                      cutlass, env, logs, "cutlass")
        native = work / "native-build"
        run(["cmake", "-S", str(source / "cuda"), "-B", str(native), "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF",
             *(["-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -g1"]
               if MODE in ("timeline_backtrace", "full_bfs_memcheck_diagnostic") else []),
             "-DCMAKE_CUDA_ARCHITECTURES=" + architecture, "-DCMAKE_CUDA_COMPILER=" + str(sdk / "bin/nvcc"),
             "-DCUTLASS_ROOT=" + str(cutlass),
             "-DMGBFS_NCCL_LSA=" + ("ON" if LSA_ENABLED else "OFF"),
             "-DMGBFS_NCCL_ROOT=" + str(nccl)], "native-configure")
        run(["cmake", "--build", str(native), "--target", "mgbfs_cuda", "-j2"],
            "native-build", timeout=1800)
        env["MGBFS_CUDA_LIB_DIR"] = str(native)
        env["LD_LIBRARY_PATH"] = str(native) + ":" + env["LD_LIBRARY_PATH"]
        if MODE == "provision_only":
            report["status"] = "PROVISIONED_NOT_VALIDATED"
            report["scope"] = "pinned source and dependencies built; no BFS acceptance claim"
            report["work"] = str(work)
            save()
            return
        if MODE in ("owner_capture_gate", "owner_capture_then_lsa_faults"):
            import re
            sys.path.insert(0, str(source / "scripts"))
            from eight_gpu_gate import run_command
            run(["cmake", "--build", str(build), "--target", "cuco_owner_probe", "-j2"],
                "owner-capture-build", timeout=1800)
            binary = build / "cuco_owner_probe"
            report.update(scope="single-device full production owner DAG capture; not LSA or full BFS",
                          executable_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                          stages=[])
            for tool in ("plain", "memcheck", "racecheck", "initcheck", "synccheck"):
                command = [str(binary)]
                if tool != "plain":
                    command = [SANITIZER_PATH, "--tool", tool, "--error-exitcode", "97", *command]
                started = time.monotonic()
                stage = dict(tool=tool, status="RUNNING", command=command)
                report["stages"].append(stage)
                save()
                text = run_command(command, logs / ("owner-capture-" + tool + ".log"), env, 180)
                if text.count("RANK_FULL_OWNER_DAG_CAPTURE_PASS") != 1 or '\"status\":\"PASS\"' not in text:
                    raise RuntimeError("OWNER_CAPTURE_ASSERTIONS_MISSING")
                if tool != "plain":
                    errors = re.findall(r"ERROR SUMMARY: (\d+) errors", text)
                    races = re.findall(r"RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)", text)
                    if any(int(n) for n in errors) or any(any(int(n) for n in row) for row in races):
                        raise RuntimeError("OWNER_CAPTURE_SANITIZER_FINDINGS")
                    if not errors and not (tool == "racecheck" and races):
                        raise RuntimeError("OWNER_CAPTURE_SANITIZER_SUMMARY_MISSING")
                stage.update(status="PASS", seconds=time.monotonic()-started)
                save()
            report["owner_capture_status"] = "PASS"
            if MODE == "owner_capture_gate":
                report["status"] = "PASS"
                return
            # Independent declared gates; lack of P2P does not replace LSA
            # with ordinary NCCL or change the runtime's configured backend.
            if not lsa_host_supported:
                report["lsa_fault_status"] = "UNSUPPORTED_HOST"
                report["status"] = "COMPLETE_PARTIAL"
                return
            report["lsa_fault_status"] = "INCOMPLETE"
            save()
        if MODE == "warmup_admission_gate":
            run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-cli",
                 "--features", "library-owner"], "warmup-cli-build", timeout=1800)
            cli = str(source / "target/release/mgbfs")
            env.update(MGBFS_OWNER_BACKEND="CUCO_RANK",
                       MGBFS_LIBRARY_POOL_BYTES=str(64 << 20), MGBFS_PROFILE="DENSE",
                       MGBFS_BENCH_CAPACITY="64", MGBFS_FUTURE_CAPACITY="128",
                       MGBFS_BUCKETS="8", MGBFS_SHARDS="4", MGBFS_JOB_BUCKETS="2",
                       MGBFS_BUCKET_CAPACITY="32", MGBFS_STATE_CODEC="matrix_u8",
                       MGBFS_ARCHIVE_CODEC="matrix_u8", MGBFS_ARCHIVE_ROWS="3",
                       MGBFS_ARCHIVE_SLOTS="128", MGBFS_PRE_DEDUP="ON",
                       MGBFS_BENCH_SKIP_ARCHIVE="0", MGBFS_ARCHIVE_STREAM="0",
                       MGBFS_CAPACITY_MODE="max_per_rank", MGBFS_RANK_MAP="0,1",
                       MGBFS_TRANSPORT_BACKEND="HOST_SIZED_NCCL")
            report["warmup_runs"] = {}

            def launch(name, wrapper=None, warmup="0", expected=None):
                root = work / name
                root.mkdir()
                output = logs / name
                argv = [cli, "bench", "--reference", "s4", "7",
                        str(root / "bootstrap"), str(root / "archive"), str(output)]
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                           "--nproc-per-node=2", "--no-python"]
                if wrapper:
                    command += ["/bin/bash", "-c", wrapper, "mgbfs-rank-launch"]
                command += argv
                case_env = dict(env, MGBFS_BENCH_WARMUP=warmup)
                try:
                    completed = subprocess.run(command, cwd=source, env=case_env,
                                               capture_output=True, text=True, timeout=120)
                except subprocess.TimeoutExpired as error:
                    (logs / (name + ".log")).write_text(str(error.stdout) + str(error.stderr))
                    raise RuntimeError("WARMUP_GATE_TIMEOUT: " + name) from error
                output_text = completed.stdout + completed.stderr
                (logs / (name + ".log")).write_text(output_text)
                if expected is not None:
                    if (completed.returncode == 0 or expected not in output_text
                            or "REMOTE_CONFIGURATION_FATAL" not in output_text
                            or (output / "group-complete.json").exists()):
                        raise RuntimeError("WARMUP_GATE_FATAL: " + name)
                    report["warmup_runs"][name] = "PASS_GROUP_FATAL"
                    save()
                    return
                if completed.returncode != 0:
                    raise RuntimeError("WARMUP_GATE_SUCCESS: " + name)
                if not (output / "group-complete.json").is_file():
                    raise RuntimeError("WARMUP_GATE_GROUP_MARKER")
                if (logs / (name + ".warmup/group-complete.json")).exists():
                    raise RuntimeError("WARMUP_GATE_FALSE_MARKER")
                for rank in range(2):
                    if (root / f"archive.warmup-rank-{rank}.mgbfsar1").exists():
                        raise RuntimeError("WARMUP_GATE_ARCHIVE_NOT_RELEASED")
                    row = json.loads((output / f"rank-{rank}.json").read_text())
                    warm_row = json.loads((logs / (name + ".warmup") /
                                           f"rank-{rank}.json").read_text())
                    if (row["status"] != "COMPLETE" or not row["warmup_completed"]
                            or warm_row["archive_commit_scope"] != "warmup_ephemeral"):
                        raise RuntimeError("WARMUP_GATE_RANK_RESULT")
                    run([cli, "verify", str(root / f"archive-rank-{rank}.mgbfsar1")],
                        f"{name}-verify-{rank}", timeout=120)
                report["warmup_runs"][name] = "PASS_MEASURED_GROUP_AND_ARCHIVES"
                save()

            launch("warmup-rank-mismatch",
                   'if [ "$RANK" = 0 ]; then export MGBFS_BENCH_WARMUP=1; fi; exec "$@"',
                   expected="REMOTE_CONFIGURATION_FATAL")
            launch("warmup-invalid",
                   'if [ "$RANK" = 0 ]; then export MGBFS_BENCH_WARMUP=bad; fi; exec "$@"',
                   expected="BENCH_WARMUP_CONFIG")
            launch("macro-invalid",
                   'if [ "$RANK" = 0 ]; then export MGBFS_MACRO_DEPTH=2; fi; exec "$@"',
                   expected="MACRO_MULTI_GPU_UNSUPPORTED")
            launch("archive-invalid",
                   'if [ "$RANK" = 0 ]; then export MGBFS_BENCH_SKIP_ARCHIVE=1; fi; exec "$@"',
                   expected="CLI_BENCH_ARCHIVE_REQUIRED")
            launch("warmup-normal", warmup="1")
            report["status"] = "COMPLETE"
            return
        if MODE == "boundary_gate":
            run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                  "--test", "bootstrap", "--test", "group_commit", "--test", "archive"],
                 "boundary-cpu-tests", timeout=900)
            run(["cargo", "test", "--locked", "-p", "mgbfs-runtime", "--features", "cuda",
                 "--lib", "optional_u32_rejects_invalid_value_without_panicking"],
                "boundary-reference-config-test", timeout=900)
            run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-cli",
                 "--features", "library-owner"], "boundary-cli-build", timeout=1800)
            if any(row["cuda_status"] != 0 or row["allowed"] != 1 for row in p2p):
                report["boundary_runs"] = {"linux_archive_and_protocol_cpu_tests": "PASS"}
                report["status"] = "UNSUPPORTED_HOST_AFTER_CPU"
                return
            run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                 "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                 "one_rank_constructor_failure_after_nccl_stops_peer",
                 "--", "--ignored", "--nocapture"],
                "boundary-constructor-fault", timeout=75)
            cli = str(source / "target/release/mgbfs")
            env.update(MGBFS_OWNER_BACKEND="CUCO_RANK",
                       MGBFS_LIBRARY_POOL_BYTES=str(64 << 20), MGBFS_PROFILE="DENSE",
                       MGBFS_BENCH_CAPACITY="64", MGBFS_FUTURE_CAPACITY="128",
                       MGBFS_BUCKETS="8", MGBFS_SHARDS="4", MGBFS_JOB_BUCKETS="2",
                       MGBFS_BUCKET_CAPACITY="32", MGBFS_STATE_CODEC="matrix_u8",
                       MGBFS_ARCHIVE_CODEC="matrix_u8", MGBFS_ARCHIVE_ROWS="3",
                       MGBFS_ARCHIVE_SLOTS="128", MGBFS_BENCH_WARMUP="0",
                       MGBFS_PRE_DEDUP="ON", MGBFS_BENCH_SKIP_ARCHIVE="0",
                       MGBFS_ARCHIVE_STREAM="0", MGBFS_CAPACITY_MODE="max_per_rank",
                       MGBFS_RANK_MAP="0,1")
            report["boundary_runs"] = {}
            for transport in ("HOST_SIZED_NCCL", "NCCL_LSA"):
                env["MGBFS_TRANSPORT_BACKEND"] = transport
                name = transport.lower()
                root = work / ("boundary-" + name)
                root.mkdir()
                output = logs / ("boundary-" + name)
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                           "--nproc-per-node=2", "--no-python", cli, "bench", "--reference",
                           "s4", "7", str(root / "bootstrap"), str(root / "archive"),
                           str(output)]
                run(command, "boundary-" + name, timeout=300)
                marker = json.loads((output / "group-complete.json").read_text())
                if (marker["status"] != "COMPLETE" or marker["world_size"] != 2
                        or marker["archive_commit_scope"] != "file_fsync"):
                    raise RuntimeError("GROUP_COMMIT_MARKER")
                layers = None
                for rank in range(2):
                    data = (output / f"rank-{rank}.json").read_bytes()
                    row = json.loads(data)
                    if (row["status"] != "COMPLETE" or row["rank"] != rank
                            or row["archive_commit_scope"] != "file_fsync"
                            or list(hashlib.sha256(data).digest()) != marker["rank_sha256"][rank]):
                        raise RuntimeError("GROUP_RANK_RESULT")
                    if layers is None:
                        layers = [0] * len(row["local_layer_sizes"])
                    if len(row["local_layer_sizes"]) != len(layers):
                        raise RuntimeError("GROUP_LAYER_SHAPE")
                    layers = [a + b for a, b in zip(layers, row["local_layer_sizes"])]
                    run([cli, "verify", str(root / f"archive-rank-{rank}.mgbfsar1")],
                        f"boundary-{name}-verify-{rank}", timeout=300)
                if layers != [1, 3, 5, 6, 5, 3, 1]:
                    raise RuntimeError("GROUP_LAYER_ORACLE")
                report["boundary_runs"][name] = "PASS_GROUP_MARKER_AND_ARCHIVES"
                save()
            env["MGBFS_TRANSPORT_BACKEND"] = "NCCL_LSA"
            fault = work / "boundary-archive-fault"
            fault.mkdir()
            (fault / "archive-rank-0.mgbfsar1").write_bytes(b"occupied")
            fault_output = logs / "boundary-archive-fault-results"
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       "--nproc-per-node=2", "--no-python", cli, "bench", "--reference",
                       "s4", "7", str(fault / "bootstrap"), str(fault / "archive"),
                       str(fault_output)]
            failed = subprocess.run(command, cwd=source, env=env, capture_output=True,
                                    text=True, timeout=180)
            output_text = failed.stdout + failed.stderr
            (logs / "boundary-archive-fault.log").write_text(output_text)
            if (failed.returncode == 0 or "ARCHIVE_EXTENT" not in output_text
                    or "REMOTE_ARCHIVE_ADMISSION_FATAL" not in output_text
                    or (fault_output / "group-complete.json").exists()):
                raise RuntimeError("ASYMMETRIC_ARCHIVE_ADMISSION_GATE")
            report["boundary_runs"]["one_rank_archive_admission_failure"] = "PASS_GROUP_FATAL"
            config_fault = work / "boundary-config-fault"
            config_fault.mkdir()
            config_output = logs / "boundary-config-fault-results"
            launcher = 'if [ "$RANK" = 0 ]; then export MGBFS_SHARDS=bad; fi; exec "$@"'
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       "--nproc-per-node=2", "--no-python", "/bin/bash", "-c", launcher,
                       "mgbfs-rank-launch", cli, "bench", "--reference", "s4", "7",
                       str(config_fault / "bootstrap"), str(config_fault / "archive"),
                       str(config_output)]
            failed = subprocess.run(command, cwd=source, env=env, capture_output=True,
                                    text=True, timeout=90)
            output_text = failed.stdout + failed.stderr
            (logs / "boundary-config-fault.log").write_text(output_text)
            if (failed.returncode == 0 or "ENV_MGBFS_SHARDS" not in output_text
                    or "REMOTE_CONFIGURATION_FATAL" not in output_text
                    or (config_output / "group-complete.json").exists()):
                raise RuntimeError("ASYMMETRIC_CONFIGURATION_GATE")
            report["boundary_runs"]["one_rank_invalid_configuration"] = "PASS_GROUP_FATAL"
            env.update(MGBFS_BENCH_CAPACITY="6", MGBFS_ARCHIVE_ROWS="1")
            small = work / "boundary-small-layer-capacity"
            small.mkdir()
            small_output = logs / "boundary-small-layer-capacity"
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       "--nproc-per-node=2", "--no-python", cli, "bench", "--reference",
                       "s4", "7", str(small / "bootstrap"), str(small / "archive"),
                       str(small_output)]
            run(command, "boundary-small-layer-capacity", timeout=300)
            if not (small_output / "group-complete.json").is_file():
                raise RuntimeError("ARCHIVE_LAYER_CAPACITY_GROUP_MARKER")
            layer_counts = None
            for rank in range(2):
                row = json.loads((small_output / f"rank-{rank}.json").read_text())
                if (row["status"] != "COMPLETE" or row["rank_capacity_records"] != 6
                        or row["disk_reserved_bytes"] != 6304):
                    raise RuntimeError("ARCHIVE_LAYER_CAPACITY_METADATA")
                if layer_counts is None:
                    layer_counts = [0] * len(row["local_layer_sizes"])
                if len(layer_counts) != len(row["local_layer_sizes"]):
                    raise RuntimeError("ARCHIVE_LAYER_CAPACITY_DEPTHS")
                layer_counts = [a + b for a, b in zip(layer_counts, row["local_layer_sizes"])]
                run([cli, "verify", str(small / f"archive-rank-{rank}.mgbfsar1")],
                    f"boundary-small-layer-capacity-verify-{rank}", timeout=300)
            if layer_counts != [1, 3, 5, 6, 5, 3, 1]:
                raise RuntimeError("ARCHIVE_LAYER_CAPACITY_ORACLE")
            report["boundary_runs"]["archive_total_exceeds_layer_capacity"] = "PASS"
            oracle = run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                          "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                          "cuco_rank_two_gpu_dense_layers_and_archives_match_oracle",
                          "--", "--ignored", "--exact", "--nocapture", "--test-threads=1"],
                         "boundary-full-state-oracle", timeout=900)
            if "test result: ok. 1 passed; 0 failed" not in oracle:
                raise RuntimeError("BOUNDARY_FULL_STATE_ORACLE")
            report["boundary_runs"]["independent_full_state_oracle"] = "PASS"
            report["status"] = "COMPLETE"
            return
        if MODE == "paired_cayleypy":
            baseline_commit = "f0f2b8e5ee61173039ab9742f3a7756c9b6365e6"
            baseline = work / "cayleypy-baseline"
            gate.checkout("https://github.com/TryDotAtwo/cayleypy.git",
                          baseline_commit, baseline, env, logs, "cayleypy")
            run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-cli",
                 "--features", "library-owner"], "paired-cli-build", timeout=1800)
            sys.path.insert(0, str(source / "scripts"))
            from distributed_gpu_bench import run_group, stats
            from library_gpu_screen import run_case
            report.update(scope="paired physical 2xT4 S10; native archive mandatory, CayleyPy no archive",
                          baseline_commit=baseline_commit, rows=[])
            save()
            expected = None
            samples = {"native": [], "cayleypy": []}
            for repeat in range(5):
                for backend in (("native", "cayleypy") if repeat % 2 == 0
                                else ("cayleypy", "native")):
                    label = f"paired-s10-{backend}-r{repeat}"
                    if backend == "native":
                        archive_root = work / label
                        case_env = dict(env, MGBFS_TRANSPORT_BACKEND="NCCL_LSA",
                                        MGBFS_SHARDS="4", MGBFS_BUCKETS="256",
                                        MGBFS_ARCHIVE_SLOTS="256")
                        result = run_case(str(source / "target/release/mgbfs"),
                                          logs / label, archive_root, "s10", 3_628_800,
                                          2, 32768, 1_000_000, 1_000_000, 96 << 20,
                                          "DENSE", "ON", case_env, owner="CUCO_RANK")
                        row = result["measurement"]
                        for rank in range(2):
                            (archive_root / f"archive-rank-{rank}.mgbfsar1").unlink()
                    else:
                        case_env = dict(env, PYTHONPATH=str(baseline),
                                        CUDA_VISIBLE_DEVICES="0,1",
                                        MGBFS_BENCH_WORLD_SIZE="2")
                        command = [sys.executable, "-m", "torch.distributed.run",
                                   "--standalone", "--nproc-per-node=2",
                                   str(source / "scripts/distributed_gpu_bench.py"),
                                   "baseline-worker", "10", "1048576", "{RANK_OUT}"]
                        row = run_group(command, logs, label, case_env, timeout=1800)
                    if row["status"] != "COMPLETE" or sum(row["layer_sizes"]) != 3_628_800:
                        raise RuntimeError("PAIRED_BFS_INCOMPLETE: " + label)
                    expected = expected or row["layer_sizes"]
                    if row["layer_sizes"] != expected:
                        raise RuntimeError("PAIRED_LAYER_MISMATCH: " + label)
                    row["paired_backend"] = backend
                    row["paired_repeat"] = repeat
                    samples[backend].append(row)
                    report["rows"].append({"label": label, "backend": backend,
                                           "search_seconds": row["search_complete_seconds"],
                                           "peak_mib_per_rank": row["smi_peak_mib_per_rank"],
                                           "archive_contract": ("verified file_fsync"
                                                                if backend == "native" else "none")})
                    save()
            report["layers"] = expected
            report["native"] = stats(samples["native"])
            report["cayleypy"] = stats(samples["cayleypy"])
            report["status"] = "COMPLETE"
            return
        if MODE in ("benchmark", "timeline", "timeline_backtrace", "timeline_analysis"):
            report["scope"] = (
                "paired two-T4 S10 CUCO_RANK DENSE; archive-verified; "
                + ("five unprofiled repeats per transport" if MODE == "benchmark" else
                   "one profiled diagnostic run per transport, including startup and archive"))
            report["benchmark_config"] = {
                "group": "s10", "batch": 32768, "capacity_per_rank": 1_000_000,
                "ring_per_rank": 1_000_000, "pool_bytes_per_rank": 96 << 20,
                "shards_per_rank": 4, "buckets": 256, "archive_slots": 256,
            }
            save()
            if MODE == "timeline_backtrace":
                env["CARGO_PROFILE_RELEASE_DEBUG"] = "1"
                env["CARGO_PROFILE_RELEASE_STRIP"] = "none"
            run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-cli",
                 "--features", "library-owner"], "cli-build", timeout=1800)
            sys.path.insert(0, str(source / "scripts"))
            from distributed_gpu_bench import stats
            from library_gpu_screen import run_case
            cli = str(source / "target/release/mgbfs")
            profiled_cli = cli
            if MODE == "timeline_backtrace":
                wrapper = work / "capture-rank-maps.sh"
                wrapper.write_text(
                    "#!/bin/bash\nset -eu\n"
                    'if [[ -n "${RANK:-}" && -n "${MGBFS_DIAGNOSTIC_MAP_DIR:-}" ]]; then\n'
                    '  mgbfs_target_pid=$$\n'
                    '  ( sleep 0.25; cp "/proc/${mgbfs_target_pid}/maps" '
                    '"${MGBFS_DIAGNOSTIC_MAP_DIR}/rank-${RANK}.maps" ) &\n'
                    "fi\n"
                    f"exec {shlex.quote(cli)} \"$@\"\n"
                )
                wrapper.chmod(0o755)
                profiled_cli = str(wrapper)
            nsys = None
            if MODE in ("timeline", "timeline_backtrace", "timeline_analysis"):
                package_name = "nsight-systems-2025.3.2_2025.3.2.474-1_amd64.deb"
                package_sha = "c7cfe27e2250eb91e1a67e7feb5f2c490c7f598e3b3a3d047aff000bc49f9d6b"
                package = work / package_name
                run(["curl", "--fail", "--location", "--max-time", "300",
                     "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/"
                     + package_name, "--output", str(package)], "nsys-download")
                with package.open("rb") as downloaded:
                    if hashlib.file_digest(downloaded, "sha256").hexdigest() != package_sha:
                        raise RuntimeError("NSYS_DIGEST_MISMATCH")
                nsys_root = work / "nsys"
                run(["dpkg-deb", "--extract", str(package), str(nsys_root)], "nsys-extract")
                candidates = list(nsys_root.glob("opt/nvidia/nsight-systems/*/target-linux-x64/nsys"))
                if len(candidates) != 1:
                    raise RuntimeError("NSYS_EXECUTABLE_INVENTORY")
                nsys = str(candidates[0])
                run([nsys, "--version"], "nsys-version")
            panel = ({"NCCL_LSA": []} if MODE in ("timeline_backtrace", "timeline_analysis") else
                     {"HOST_SIZED_NCCL": [], "NCCL_LSA": []})
            expected_dispatch = {"HOST_SIZED_NCCL": "HostSizedNccl", "NCCL_LSA": "Lsa"}
            for repeat in range(5 if nsys is None else 1):
                order = list(panel) if repeat % 2 == 0 else list(reversed(panel))
                for transport in order:
                    label = f"s10-{transport.lower()}-r{repeat}"
                    case_env = dict(env, MGBFS_TRANSPORT_BACKEND=transport,
                                    MGBFS_SHARDS="4", MGBFS_BUCKETS="256",
                                    MGBFS_ARCHIVE_SLOTS="256")
                    if nsys is not None:
                        case_env["MGBFS_PROFILE_SEARCH"] = "1"
                    if MODE == "timeline_backtrace":
                        case_env["MGBFS_NSYS_CUDA_BACKTRACE"] = "sync,memory"
                        case_env["MGBFS_DIAGNOSTIC_MAP_DIR"] = str(logs / label)
                        case_env["NSYS_CONFIG_DIRECTIVES"] = (
                            f'DbgFileSearchPath="{source / "target/release"}:{native}:{build}"'
                        )
                    result = run_case(profiled_cli, logs / label, work / label, "s10",
                                      3_628_800, 2, 32768, 1_000_000, 1_000_000,
                                      96 << 20, "DENSE", "ON", case_env,
                                      owner="CUCO_RANK", nsys=nsys)
                    if any(rank.get("transport_backend") != expected_dispatch[transport]
                           for rank in result["measurement"]["rank_results"]):
                        raise RuntimeError("TRANSPORT_DISPATCH_MISMATCH")
                    panel[transport].append(result["measurement"])
                    if nsys is not None:
                        run([nsys, "stats", "--report",
                             "cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,osrt_sum",
                             "--format", "csv", result["trace"]],
                            label + "-nsys-stats", timeout=600)
                    if MODE == "timeline_analysis":
                        run([nsys, "analyze", "--rule",
                             "cuda_api_sync,gpu_gaps,gpu_time_util", result["trace"]],
                            label + "-nsys-analysis", timeout=600)
                    if MODE == "timeline_backtrace":
                        for rank in range(2):
                            mapping = logs / label / f"rank-{rank}.maps"
                            if not mapping.is_file() or str(source / "target/release/mgbfs") not in mapping.read_text():
                                raise RuntimeError("NSYS_RANK_MAPS_MISSING")
                        database = logs / label / "timeline.sqlite"
                        run([nsys, "export", "--type", "sqlite", "--force-overwrite=true",
                             "--output", str(database), result["trace"]],
                            label + "-nsys-export", timeout=600)
                        run([sys.executable, str(source / "scripts/nsys_sync_callsites.py"),
                             str(database), str(logs / label / "sync-callsites.json")],
                            label + "-sync-callsites", timeout=600)
                        addresses = logs / label / "rank-addresses.json"
                        run([sys.executable, str(source / "scripts/nsys_rank_maps.py"),
                             str(logs / label / "sync-callsites.json"), str(addresses),
                             "--map", f"0:{logs / label / 'rank-0.maps'}",
                             "--map", f"1:{logs / label / 'rank-1.maps'}"],
                            label + "-rank-addresses", timeout=600)
                        address_rows = json.loads(addresses.read_text())["rows"]
                        offsets = sorted({frame["offset"] for row in address_rows
                                          for frame in row["project_frames"]
                                          if frame["module"] == cli},
                                         key=lambda value: int(value, 16))
                        symbols = run(["addr2line", "-f", "-C", "-e", cli, *offsets],
                                      label + "-addr2line", timeout=600).splitlines()
                        if len(symbols) != 2 * len(offsets):
                            raise RuntimeError("NSYS_ADDR2LINE_SHAPE")
                        resolved = {offset: {"function": symbols[2 * index],
                                             "source": symbols[2 * index + 1]}
                                    for index, offset in enumerate(offsets)}
                        (logs / label / "rank-symbols.json").write_text(
                            json.dumps({"source": SOURCE, "executable": cli,
                                        "symbols": resolved}, indent=2))
                        report["symbolized_offsets"] = len(resolved)
                        database.unlink()
                        Path(result["trace"]).unlink()
                    report["runs"] = {key: len(value) for key, value in panel.items()}
                    save()
            if nsys is None:
                report["screen_statistics"] = {key: stats(value) for key, value in panel.items()}
            else:
                report["timeline_scope"] = "timed BFS CUDA profiler range; archive submissions included"
                if MODE == "timeline_backtrace":
                    report["callsite_scope"] = "LSA CUDA sync/copy callchains; profiled diagnostic"
                if MODE == "timeline_analysis":
                    report["analysis_scope"] = "LSA sync API and GPU gap expert rules; profiled diagnostic"
            report["status"] = "COMPLETE"
            return
        run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
             "--features", "cuda,library-owner", "--test", "library_multi_gpu",
             "--no-run"], "bfs-test-build", timeout=1800)
        if MODE in ("host_fault_only", "host_fault_and_process"):
            env["MGBFS_TRACE_ROUTE"] = "1"
            env["MGBFS_TRACE_ROUTE_NO_SYNC"] = "1"
            for name in ("lsa_one_exchange_matches_peer_payload",
                         "cuco_rank_lsa_one_rank_host_owner_error_stops_group"):
                checked = run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                               "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                               name, "--", "--ignored", "--exact", "--nocapture",
                               "--test-threads=1"], name, timeout=45)
                if "test result: ok. 1 passed; 0 failed" not in checked:
                    raise RuntimeError("HOST_FAULT_RESULT: " + name)
                report[name] = "PASS"
                save()
            report["host_fault"] = "PASS"
            if MODE == "host_fault_only":
                report["status"] = "COMPLETE"
                return
        if MODE in ("process_host_fault_only", "host_fault_and_process", "process_faults_only",
                    "process_faults_host_sized", "owner_capture_then_lsa_faults", "lsa_abort_stack"):
            sys.path.insert(0, str(source / "scripts"))
            from process_scope import spawn_group, stop_group
            # Exercise real Linux orphan/session semantics before trusting the
            # same supervisor to classify GPU rank termination.
            for pattern in ("test_process_scope_linux.py", "test_rank_process_cleanup.py"):
                run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                     "-p", pattern], "linux-" + pattern, timeout=60)
            run(["cargo", "build", "--locked", "-p", "mgbfs-cli",
                 "--features", "library-owner"], "process-fault-cli-build", timeout=1800)
            env.update(MGBFS_OWNER_BACKEND="CUCO_RANK",
                       MGBFS_TRACE_FAILURE_TEARDOWN="1",
                       MGBFS_LIBRARY_POOL_BYTES=str(64 << 20), MGBFS_PROFILE="DENSE",
                       MGBFS_BENCH_CAPACITY="64", MGBFS_FUTURE_CAPACITY="128",
                       MGBFS_BUCKETS="8", MGBFS_SHARDS="4", MGBFS_JOB_BUCKETS="2",
                       MGBFS_BUCKET_CAPACITY="32", MGBFS_STATE_CODEC="matrix_u8",
                       MGBFS_ARCHIVE_CODEC="matrix_u8", MGBFS_ARCHIVE_ROWS="3",
                       MGBFS_ARCHIVE_SLOTS="128", MGBFS_BENCH_WARMUP="0",
                       MGBFS_PRE_DEDUP="ON", MGBFS_BENCH_SKIP_ARCHIVE="0",
                       MGBFS_ARCHIVE_STREAM="0", MGBFS_CAPACITY_MODE="max_per_rank",
                       MGBFS_RANK_MAP="0,1",
                       MGBFS_TRANSPORT_BACKEND="NCCL_LSA" if LSA_ENABLED else "HOST_SIZED_NCCL")
            faults = [("owner", "MGBFS_TEST_OWNER_HOST_FAULT_RANK",
                       "TEST_INJECTED_OWNER_HOST_ERROR")]
            if MODE in ("process_faults_only", "process_faults_host_sized", "owner_capture_then_lsa_faults"):
                faults += [("archive-admission", "MGBFS_TEST_ARCHIVE_ADMISSION_FAULT_RANK",
                            "TEST_INJECTED_ARCHIVE_ADMISSION_ERROR"),
                           ("archive-finish", "MGBFS_TEST_ARCHIVE_FINISH_FAULT_RANK",
                            "TEST_INJECTED_ARCHIVE_FINISH_ERROR")]
            report["scope"] = ("two independent rank processes; DENSE CUCO_RANK; "
                               + report["selected_transport"]
                               + "; startup/owner/archive-finalize faults on each rank")
            for fault_name, env_key, expected_error, fault_rank in (
                (*fault, rank) for fault in faults for rank in (0, 1)
            ):
                for key in ("MGBFS_TEST_OWNER_HOST_FAULT_RANK",
                            "MGBFS_TEST_ARCHIVE_ADMISSION_FAULT_RANK",
                            "MGBFS_TEST_ARCHIVE_FINISH_FAULT_RANK"):
                    env.pop(key, None)
                env[env_key] = str(fault_rank)
                name = f"process-{fault_name}-fault-rank-{fault_rank}"
                root = work / name
                root.mkdir()
                output = logs / name
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                           "--nproc-per-node=2", "--monitor-interval=30", "--no-python",
                           str(source / "target/debug/mgbfs"), "bench", "--reference",
                           "s4", "1", str(root / "bootstrap"), str(root / "archive"),
                           str(output)]
                if MODE == "lsa_abort_stack":
                    # Reuse the parent-launch debugger protocol; no ptrace attach
                    # or security-policy change. This diagnostic cannot count as
                    # a fault termination gate: interruption changes scheduling.
                    report["diagnostic"] = diagnostic_replay(
                        source / "target/debug/mgbfs", source, env, logs,
                        target_command=command, sample_points=(),
                        supervision_seconds=27, sample_origin_abort=True)
                    report["status"] = "DIAGNOSTIC_ONLY"
                    save()
                    return
                with (logs / (name + ".log")).open("w") as stream:
                    started = time.monotonic()
                    process = spawn_group(command, cwd=source, env=env, stdout=stream,
                                          stderr=subprocess.STDOUT)
                    try:
                        returncode = process.wait(timeout=60)
                    except subprocess.TimeoutExpired as error:
                        report[name] = {"status": "TIMEOUT", "fault_rank": fault_rank}
                        save()
                        raise RuntimeError("PROCESS_FAULT_TIMEOUT: " + fault_name) from error
                    finally:
                        stop_group(process)
                if returncode == 0 or (output / "group-complete.json").exists():
                    raise RuntimeError("PROCESS_FAULT_FALSE_COMPLETE: " + fault_name)
                checked = (logs / (name + ".log")).read_text(errors="replace")
                if returncode == 98 or "PROCESS_SCOPE_CLEANUP_FAILED" in checked:
                    raise RuntimeError("PROCESS_FAULT_CLEANUP_FAILED: " + fault_name)
                if returncode == 99 or "PROCESS_SCOPE_FORCED_CLEANUP" in checked:
                    raise RuntimeError("PROCESS_FAULT_FORCED_CLEANUP: " + fault_name)
                if "closing signal SIGTERM" in checked or "Signal 15 (SIGTERM)" in checked:
                    raise RuntimeError("PROCESS_FAULT_LAUNCHER_KILLED_RANK: " + fault_name)
                if expected_error not in checked:
                    raise RuntimeError("PROCESS_FAULT_NOT_REACHED: " + fault_name)
                # The existing subreaper can terminate adopted descendants.
                # Launcher termination alone is not graceful rank termination.
                report[name] = {"status": "SUPERVISED_NO_COMPLETE",
                                "fault_rank": fault_rank, "returncode": returncode,
                                "elapsed_seconds": time.monotonic() - started,
                                "runtime_bounded_exit_proven": False,
                                "supervisor": "existing process_scope subreaper"}
                save()
            report["status"] = "COMPLETE"
            if MODE == "owner_capture_then_lsa_faults":
                report["lsa_fault_status"] = "SUPERVISED_NO_COMPLETE"
            return
        if MODE == "host_sized_only":
            report["scope"] = ("two physical T4; HostSizedNccl only; no LSA or P2P claim; "
                               "full-state oracle and archive fixtures")
            for name in ("library_two_rank_layers_and_archives_match_oracle",
                         "cuco_rank_two_gpu_dense_layers_and_archives_match_oracle"):
                checked = run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                               "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                               name, "--", "--exact", "--nocapture", "--test-threads=1"],
                              "host-sized-" + name, timeout=1800)
                if "test result: ok. 1 passed; 0 failed" not in checked:
                    raise RuntimeError("HOST_SIZED_ORACLE: " + name)
                report[name] = "PASS"
                save()
            report["status"] = "COMPLETE"
            return
        if MODE in ("lsa_leaf_sanitizer", "lsa_leaf_harness_fault",
                    "lsa_leaf_remaining_sanitizers", "lsa_leaf_initcheck_debug"):
            name = "lsa_one_exchange_matches_peer_payload"
            binary = [path for path in (source / "target/debug/deps").glob("library_multi_gpu-*")
                      if path.is_file() and os.access(path, os.X_OK)]
            if len(binary) != 1:
                raise RuntimeError("LSA_LEAF_BINARY_INVENTORY")
            command = [str(binary[0]), name, "--ignored", "--exact", "--nocapture",
                       "--test-threads=1"]
            plain = run(command, "lsa-leaf-plain", timeout=180)
            if "test result: ok. 1 passed; 0 failed" not in plain:
                raise RuntimeError("LSA_LEAF_PLAIN_RESULT")
            report["plain_leaf"] = "PASS"
            save()
            if MODE == "lsa_leaf_harness_fault":
                fault_env = dict(env, MGBFS_TEST_LSA_BAD_PAYLOAD="1")
                try:
                    failed = subprocess.run(command, cwd=source, env=fault_env,
                                            capture_output=True, text=True, timeout=30)
                except subprocess.TimeoutExpired as error:
                    report["fault_result"] = "TIMEOUT"
                    raise RuntimeError("LSA_LEAF_FAULT_HUNG") from error
                fault_output = failed.stdout + failed.stderr
                (logs / "lsa-leaf-bad-payload.log").write_text(fault_output)
                if failed.returncode == 0 or "assertion" not in fault_output:
                    raise RuntimeError("LSA_LEAF_FAULT_NOT_DETECTED")
                report["fault_result"] = "NONZERO_WITH_ASSERTION"
                report["fault_returncode"] = failed.returncode
                report["status"] = "COMPLETE"
                return
            if MODE in ("lsa_leaf_remaining_sanitizers", "lsa_leaf_initcheck_debug"):
                report["leaf_tools"] = {}
                tools = (("initcheck",) if MODE == "lsa_leaf_initcheck_debug"
                         else ("initcheck", "synccheck"))
                if MODE == "lsa_leaf_initcheck_debug":
                    env["NCCL_DEBUG"] = "INFO"
                for tool in tools:
                    try:
                        checked = subprocess.run(
                            ["compute-sanitizer", "--tool", tool,
                             "--report-api-errors", "no", "--error-exitcode", "97",
                             *command], cwd=source, env=env, capture_output=True,
                            text=True, timeout=300)
                        output = checked.stdout + checked.stderr
                        (logs / ("lsa-leaf-" + tool + ".log")).write_text(output)
                        report["leaf_tools"][tool] = {
                            "returncode": checked.returncode,
                            "test_passed": "test result: ok. 1 passed; 0 failed" in output,
                            "zero_errors": "ERROR SUMMARY: 0 errors" in output,
                        }
                    except subprocess.TimeoutExpired:
                        report["leaf_tools"][tool] = {"status": "TIMEOUT"}
                    save()
                if not all(result.get("returncode") == 0 and
                           result.get("test_passed") and result.get("zero_errors")
                           for result in report["leaf_tools"].values()):
                    raise RuntimeError("LSA_LEAF_REMAINING_SANITIZER_FAILURE")
                report["status"] = "COMPLETE"
                return
            report["leaf_tools"] = {}
            for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
                checked = run(["compute-sanitizer", "--tool", tool,
                               "--report-api-errors", "no", "--error-exitcode", "97",
                               *command], "lsa-leaf-" + tool, timeout=300)
                clean_summary = ("RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
                                 if tool == "racecheck" else "ERROR SUMMARY: 0 errors")
                if ("test result: ok. 1 passed; 0 failed" not in checked
                        or clean_summary not in checked):
                    raise RuntimeError("LSA_LEAF_SANITIZER_RESULT: " + tool)
                report["leaf_tools"][tool] = "PASS_API_ERROR_REPORTING_DISABLED"
                save()
            report["status"] = "COMPLETE"
            return
        if MODE == "owner_capacity_gate":
            for name in (
                "cuco_rank_two_gpu_dense_layers_and_archives_match_oracle",
                "cuco_rank_lsa_one_rank_owner_capacity_failure_stops_group",
                "cuco_rank_lsa_one_rank_host_owner_error_stops_group",
            ):
                result = run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                              "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                              name, "--", "--ignored", "--exact", "--nocapture",
                              "--test-threads=1"], name, timeout=300)
                if "test result: ok. 1 passed; 0 failed" not in result:
                    raise RuntimeError("OWNER_CAPACITY_GATE_RESULT: " + name)
                report[name] = "PASS"
                save()
            report["status"] = "COMPLETE"
            return
        if MODE == "production_fault_gate":
            report["scope"] = ("two physical P2P T4; owner capacity, archive slot, "
                               "and device retirement fatal propagation")
            save()
            for name in (
                "cuco_rank_lsa_one_rank_owner_capacity_failure_stops_group",
                "archive_slot_failure_votes_group_fatal_before_lsa_exchange",
                "retirement_fifo_fault_votes_group_fatal_on_two_devices",
            ):
                command = ["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                           "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                           name, "--", "--exact", "--nocapture", "--test-threads=1"]
                if name != "retirement_fifo_fault_votes_group_fatal_on_two_devices":
                    command.insert(-3, "--ignored")
                result = run(command, "production-" + name, timeout=180)
                if "test result: ok. 1 passed; 0 failed" not in result:
                    raise RuntimeError("PRODUCTION_FAULT_GATE_RESULT: " + name)
                report[name] = "PASS"
                save()
            report["status"] = "COMPLETE"
            return
        if MODE == "full_bfs_direct_gate":
            # Real target validation, not debugger/launcher scheduling. Reuse
            # the existing strict oracle/sanitizer parser and rank subreaper.
            sys.path.insert(0, str(source / "scripts"))
            from eight_gpu_gate import run_command, validate_log
            binaries = [p for p in (source / "target/debug/deps").glob("library_multi_gpu-*")
                        if p.is_file() and os.access(p, os.X_OK)]
            if len(binaries) != 1:
                raise RuntimeError("BINARY_INVENTORY")
            binary = binaries[0]
            fixture = "cuco_rank_two_gpu_dense_layers_and_archives_match_oracle"
            report.update(scope="two T4; HOST_SIZED_NCCL DENSE CUCO_RANK; direct full-state/archive oracle and four full-BFS sanitizers; in-process rank threads, not independent processes",
                          binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                          stages=[])
            env["NCCL_DEBUG"] = "INFO"
            for tool in ("plain", "memcheck", "racecheck", "initcheck", "synccheck"):
                command = [str(binary), fixture, "--exact", "--nocapture", "--test-threads=1"]
                if tool != "plain":
                    command = [SANITIZER_PATH, "--tool", tool, "--error-exitcode", "97", *command]
                stage = {"tool": tool, "command": command, "status": "INCOMPLETE"}
                report["stages"].append(stage)
                started = time.monotonic()
                save()
                try:
                    text = run_command(command, logs / ("direct-" + tool + ".log"), env, 180)
                    validate_log(text, tool)
                    stage["status"] = "PASS"
                except TimeoutError as error:
                    stage.update(status="TIMEOUT", error=str(error))
                except Exception as error:
                    stage.update(status="FAIL", error=str(error))
                stage["elapsed_seconds"] = time.monotonic() - started
                save()
                if tool == "plain" and stage["status"] != "PASS":
                    break
            report["status"] = ("PASS" if len(report["stages"]) == 5 and
                                all(stage["status"] == "PASS" for stage in report["stages"])
                                else "VALIDATION_FAILED")
            return
        if MODE == "full_bfs_memcheck_diagnostic":
            report["scope"] = "HOST_SIZED_NCCL U3(3) then S4; one memcheck traced replay; DEBUGGER_PARENT_MI; LSA OFF"
            binaries = [p for p in (source / "target/debug/deps").glob("library_multi_gpu-*") if p.is_file() and os.access(p, os.X_OK)]
            if len(binaries) != 1:
                raise RuntimeError("BINARY_INVENTORY")
            report["diagnostic"] = diagnostic_replay(binaries[0], source, env, logs)
            report["status"] = report["diagnostic"]["status"]
            save()
            return
        if MODE == "device_fatal_gate":
            result = run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                          "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                          "retirement_fifo_fault_votes_group_fatal_on_two_devices",
                          "--", "--exact", "--nocapture", "--test-threads=1"],
                         "device-fatal-gate", timeout=180)
            if "test result: ok. 1 passed; 0 failed" not in result:
                raise RuntimeError("DEVICE_FATAL_GATE_RESULT")
            report["device_fatal_gate"] = "PASS"
            report["status"] = "COMPLETE"
            return
        if MODE == "archive_fault_gate":
            for name in ("archive_slot_failure_votes_group_fatal_before_exchange",
                         "archive_slot_failure_votes_group_fatal_before_lsa_exchange"):
                result = run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                              "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                              name, "--", "--ignored", "--exact", "--nocapture",
                              "--test-threads=1"], name, timeout=180)
                if "test result: ok. 1 passed; 0 failed" not in result:
                    raise RuntimeError("ARCHIVE_FAULT_TEST_RESULT: " + name)
                report[name] = "PASS"
                save()
            report["status"] = "COMPLETE"
            return
        result = run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                      "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                      "cuco_rank_two_gpu_dense_layers_and_archives_match_oracle",
                      "--", "--ignored", "--exact", "--nocapture", "--test-threads=1"],
                     "lsa-full-bfs", timeout=900)
        if "test result: ok. 1 passed; 0 failed" not in result:
            raise RuntimeError("BFS_TEST_RESULT")
        binaries = [path for path in (source / "target/debug/deps").glob("library_multi_gpu-*")
                    if path.is_file() and os.access(path, os.X_OK)]
        if len(binaries) != 1:
            raise RuntimeError("BFS_TEST_BINARY_INVENTORY")
        report["plain_full_bfs"] = "PASS"
        save()
        if MODE == "rounds_gate":
            for test_name in (
                "library_two_rank_layers_and_archives_match_oracle",
                "cuco_rank_two_gpu_dense_layers_and_archives_match_oracle",
                "cuco_rank_lsa_one_rank_owner_capacity_failure_stops_group",
                "cuco_rank_lsa_one_rank_host_owner_error_stops_group",
            ):
                is_fault = "one_rank" in test_name
                result = run([str(binaries[0]), test_name,
                              *(["--ignored"] if is_fault else []),
                              "--exact", "--nocapture", "--test-threads=1"],
                             test_name, timeout=180 if is_fault else 1800)
                if "test result: ok. 1 passed; 0 failed" not in result:
                    raise RuntimeError("ROUND_SCHEDULE_TEST_RESULT: " + test_name)
            report["host_sized_profile_gate"] = "PASS"
            report["status"] = "COMPLETE"
            return
        env["NCCL_DEBUG"] = "INFO"
        sanitized = run(["compute-sanitizer", "--tool", "memcheck",
                         "--target-processes", "application-only",
                         "--report-api-errors", "no",
                         "--kernel-name", "kns=lsa_publish_count",
                         "--kernel-name", "kns=lsa_copy_exact",
                         "--kernel-name", "kns=import_transport_fatal",
                         "--error-exitcode", "99",
                         str(binaries[0]),
                         "cuco_rank_lsa_single_fixture_for_sanitizer",
                         "--ignored", "--exact", "--nocapture", "--test-threads=1"],
                        "lsa-single-fixture-filtered-memcheck", timeout=120)
        if "test result: ok. 1 passed; 0 failed" not in sanitized or \
                "ERROR SUMMARY: 0 errors" not in sanitized:
            raise RuntimeError("BFS_MEMCHECK_RESULT")
        report["full_bfs_memcheck"] = (
            "PASS_FILTERED_TRANSPORT_KERNELS_APPLICATION_ONLY_API_ERRORS_DISABLED")
        report["status"] = "COMPLETE"
    except Exception as error:
        report["error"] = str(error)
        for stage in report.get("stages", []):
            if stage.get("status") == "RUNNING":
                stage.update(status="FAIL", error=str(error))
        raise
    finally:
        save()
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
