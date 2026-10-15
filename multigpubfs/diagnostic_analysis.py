"""Evidence-referenced synthesis. Observations, hypotheses and unknowns remain distinct."""
import errno,re

def v1_device_allows(text,major,minor,access='rw'):
 allowed=set()
 for line in text.splitlines():
  fields=line.split()
  if len(fields)!=3 or ':' not in fields[1]:return None
  kind,device,modes=fields;ma,mi=device.split(':',1)
  if kind not in ('a','c','b') or any(c not in 'rwm' for c in modes):return None
  if kind in ('a','c') and ma in ('*',str(major)) and mi in ('*',str(minor)):allowed.update(modes)
 return set(access)<=allowed

def analyze(report):
 steps=report.get('steps',[]);by={s['name']:s for s in steps}
 def data(name):
  step=by.get(name,{})
  return step.get('result',{}).get('data') or next(iter(step.get('partial_evidence',{}).values()),{})
 confirmed=[];hypotheses=[];actions=[];unknown=[];excluded=[]
 def fact(code,text,evidence):confirmed.append({'code':code,'statement':text,'evidence':evidence})
 def hypothesis(code,text,evidence):hypotheses.append({'code':code,'statement':text,'evidence':evidence})
 driver=data('driver-api');env=data('environment-evidence');nvml=data('nvml-evidence');torch=data('torch-evidence');matrix=data('loader-contrasts');classification=driver.get('classification')
 if classification and driver.get('status')=='FAIL':fact(classification,driver.get('summary','Driver probe failed'),['driver-api/result/data/classification','driver-api/result/data/driver','driver-api/result/data/runtime'])
 for item in env.get('device_nodes',[]):
  opened=item.get('open',{})
  if opened.get('status')=='FAIL' and opened.get('errno') in (errno.EACCES,errno.EPERM):
   fact('DEVICE_OPEN_DENIED','Read/write open of '+item['path']+' was denied; exact policy mechanism is not established.',['environment-evidence/result/data/device_nodes'])
   if driver.get('status')=='FAIL':hypothesis('GPU_ACCESS_RESTRICTION','Device permissions, cgroup or LSM policy may contribute to failed initialization.',['environment-evidence/result/data/device_nodes','driver-api'])
 for group in env.get('cgroups',{}).get('records',[]):
  rules=group.get('files',{}).get('devices.list')
  if group.get('version')==1 and rules is not None:
   for node in env.get('device_nodes',[]):
    if node.get('character_device') and v1_device_allows(rules,node['major'],node['minor']) is False:
     fact('V1_DEVICE_RULE_NOT_GRANTED','Recorded device rules do not grant read/write access to '+node['path']+'. Actual open evidence is reported separately.',['environment-evidence/result/data/cgroups','environment-evidence/result/data/device_nodes'])
 security=env.get('process_security',{})
 if security.get('Seccomp') not in (None,'0'):unknown.append('Seccomp is active; this alone does not prove it blocked CUDA.')
 if any(x.get('version')==2 for x in env.get('cgroups',{}).get('records',[])):unknown.append('cgroup-v2 device BPF policy is not exposed by devices.list; no permit/deny claim is made from file absence.')
 visibility=env.get('environment',{})
 if visibility.get('CUDA_VISIBLE_DEVICES') in ('','-1'):
  fact('CUDA_VISIBILITY_DISABLED','CUDA visibility explicitly disables devices in this process.',['environment-evidence/result/data/environment/CUDA_VISIBLE_DEVICES']);actions.append('Resubmit through the authorized GPU-job selection; do not bypass the allocation visibility mask.')
 if visibility.get('NVIDIA_VISIBLE_DEVICES') in ('none','void',''):
  fact('CONTAINER_VISIBILITY_DISABLED','Container NVIDIA visibility is configured as disabled. Actual allocation must be verified separately.',['environment-evidence/result/data/environment/NVIDIA_VISIBLE_DEVICES'])
 capabilities=visibility.get('NVIDIA_DRIVER_CAPABILITIES')
 if capabilities and 'all' not in capabilities.split(',') and 'compute' not in capabilities.split(','):
  hypothesis('CONTAINER_COMPUTE_NOT_REQUESTED','Container capability configuration omits compute; driver compute-library exposure may be missing.',['environment-evidence/result/data/environment/NVIDIA_DRIVER_CAPABILITIES'])
 if nvml.get('status')=='PASS':
  loaded=driver.get('driver',{}).get('resolved_library','') or '';version=re.search(r'libcuda\.so\.(\d{3}\.\d+(?:\.\d+)?)',loaded)
  if version and nvml.get('driver_version') and version.group(1)!=nvml['driver_version']:
   fact('DRIVER_LIBRARY_MANAGEMENT_VERSION_DIFFERENCE','Loaded driver library filename and NVML-reported driver versions differ. This can be valid forward compatibility; API success/failure determines whether it matters.',['driver-api/result/data/driver/resolved_library','nvml-evidence/result/data/driver_version'])
 if nvml.get('status')=='PASS':
  fact('NVML_MANAGEMENT_AVAILABLE','NVML initialization works; management count is not proof of CUDA access.',['nvml-evidence/result/data'])
  if driver.get('status')=='FAIL':hypothesis('CUDA_SPECIFIC_FAILURE','Management interface works while CUDA failed; investigate the CUDA library/runtime path and access conditions.',['nvml-evidence','driver-api'])
 if torch.get('status')=='PASS' and driver.get('status')=='FAIL':
  fact('TORCH_CUDA_WORKS_NATIVE_PROBE_FAILS','Installed Torch initialized CUDA in its original client environment while the native-environment probe failed.',['torch-evidence/result/data','driver-api'])
  hypothesis('STACK_ENVIRONMENT_DIFFERENCE','Runtime/build or loader differences between Torch and native are a likely source; neither driver age nor absent GPU is proved by this difference alone.',['torch-evidence','driver-api']);actions.append('Compare the recorded Torch CUDA build and the native runtime/library paths; retain the native baseline failure and contrast results.')
 variants=matrix.get('results',[]);recheck=next((x for x in variants if x['name']=='baseline-recheck'),None)
 good=[x for x in variants if x['status']=='PASS' and x['name']!='baseline-recheck']
 if good and recheck and recheck['status']=='FAIL':
  fact('LOADER_CONTRAST_REPRODUCED','An isolated loader change made driver and native discovery pass, while the original environment failed again. This proves a discovery contrast, not full BFS repair.',['loader-contrasts/result/data/results'])
  actions.append('Use the successful contrast as a candidate user-level loader configuration for the same authorized GPU allocation; validate the regular kernel/NCCL tests before adopting it.')
 elif good:
  hypothesis('TRANSIENT_OR_LOADER_FAILURE','A variant passed but baseline failure was not reproduced; loader causality remains unconfirmed.',['loader-contrasts/result/data/results'])
 if by.get('native-runtime',{}).get('status')=='PASS':excluded.append({'statement':'The native binary executed and packaged integrity/dependency checks passed. An execute-permission or basic native-loader failure does not explain subsequently reached native calls; later JIT or library-specific initialization failures remain possible.','evidence':['native-runtime']})
 runtime=data('native-runtime')
 if runtime.get('binary_executed'):
  fact('NATIVE_EXECUTED_BEFORE_INSPECTION_FAILURE','The native binary executed key-info; subsequent dependency inspection is a separate failure.',['native-runtime/partial_evidence/runtime.json'])
 if 'noexec' in env.get('native_executable_mount',{}).get('options',[]) and by.get('native-runtime',{}).get('result',{}).get('exception_type')=='PermissionError':
  fact('NATIVE_EXEC_ON_NOEXEC_MOUNT','Native execution was denied and its resolved mount is marked noexec.',['environment-evidence/result/data/native_executable_mount','native-runtime'])
 shm=env.get('filesystems',{}).get('/dev/shm',{})
 if shm.get('error'):
  fact('SHARED_MEMORY_WRITE_CHECK_FAILED','The /dev/shm write/read test failed; CUDA discovery and actual NCCL results are evaluated independently.',['environment-evidence/result/data/filesystems'])
  if any(name.startswith('nccl') and value.get('status') in ('FAIL','TIMEOUT') for name,value in by.items()):hypothesis('NCCL_IPC_RESTRICTION','Shared-memory restrictions may contribute to the observed NCCL failure; transport and native logs are required for attribution.',['environment-evidence/result/data/filesystems','nccl-init','nccl-exchange'])
 if by.get('filesystem',{}).get('status')=='PASS':excluded.append({'statement':'The tested atomic file-exchange paths worked; this does not verify every filesystem or IPC mechanism.','evidence':['filesystem']})
 for name,step in by.items():
  if step.get('status')=='SKIPPED':unknown.append(name+': '+step.get('reason','not tested'))
  if step.get('status')=='TIMEOUT':unknown.append(name+' timed out at '+step.get('last_phase','unknown phase')+'; preserved partial evidence is not acceptance.')
 if by.get('cuda-discovery',{}).get('status') in ('FAIL','TIMEOUT'):
  excluded.append({'statement':'BFS and NCCL were not reached; their performance, graph capacity and dedup cannot be blamed for this startup observation.','evidence':['cuda-discovery']})
 if env.get('loopback',{}).get('status')=='FAIL':
  fact('LOOPBACK_TEST_FAILED','The bounded local TCP bind/connect failed. This may affect rendezvous; it does not explain a CUDA discovery error by itself.',['environment-evidence/result/data/loopback'])
 for name in ('nvml-evidence','torch-evidence','environment-evidence','loader-contrasts','native-loader-trace'):
  step=by.get(name,{})
  if step.get('status') in ('WARNING','FAIL','TIMEOUT'):unknown.append(name+' unavailable or failed; inspect its own error/log, not a guessed cause.')
 for step in steps:
  if step.get('errno') in (errno.EAGAIN,errno.EACCES,errno.EPERM,errno.ENOMEM,errno.EMFILE):
   fact('DIAGNOSTIC_PROCESS_SPAWN_FAILED',step['name']+' could not create its diagnostic process: '+step.get('error','')+'. Parent resource limits are retained; exact restriction may remain unknown.',['steps/'+step['name'],'parent_resource_limits'])
  messages=step.get('log_tail','')+' '+step.get('result',{}).get('error','')
  lower=messages.lower()
  if step.get('status')=='FAIL' and ('GLIBC_' in messages or 'GLIBCXX_' in messages or 'CXXABI_' in messages) and 'not found' in lower:
   fact('ABI_VERSION_UNAVAILABLE','The loader reports an unavailable required system/compiler ABI version. This is a library/binary compatibility failure, not proof of absent GPU hardware.',['steps/'+step['name']]);actions.append('Use a binary built for the observed Linux ABI or the correct existing compiler-library module. Preserve the exact missing version and loaded-library paths.')
  if step.get('status')=='FAIL' and 'cannot open shared object file' in lower:
   fact('SHARED_LIBRARY_LOAD_FAILED','A required shared library could not be opened; inspect the exact library name and loader trace, not an assumed CUDA-driver age.',['steps/'+step['name']])
  if step.get('status')=='FAIL' and 'no kernel image' in messages.lower():
   fact('NATIVE_KERNEL_IMAGE_UNAVAILABLE','The CUDA kernel launch reports no executable image for this GPU. Driver discovery alone is not sufficient; inspect actual GPU compute capability and the declared build targets.',['steps/'+step['name'],'driver-api/result/data/driver/devices','environment-evidence/result/data/native_build_declaration']);actions.append('Install a build targeting the observed GPU architecture; do not equate this kernel-image error with missing GPU hardware.')
  if step.get('exit_code')==-9:
   hypothesis('PROCESS_KILLED_NOT_GPU_PROOF',step['name']+' was killed with SIGKILL. OOM, job-controller termination or policy enforcement are candidates; the signal alone does not establish one cause.',['steps/'+step['name'],'environment-evidence/result/data/cgroups'])
 if driver.get('driver',{}).get('initialization',{}).get('code')==802:
  hypothesis('SYSTEM_OR_FABRIC_NOT_READY','Driver API reports system-not-ready. Fabric Manager state is recorded where readable, but whether this hardware needs it and why initialization failed remain unresolved.',['driver-api/result/data/driver/initialization','environment-evidence/result/data/fabric_manager'])
 for case in variants:
  if case['status'] in ('TIMEOUT','SKIPPED','INCOMPLETE'):unknown.append('Loader contrast '+case['name']+': '+case.get('reason',case.get('error','not completed')))
 failures=[x['name'] for x in steps if x['status'] in ('FAIL','TIMEOUT')]
 if failures and not actions:actions.append('Return report.json and diagnostics.zip from the same job. Use the evidence paths and failure-specific library versions to choose a user-level environment fix; do not blindly increase timeouts or install a driver with pip.')
 builds=report.get('deep_analysis',{}).get('runtime_build_matrix',{})
 for comparison in builds.get('comparisons',[]):
  if comparison['outcome']=='VARIANT_DEPENDENT':
   fact('DIAGNOSTIC_BUILD_CONTRAST_OBSERVED','GPU '+str(comparison['device'])+' '+comparison['path']+' passes for some diagnostic builds and fails for others. Runtime/SM120 controlled pairs are recorded; this alone does not prove a unique cause.',['deep_analysis/runtime_build_matrix/comparisons'])
   actions.append('Compare the same-source runtime and native-SM120 pairs in deep-summary.txt; use actual library hashes and call boundaries, with the inherited JIT-cache limitation.')
  for identifier,observed in comparison['variants'].items():
   if observed.get('call_boundary'):
    fact('NATIVE_CALL_BOUNDARY_RECORDED',identifier+' GPU '+str(comparison['device'])+' '+comparison['path']+': '+observed['call_boundary']+'. This locates the last flushed boundary rather than proving a driver root cause.',['steps/'+observed['name']+'/last_native_phase'])
 working=report.get('working_configuration',{})
 if working.get('status')=='READY':
  scope='single-GPU fallback; other requested GPUs are not accepted' if working.get('single_gpu_fallback') else 'selected GPU set '+str(working['selected']['devices'])
  actions.insert(0,'Verified native BFS configuration: '+scope+'. Recheck with: '+working['command'])
  if working.get('calculation_example'):actions.insert(1,'Calculation example; choose your graph parameters and a new output directory: '+working['calculation_example'])
 elif working:actions.append('No ready server command: '+working.get('reason',working['status'])+'. Keep this diagnostic bundle; no unverified configuration is recommended.')
 priority=('DIAGNOSTIC_BUILD_CONTRAST_OBSERVED','LOADER_CONTRAST_REPRODUCED','DRIVER_STUB_LOADED','DRIVER_KERNEL_LIBRARY_MISMATCH','DRIVER_MAJOR_TOO_OLD','CUDA_VISIBILITY_DISABLED','DRIVER_LIBRARY_UNAVAILABLE','NO_VISIBLE_GPU','NATIVE_KERNEL_IMAGE_UNAVAILABLE','DIAGNOSTIC_PROCESS_SPAWN_FAILED','ABI_VERSION_UNAVAILABLE','SHARED_LIBRARY_LOAD_FAILED','RUNTIME_DRIVER_COMPATIBILITY_FAILED','NATIVE_CALL_BOUNDARY_RECORDED')
 best=next((item for code in priority for item in confirmed if item['code']==code),None) if failures else None
 explanation={'confidence':'observed','statement':best['statement'],'evidence':best['evidence']} if best else {'confidence':'unresolved' if failures else 'tested-required-components-passed','statement':'No single cause is established; inspect recorded hypotheses and missing evidence.' if failures else 'Required checks reported no failure. Optional evidence and untested policies remain explicitly limited.','evidence':failures}
 return {'schema':1,'best_supported_explanation':explanation,'primary_failure':failures[0] if failures else None,'confirmed_findings':confirmed,'hypotheses':hypotheses,'ruled_out_for_this_observation':excluded,'unknowns':unknown,'user_next_actions':actions,'operator_boundary':'Kernel-driver replacement, kernel errors and enforced GPU permissions cannot be repaired by a user-space library. If evidence stops at inaccessible policy, that uncertainty remains explicit.'}

def render(analysis):
 lines=['Evidence analysis; primary failure: '+str(analysis['primary_failure']),'BEST '+analysis['best_supported_explanation']['confidence']+': '+analysis['best_supported_explanation']['statement']]
 for key,label in (('confirmed_findings','OBSERVED'),('hypotheses','HYPOTHESIS')):
  for item in analysis[key]:lines.append(label+' '+item['code']+': '+item['statement']+' [evidence: '+', '.join(item['evidence'])+']')
 for item in analysis['ruled_out_for_this_observation']:lines.append('EXCLUDED '+item['statement'])
 for text in analysis['unknowns']:lines.append('UNKNOWN '+text)
 for text in analysis['user_next_actions']:lines.append('NEXT '+text)
 return '\n'.join(lines)+'\n'
