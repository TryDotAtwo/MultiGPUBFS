"""Reverify retained v90 archives on Kaggle; no BFS, GPU or local state download."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

SOURCE = '11338625e6bda5b7d22b939bb1668e1f15c7efe0'
ORIGINAL_SOURCE = '749c691363007836969536c8629e9b30bc92e842'
ORIGINAL_SUMMARY_SHA256 = '664e545880881f022f2fe52c88858a87ecaeffa5c84ca84ce1dcee141b7d3cdc'


def main():
    work = Path(tempfile.mkdtemp(prefix='mgbfs-recover-', dir='/tmp'))
    source = work / 'source'
    subprocess.run(['git','clone','--quiet','https://github.com/TryDotAtwo/MultiGPUBFS.git',str(source)], check=True)
    subprocess.run(['git','-C',str(source),'checkout','--detach',SOURCE], check=True)
    actual = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if actual != SOURCE:
        raise RuntimeError('RECOVERY_VERIFIER_SOURCE')
    subprocess.run([sys.executable,'-m','pip','install','--only-binary=:all:',
                    '--no-deps','pyarrow==19.0.1'],check=True)
    sys.path.insert(0,str(source/'scripts'))
    from replay_lsa_cancel_candidate import verify_process_archives, instrumentation_clean
    candidates = []
    for path in Path('/kaggle/input').rglob('summary.json'):
        if path.parent.name != 'lsa-bfs-gate':
            continue
        report = json.loads(path.read_text())
        if report.get('source') == ORIGINAL_SOURCE and len(report.get('typed_runs',[])) == 20:
            candidates.append((path.parent,report))
    if len(candidates) != 1:
        raise RuntimeError('V90_INPUT_INVENTORY')
    root, original = candidates[0]
    if hashlib.sha256((root/'summary.json').read_bytes()).hexdigest() != ORIGINAL_SUMMARY_SHA256:
        raise RuntimeError('RECOVERY_ORIGINAL_SUMMARY_DIGEST')
    output = Path('/kaggle/working/v90-recovery')
    output.mkdir(parents=True,exist_ok=True)
    recovered = dict(scope='Retained archive verification, not a new BFS or reconstructed exit-code report',
        original_source=ORIGINAL_SOURCE, verifier_source=actual,
        original_summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
        status='INCOMPLETE', cases=[])
    def save():
        (output/'summary.json').write_text(json.dumps(recovered,indent=2))
    save()
    for prior in original['typed_runs']:
        label = prior['label']
        case = root/label/'healthy-None'
        row = dict(label=label,original_replay_pass=prior['pass'],artifact_verified=False)
        try:
            config_bytes = (root/(label+'-config.json')).read_bytes()
            if hashlib.sha256(config_bytes).hexdigest() != prior['run_config_sha256']:
                raise ValueError('RECOVERY_CONFIG_SNAPSHOT')
            config = json.loads(config_bytes)
            detail = json.loads((root/label/'summary.json').read_bytes())
            if detail['base_commit'] != ORIGINAL_SOURCE:
                raise ValueError('RECOVERY_RUNTIME_SOURCE')
            row['full_state_oracle'] = verify_process_archives(case,n=4,
                modulus=2 if prior['tool']=='nsys' else None,
                expected_seed=f"{int.from_bytes(bytes(config['seed']),'little'):032x}",
                expected_epoch_window=3,expected_config_digest=detail['expected_config_digest'],
                expected_run_contract='RunConfigV1',expected_route_banks=prior['route_banks'],
                require_bank_reuse=prior['tool']=='nsys')
            if prior['tool']=='initcheck':
                if not all(instrumentation_clean((case/f'rank-{rank}.log').read_text(errors='replace'),'initcheck')
                           for rank in (0,1)):
                    raise ValueError('RECOVERY_SANITIZER_ERRORS')
            row['artifact_verified'] = True
            row['raw_rank_timelines_retained'] = [str(case/f'rank-{rank}.nsys-rep')
                for rank in (0,1) if (case/f'rank-{rank}.nsys-rep').exists()]
        except Exception as error:
            row['error'] = str(error)
        recovered['cases'].append(row)
        save()
        print(json.dumps(row),flush=True)
    recovered['status'] = 'RECOVERY_PROCESSED_WITH_OPEN_FAILURES'
    save()


if __name__ == '__main__':
    main()
