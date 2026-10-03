"""Release GPU-host state files only after immutable HF readback verification."""
import json
import time
from pathlib import Path

try:
    from .bfs_tail_archive import atomic_json
    from .tail_cohort import case_groups, publication_cases
    from .verify_tail_hf import verify_payload
except ImportError:
    from bfs_tail_archive import atomic_json
    from tail_cohort import case_groups, publication_cases
    from verify_tail_hf import verify_payload


def release(root, ledger, repo_id, api, token, *, deadline=None,
            group_size=20, group_bytes=2_000_000_000):
    import requests
    from huggingface_hub import hf_hub_url
    root = Path(root).resolve()
    revision = api.repo_info(repo_id, repo_type='dataset').sha
    publications = publication_cases(root, ledger, group_size=group_size,
                                     group_bytes=group_bytes)
    run_id = ledger['configuration']['base']['run_id']
    groups = list(case_groups(root, ledger, group_size=group_size, group_bytes=group_bytes))
    released = []
    with requests.Session() as client:
        client.headers['Authorization'] = 'Bearer '+token
        for members, sealed in groups:
            if not sealed:
                continue
            destinations = {publications[key][0] for key, _ in members}
            for destination in destinations:
                destination = destination.resolve()
                if not destination.is_relative_to(root/'cohorts'):
                    raise ValueError('release destination escapes cohort root')
                marker = destination/'release-receipt.json'
                selected = [(key, saved) for key, saved in members
                            if publications[key][0].resolve() == destination]
                files, verified = {}, {}
                for key, _ in selected:
                    local = json.loads(publications[key][1].read_text())
                    remote = f'tail-runs/{run_id}-{key}/manifest.json'
                    if deadline is not None and time.time() >= deadline:
                        raise TimeoutError('HF release verification deadline; inputs retained')
                    with client.get(hf_hub_url(repo_id, remote, repo_type='dataset',
                                    revision=revision), timeout=(30,120)) as response:
                        response.raise_for_status()
                        if response.json() != local:
                            raise ValueError('HF release manifest differs; inputs retained')
                    for entry in local['files']:
                        remote = entry['repo_path']
                        if remote in files and files[remote] != entry:
                            # Shared entries have distinct case spans but equal physical identity.
                            other = files[remote]
                            if (other['bytes'], other['sha256']) != (entry['bytes'], entry['sha256']):
                                raise ValueError('conflicting shared release payload')
                        files[remote] = entry
                previous = json.loads(marker.read_text()) if marker.exists() else {}
                for remote, entry in files.items():
                    signature = dict(bytes=entry['bytes'], sha256=entry['sha256'])
                    if (previous.get('repo_id') == repo_id and previous.get('revision')
                            and previous.get('verified_payloads', {}).get(remote) == signature):
                        verified[remote] = signature
                        continue
                    if deadline is not None and time.time() >= deadline:
                        raise TimeoutError('HF release verification deadline; inputs retained')
                    with client.get(hf_hub_url(repo_id, remote, repo_type='dataset',
                                    revision=revision), stream=True, timeout=(30,120)) as response:
                        verify_payload(response, entry)
                    verified[remote] = signature
                # Durable receipt precedes every unlink. A failed or interrupted readback
                # cannot release inputs; interrupted unlink is safely repeatable.
                atomic_json(marker, dict(repo_id=repo_id, revision=revision,
                    verified_payloads=verified, cases=[key for key, _ in selected]))
                for entry in files.values():
                    payload = (destination/entry['path']).resolve()
                    if not payload.is_relative_to(destination) or payload.suffix != '.parquet':
                        raise ValueError('unsafe release payload path')
                    payload.unlink(missing_ok=True)
                for key, saved in selected:
                    saved = saved.resolve()
                    if saved != root/key/'saved':
                        raise ValueError('unsafe saved-state release root')
                    # Only state payloads are removed. Manifests, native reports, timings,
                    # configuration and all provenance remain available locally.
                    for payload in saved.rglob('*.bin'):
                        if not payload.resolve().is_relative_to(saved):
                            raise ValueError('unsafe saved-state release payload')
                        payload.unlink()
                    released.append(key)
    return dict(revision=revision, released_cases=released)
