"""One scoped, authorized two-GPU attempt; never retries instance creation."""
import json
import time
from pathlib import Path
import requests
from windows_vast_key import load_key


def main():
    root = Path(__file__).resolve().parents[1] / 'test_results/tail-vast-20261002'
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'lease.json').exists():
        raise RuntimeError('existing lease; do not duplicate')
    session = requests.Session()
    session.headers['Authorization'] = 'Bearer ' + load_key()
    base = 'https://console.vast.ai/api/v0/'
    query = dict(num_gpus={'eq': 2}, gpu_name={'eq': 'RTX 4060 Ti'},
                 rentable={'eq': True}, verified={'eq': True},
                 reliability={'gte': .99}, disk_space={'gte': 100},
                 direct_port_count={'gte': 1}, type='ondemand', limit=30)
    response = session.post(base + 'bundles/', json=query, timeout=30)
    response.raise_for_status()
    offers = response.json()['offers']
    offer = min(offers, key=lambda o: o['dph_total'] + o['storage_cost'] * 100 / 720)
    hourly = offer['dph_total'] + offer['storage_cost'] * 100 / 720
    if hourly > .30 or offer['cpu_ram'] < 32000 or offer['gpu_ram'] < 8000:
        raise RuntimeError('quote outside configured bounds')
    quote = {k: offer.get(k) for k in ('id', 'machine_id', 'gpu_name', 'num_gpus',
        'gpu_ram', 'gpu_total_ram', 'cpu_ram', 'dph_total', 'storage_cost',
        'inet_up_cost', 'inet_down_cost', 'reliability')}
    quote.update(disk_gb=100, estimated_hourly_usd=hourly, cap_usd=10,
                 deadline_hours=2, transfer_budget_usd=2, reserve_usd=5)
    (root / 'quote.json').write_text(json.dumps(quote, indent=2))
    if max(offer['inet_up_cost'], offer['inet_down_cost']) * 100 > 2:
        raise RuntimeError('traffic quote exceeds allowance')
    label = 'mgbfs-lrx13-tail-test-20261002'
    response = session.get(base + 'instances/', timeout=20)
    response.raise_for_status()
    if any(i.get('label') == label for i in response.json()['instances']):
        raise RuntimeError('existing labeled rental')
    started = time.time()
    # Save intent before request so an ambiguous response is recoverable by label.
    (root / 'creation-intent.json').write_text(json.dumps(dict(label=label, offer=quote)))
    response = session.put(base + f"asks/{offer['id']}/", json=dict(
        image='vastai/pytorch:@vastai-automatic-tag', disk=100,
        runtype='ssh_direct', label=label, cancel_unavail=True), timeout=45)
    response.raise_for_status()
    body = response.json()
    if body.get('success') is not True:
        raise RuntimeError('rental not confirmed')
    lease = dict(id=int(body['new_contract']), machine_id=offer['machine_id'],
                 label=label, start_date=started, destroy_at_unix=started+7200,
                 work_deadline_unix=started+6900, total_project_cap_usd=10,
                 hourly_usd_upper=.30, traffic_budget_usd=2,
                 purpose='isolated tail archive and CUCO comparison')
    (root / 'lease.json').write_text(json.dumps(lease, indent=2))
    print(json.dumps(dict(lease=lease, quote=quote), indent=2))


if __name__ == '__main__':
    main()
