"""External rental deadline guard. Deletes only a pre-recorded matching rental.

Deletion is irreversible: retrieve small result files before the deadline.
Requires this Windows host to remain online. Provider/network outages can delay
teardown; this is not a provider-side hard billing cap. No rental creation here.
"""
import argparse
import json
import math
from pathlib import Path
import time


def step(target, now, deadline, api):
    if (any(type(target.get(k)) is not int or target[k] <= 0 for k in ('id', 'machine_id'))
            or not isinstance(target.get('label'), str)
            or not target['label'].startswith('mgbfs-lrx13-')
            or not math.isfinite(now) or not math.isfinite(deadline)):
        raise ValueError('WATCHDOG_TARGET_INVALID')
    response = api('GET', target['id'])
    if not isinstance(response, dict) or 'instances' not in response:
        raise ValueError('WATCHDOG_RESPONSE_INVALID')
    instance = response['instances']
    if instance is None:
        return 'ABSENT'
    if not isinstance(instance, dict) or any(instance.get(k) != target[k] for k in ('id', 'machine_id', 'label')):
        raise ValueError('WATCHDOG_TARGET_MISMATCH')
    if now < deadline:
        return 'WAITING'
    result = api('DELETE', target['id'])
    if not isinstance(result, dict) or result.get('success') is not True:
        raise RuntimeError('WATCHDOG_DELETE_NOT_CONFIRMED')
    # A later GET must confirm absence; DELETE acceptance alone is not completion.
    return 'DELETE_REQUESTED'


def main():
    import requests
    from windows_vast_key import load_key
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lease', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    lease = json.loads(args.lease.read_text(encoding='utf-8'))
    deadline = lease['destroy_at_unix']
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        parser.error('finite absolute deadline required')
    # Monotonic elapsed time prevents a clock rollback extending the rental.
    initial_wall, initial_monotonic = time.time(), time.monotonic()
    headers = {'Authorization':'Bearer '+load_key()}
    def api(method, instance):
        response = requests.request(method, f'https://console.vast.ai/api/v0/instances/{instance}',
                                    headers=headers, timeout=10, allow_redirects=False)
        if response.status_code == 404 and method == 'GET':
            body = response.json()
            if body.get('error') == 'not_found':
                return {'instances':None}
        if 300 <= response.status_code < 400:
            raise RuntimeError('WATCHDOG_REDIRECT_REFUSED')
        response.raise_for_status()
        return response.json()
    import ctypes
    # Prevent idle sleep while armed; user shutdown/network loss remains possible.
    awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not awake:
        raise RuntimeError('WATCHDOG_SLEEP_GUARD_FAILED')
    try:
        with args.log.open('x', encoding='utf-8', buffering=1) as output:
            while True:
                now = max(time.time(), initial_wall+time.monotonic()-initial_monotonic)
                try:
                    status = step(lease, now, deadline, api)
                except ValueError:
                    output.write(json.dumps(dict(at=now, status='IDENTITY_OR_SCHEMA_FAILURE'))+'\n')
                    raise
                except Exception as error:
                    # Do not log raw HTTP bodies, credential headers or URLs.
                    status = 'API_RETRY_' + type(error).__name__
                output.write(json.dumps(dict(at=now, status=status, id=lease['id']))+'\n')
                if status == 'ABSENT':
                    break
                time.sleep(10)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
