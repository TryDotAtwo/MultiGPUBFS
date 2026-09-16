"""Conservative quote arithmetic; NOT a billing watchdog or rental authorization.

Transfer byte budgets must include build downloads, archives, metadata and all
retries. Hourly prices cover the whole instance, not one GPU. Time starts at
billing start, not BFS start. Unknown fees must be included in reserve_usd.
"""
import argparse
import json
from fractions import Fraction
from pathlib import Path


def plan_budget(quote):
    prices = {}
    for key in ('cap_usd', 'spent_usd', 'reserve_usd', 'compute_usd_hour',
                'storage_usd_hour', 'ingress_usd_unit', 'egress_usd_unit'):
        value = quote.get(key)
        if type(value) is not str:
            raise ValueError('RENTAL_PRICE_MUST_BE_DECIMAL_STRING: ' + key)
        try:
            prices[key] = Fraction(value)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError('RENTAL_PRICE_INVALID: ' + key) from error
        if prices[key] < 0:
            raise ValueError('RENTAL_PRICE_NEGATIVE: ' + key)
    for key in ('transfer_unit_bytes', 'ingress_bytes', 'egress_bytes',
                'billing_quantum_seconds', 'teardown_seconds'):
        if type(quote.get(key)) is not int or quote[key] < 0:
            raise ValueError('RENTAL_INTEGER_INVALID: ' + key)
    if min(quote['transfer_unit_bytes'], quote['billing_quantum_seconds'],
           quote['teardown_seconds']) == 0:
        raise ValueError('RENTAL_UNIT_OR_TEARDOWN_ZERO')
    hourly = prices['compute_usd_hour'] + prices['storage_usd_hour']
    if hourly <= 0 or prices['cap_usd'] <= 0:
        raise ValueError('RENTAL_RATE_OR_CAP_ZERO')
    transfer = (quote['ingress_bytes'] * prices['ingress_usd_unit'] +
                quote['egress_bytes'] * prices['egress_usd_unit']) / quote['transfer_unit_bytes']
    available = prices['cap_usd'] - prices['spent_usd'] - prices['reserve_usd'] - transfer
    quantum = quote['billing_quantum_seconds']
    seconds = int((available * 3600 / hourly) // quantum) * quantum
    work = seconds - quote['teardown_seconds']
    if work <= 0:
        raise ValueError('RENTAL_BUDGET_EXHAUSTED')
    return dict(max_billed_seconds=seconds, work_deadline_seconds=work,
                teardown_seconds=quote['teardown_seconds'],
                transfer_budget_usd=str(transfer),
                hourly_usd=str(hourly), enforces_billing_stop=False,
                cost_representation='exact rational USD',
                deadline_origin='instance billing start')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('quote', type=Path)
    args = parser.parse_args()
    print(json.dumps(plan_budget(json.loads(args.quote.read_text(encoding='utf-8'))), indent=2))


if __name__ == '__main__':
    main()
