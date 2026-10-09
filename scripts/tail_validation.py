"""Search validation is independent of snapshot/publication byte integrity."""
def validation_status(record, two_seeds=True):
    if record.get('runner_error') or any(x.get('runner_error') for x in record.get('replicas', [])):
        return 'FAILED'
    if not two_seeds:
        return 'NOT_VALIDATED'
    comparison = record.get('comparison', {})
    if comparison.get('matched') is False:
        return 'FAILED'
    if comparison.get('matched') is True:
        return 'VERIFIED_LAYER_COUNTS'
    return 'NOT_VALIDATED'

def validation_summary(ledger):
    records = [x for x in ledger['cases'].values() if x.get('attempted')]
    two_seeds = ledger.get('configuration', {}).get('base', {}).get('two_seeds', False)
    statuses = [validation_status(x, two_seeds) for x in records]
    complete = sum(x.get('status') == 'COMPLETE' and s == 'VERIFIED_LAYER_COUNTS'
                   for x, s in zip(records, statuses))
    return dict(validation_status=('FAILED' if 'FAILED' in statuses else
                'VERIFIED_LAYER_COUNTS' if statuses and all(s == 'VERIFIED_LAYER_COUNTS' for s in statuses)
                else 'NOT_VALIDATED'), complete=complete,
                search_complete=sum(x.get('status') == 'COMPLETE' for x in records),
                validation_failed=statuses.count('FAILED'))
