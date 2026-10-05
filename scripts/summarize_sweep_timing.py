"""Aggregate measured pair timing without treating failed prefixes as full searches."""
from collections import Counter
import argparse,json
from pathlib import Path


def summarize(ledger):
    cases=list(ledger.get('cases',{}).values())
    attempted=[x for x in cases if x.get('attempted')]
    complete=[x for x in attempted if x.get('status')=='COMPLETE']
    by_backend={}
    for record in attempted:
        timing=record.get('timing',{})
        backend=timing.get('execution_backend','UNKNOWN')
        group=by_backend.setdefault(backend,dict(pairs=0,complete_pairs=0,runner_seconds=0.0,
            transition_seconds=0.0,complete_runner_seconds=0.0,complete_search_seconds=0.0,
            complete_pairs_with_missing_search_timing=0,incomplete_committed_layer_seconds=0.0))
        group['pairs']+=1
        group['runner_seconds']+=timing.get('runner_wall_seconds',0.0) or 0.0
        group['transition_seconds']+=timing.get('transition_before_seconds',0.0) or 0.0
        if record.get('status')!='COMPLETE':
            group['incomplete_committed_layer_seconds']+=timing.get('completed_layer_seconds',0.0) or 0.0
            continue
        group['complete_pairs']+=1
        runs=record.get('comparison',{}).get('runs')
        if runs:
            measured=[x.get('production_search_seconds') for x in runs]
        else:measured=[timing.get('production_search_seconds')]
        if any(x is None for x in measured):
            group['complete_pairs_with_missing_search_timing']+=1
            continue
        group['complete_runner_seconds']+=timing.get('runner_wall_seconds',0.0) or 0.0
        group['complete_search_seconds']+=sum(measured)
    for group in by_backend.values():
        group['complete_nonsearch_runner_seconds']=group['complete_runner_seconds']-group['complete_search_seconds']
        group['timed_complete_pairs']=group['complete_pairs']-group['complete_pairs_with_missing_search_timing']
    return dict(classified_pairs=len(cases),statuses=dict(Counter(x.get('status','UNKNOWN') for x in cases)),
        attempted_pairs=len(attempted),complete_pairs=len(complete),
        resource_or_other_incomplete_attempted_pairs=len(attempted)-len(complete),
        skipped_pairs=len(cases)-len(attempted),by_backend=by_backend,
        scopes=dict(complete_search='sum all measured seeds, complete graphs only; includes native archive backpressure',
            complete_nonsearch_runner='runner minus all measured complete search seeds; admission, setup, local save and comparison, excludes separate transition',
            incomplete_committed_layer='committed first-run prefix only; excludes failed-layer work and cleanup; not a complete search time',
            transition='sum recorded transition_before_seconds on attempted pairs; unrecorded pruning/publication time is not inferred',
            publication='not included; obtain upload and verification timing from automatic report'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('ledger',type=Path)
    args=parser.parse_args();print(json.dumps(summarize(json.loads(args.ledger.read_text(encoding='utf-8'))),indent=2))

if __name__=='__main__':main()
