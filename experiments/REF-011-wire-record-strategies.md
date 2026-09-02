# REF-011: eager versus two-phase wire records

Date: 2026-08-27  
Status: mixed result; universal two-phase hypothesis rejected

## Question

Should a distributed exact BFS send candidate key and parent metadata eagerly,
or first exchange keys and return metadata only for accepted remote states?

## Protocols modeled

Let:

- `R` be remote candidates remaining after per-source-rank dedup;
- `A` be newly accepted states with no local producer at their owner;
- `B` be acceptance bitmap bytes, padded separately for every nonempty
  source/destination buffer in every round;
- `K` be candidate-key bytes;
- `M` be parent-reference plus move bytes.

Payload bytes are:

```text
eager     = R * (K + M)
two-phase = R * K + B + A * M
```

Thus two-phase wins exactly when:

```text
B < (R - A) * M
```

It saves metadata for rejected or redundant remote records, but adds an
acceptance bitmap and another dependent communication phase. Transport headers,
alignment imposed by a specific API, latency, and bandwidth effects are not in
this byte-only model.

The source retains its candidate/parent buffer until receiving a bitmap. The
owner uses a locality-aware deterministic tie-break: if a candidate has a local
producer, that parent is preferred; otherwise the lowest source rank wins.

## Formats

Four illustrative payload layouts were swept:

| name | key | parent | move | eager record |
|---|---:|---:|---:|---:|
| packed rank16 | 2 B | 2 B | 1 B | 5 B |
| packed rank32 | 4 B | 4 B | 1 B | 9 B |
| aligned rank64 | 8 B | 8 B | 8 B | 24 B |
| state128/parent128 | 16 B | 16 B | 4 B | 36 B |

These are model points, not claims about an existing wire ABI.

## S8 results

The REF-010 searches were repeated for depths 2, 8, 14, 20, and 28, direct and
mixed ownership, and `P=2,4,8`. This produced 120 format/configuration rows.

Selected eight-rank results:

| depth | owner | format | eager | two-phase | byte reduction |
|---:|---|---|---:|---:|---:|
| 14 | direct | rank16 | 40,075 | 18,591 | 53.61% |
| 14 | mixed | rank16 | 64,440 | 37,060 | 42.49% |
| 14 | direct | state128 | 288,540 | 138,655 | 51.95% |
| 14 | mixed | state128 | 463,968 | 270,583 | 41.68% |
| 28 | direct | rank16 | 616,940 | 269,060 | 56.39% |
| 28 | mixed | rank16 | 943,075 | 488,782 | 48.17% |
| 28 | direct | state128 | 4,441,968 | 2,032,906 | 54.23% |
| 28 | mixed | state128 | 6,790,140 | 3,624,228 | 46.63% |

Direct ownership benefits twice: it creates fewer remote records and accepted
states are more likely to have a local producer. At depth 28/P8, only 2,142 of
123,388 direct remote records (1.74%) need deferred metadata, versus 29,108 of
188,615 mixed records (15.43%).

Combining direct ownership and two-phase transfer reduced the modeled payload
by about 70.06% for state128 at depth 28 relative to mixed ownership with eager
records. This combines two independent choices and does not account for the
direct mapping's possible load imbalance.

## Rejected universal hypothesis

At depth 2, every one of the 24 format/owner/world-size combinations made
two-phase transfer worse. Every remote candidate was accepted (`A = R`), so no
metadata was suppressed and bitmap bytes were pure overhead. For example,
depth-2/P8/direct rank16 used 30 eager bytes versus 36 two-phase bytes, 20% more.

This is a concrete counterexample to an always-defer-metadata policy.

## Per-round hybrid

An oracle hybrid chose the smaller byte count independently for each superstep.
It selected eager for both depth-2 rounds. In every deeper P8 run it selected
eager for two early rounds and two-phase for all later rounds. It was never
worse than eager in the 120 modeled configurations.

The hybrid's extra byte saving over whole-search two-phase was small on deep
searches, but it removed shallow-search regressions. A real implementation
cannot use future acceptance counts; it needs a predictor based on recent
acceptance/duplicate ratios or a conservative frontier-size threshold.

## Further implications

- Candidate counts are insufficient to size network traffic: key width, parent
  representation, move encoding, and accepted-parent policy all matter.
- Bitmap padding is material for small buffers. At depth 14/P8/direct, separate
  per-peer/per-round padding used 1,175 bytes versus 1,002 bytes for one ideal
  global bitmap, a 17.27% padding overhead.
- Two-phase reduces bytes but adds a dependency and requires retaining source
  metadata. It can lose on latency even when its payload is smaller.
- Deferred parent transfer is unnecessary if only distance/reachability is
  required. Exact path reconstruction makes the metadata policy observable.
- A later GPU benchmark must compare eager, deferred, and hybrid protocols at
  equal path semantics and include buffer memory, packing kernels, collective
  latency, and achieved link bandwidth.

## Reproduction

```powershell
py -m experiments.run_ref011
```

Raw data: `REF-011-wire-byte-sweep.csv`.
