# REF-023: REF-010 Docker reproduction and output audit

Date: 2026-08-28

Status: pass after one comparison-method correction

## Question

Does the current REF-010 source reproduce its retained exactness/routing
artifacts inside Docker, and which note 57 output contracts do those artifacts
actually validate?

## Environment

The existing local image was used without rebuilding:

```text
tag: multigpubfs-gpu:dev
image: sha256:55f9efc3c2d82a3110e23f9fdc194026d6f55197105d10dfd6f48a4d0240bf0f
created: 2026-08-27T21:57:56.102054143Z
workspace mount: /workspace, read-only
temporary output: container /tmp/ref010-audit
```

The image entrypoint is the Rust/CUDA smoke binary, so the experiment overrode
the entrypoint with `/bin/bash` and used the already installed Python 3.10.12
only to rerun the pre-existing REF-010 simulator. No host calculation or source
change was made.

## Command shape

```text
docker run --rm --entrypoint /bin/bash \
  -v <workspace>:/workspace:ro -w /workspace \
  multigpubfs-gpu:dev \
  -lc "python3 -m experiments.run_ref010 --output-dir /tmp/ref010-audit ..."
```

Generated files were compared with:

```text
/workspace/experiments/REF-010-directed-validation.json
/workspace/experiments/REF-010-s8-routing.csv
```

## Result

The rerun completed:

```text
directed graphs: 4,096
ordered distinct pairs per configuration: 49,152
configurations: 6
total distributed searches: 294,912
S8 routing rows: 40
distance mismatches: 0
path replay failures: 0
round accounting failures: 0
```

After normalizing line endings, both regenerated artifacts matched the retained
files exactly. The retained file hashes were:

```text
REF-010-directed-validation.json
  sha256 461e2fa433a1aa52343a886d8b908bd91008f9ccb419542d9805fa5304a12d8f

REF-010-s8-routing.csv
  sha256 80dae1a33e39270565c74be7f1300cc67177e5976c934b5a387ff1110ef597ab
```

The focused Docker unit suite also passed:

```text
python3 -m unittest tests.test_distributed_bidirectional -v
Ran 2 tests
OK
```

It covers one three-owner path/accounting fixture and rejection of an owner
outside the declared world.

## Preserved failed comparison

The first check used raw `cmp`. It reported:

```text
REF-010-directed-validation.json differ: char 2, line 1
```

The regenerated JSON used Linux LF while the retained JSON used Windows CRLF.
This was a representation mismatch, not a content mismatch. The corrected
comparison used `diff --strip-trailing-cr`; it produced no differences for
either artifact.

This matters because a byte hash is a file-identity claim, while semantic JSON
or normalized-text equality is a content claim. Both should be named rather
than treating newline conversion as an algorithm failure.

## Output-contract finding

REF-010 validates, within its finite and simulated scope:

- target reachability/distance equality with an independent unidirectional BFS;
- one replayable shortest move sequence;
- aggregate per-round conservation equations;
- complete-level bidirectional stopping under logical owner routing;
- deterministic reproduction of the retained metrics on the current source.

It does not validate:

- a real transport, NCCL collective, GPU kernel, queue, or in-flight message;
- asynchronous or partially completed layers;
- deterministic/canonical shortest words across implementations;
- complete predecessor DAGs, shortest-path counts, all paths, or sampling;
- crash/retry/checkpoint semantics;
- wall time, bandwidth, topology, or scaling speedup.

## Evidence scope

The exhaustive corpus contains every loop-free directed simple graph on four
vertices, but not self-loops, parallel labeled edges, state hash collisions, or
larger arbitrary graphs. S8 adds 40 exact Cayley cases at selected depths and
ownership mappings. Finite exhaustive validation supports the model within
those corpora; the abstract correctness argument still depends on complete
successors, exact identity, authoritative owner decisions, complete
supersteps, and the stopping theorem.

No implementation or optimization was added.
