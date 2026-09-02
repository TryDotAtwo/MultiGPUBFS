# REF-046: discovery-publication interleavings

Date: 2026-08-29  
Status: pass; bounded Rust state model, not a runtime protocol validation

## Question

Do bounded interleavings distinguish blind already-visited drop from a helpable
claim descriptor and a logged publication intent under one injected claimant
stop?

## Declared model

A child has already been claimed in exact visited state. Three later events are
permuted exhaustively:

```text
stop claimant
claimant attempts publication
recovery observes the claim
```

There are `3! = 6` local schedules. The protocols differ only in what survives
the claim:

- `BlindDrop`: visited membership survives, but no recoverable payload duty;
- `HelpableDescriptor`: recovery can publish the retained descriptor;
- `LoggedIntent`: recovery can drain the retained intent.

Publication is modeled as an idempotent set commit. Physical attempts and
unique committed records are counted separately.

The path fixture is `s->a->b->c`. Exactly one publication position is subjected
to the six event orders, so the declared single-stop corpus has
`3 edges * 6 orders = 18` schedules. The other two publications complete
normally.

## Test-first evidence

The first Docker build accidentally omitted `tests/` from its build context and
passed without running the new integration tests. This invalid RED was rejected.
`docker/Dockerfile.gpu` now copies `tests/`, making integration tests part of
the normal builder gate.

After the library target was declared, the empty model produced a valid
behavioral RED: all four initial tests failed with zero schedules instead of
six. The minimal model then passed those tests. A fifth chain-coverage test was
added RED-first and passed after the three-edge enumeration was implemented.

Final Docker builder gate:

```text
existing binary tests: 1 passed
REF-046 integration tests: 5 passed
doc tests: 0 failed
rustfmt --check: passed
release build: passed
```

## Observed finite outcomes

| protocol | local schedules | visited but unpublished | three-edge target reached |
|---|---:|---:|---:|
| blind drop | 6 | 3 | 9 / 18 |
| helpable descriptor | 6 | 0 | 18 / 18 |
| logged intent | 6 | 0 | 18 / 18 |

For both recoverable protocols, at least one schedule performs two physical
publication attempts while the idempotent set commit retains one unique
record. This is safe for set membership in the model. It would not by itself be
safe for non-idempotent count, sum, or all-occurrence output.

## Interpretation

Atomic novelty and frontier publication are two commitments. Blind duplicate
drop can leave an exact visited bit with no future expansion record. A
descriptor or logged intent closes this particular bounded gap because some
live actor retains enough information to finish publication.

The result does not choose between descriptors and logs. Both satisfy the same
finite safety property here while making different unmodeled cost, persistence,
and recovery assumptions.

## Boundaries

- Events are a sequential finite abstraction, not Rust/CUDA atomics or a memory
  model.
- Only one claimant stop and one child obligation at a time are modeled.
- There is no queue capacity, crash/restart, transport loss, ABA, epoch reuse,
  multi-GPU failure, or persistent storage.
- Recovery is guaranteed to execute; termination detection is not implemented.
- Idempotent set output is modeled; richer additive outputs need logical IDs or
  another exactly-once contribution contract.
- No timing, GPU code, optimizer, or production protocol was added.

## Reproduction

```powershell
docker build -f docker/Dockerfile.gpu --target builder `
  -t multigpubfs-ref046-green .
```

The builder runs `cargo fmt --check`, the release build, and the complete Rust
test suite inside Docker.

The earlier infrastructure-only record remains in
`REF-046-discovery-publication-interleavings-not-run.md` as evidence of the
blocked attempts before Docker access returned.

