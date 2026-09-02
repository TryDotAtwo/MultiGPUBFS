# REF-045: preferential-attachment BFS probe — not run

Date: 2026-08-28

## Question

After controlling the realized degree sequence, which BFS-layer effects remain
different between a preferential-attachment graph and a randomized null model?

## Status

Not run because Docker was unavailable.  This is an infrastructure-limited
non-result, not evidence for or against the hypothesis.

The one readiness check was:

```text
docker info --format '{{.ServerVersion}}'
```

It returned permission denied for the `dockerDesktopLinuxEngine` named pipe.
Per the research scope, no Docker repair was attempted.

## Intended semantic comparison

- Rust only, executed inside Docker.
- Frozen, exactly specified preferential-attachment graph.
- Degree-preserving pairing/switching null model, with any loop or parallel-edge
  semantics declared.
- Exact ordinary BFS with full vertex identity.
- Degree, birth-time, frontier, candidate, collision, and owner-routing
  summaries by complete level.

## Unknowns preserved

- Which null-model construction is least misleading at tractable scale?
- How much switching is sufficient before treating age correlations as erased?
- Should roots be uniform, birth-time stratified, degree stratified, or all
  three?
- How stable are first-core-entry depth and peak candidate multiplicity across
  samples?

No code, timing, or numerical result exists for REF-045 yet.
