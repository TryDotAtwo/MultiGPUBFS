# BFS discovery, publication, and helpable commit

## 1. The gap behind one atomic visited bit

An exact identity claim answers only:

```text
which candidate first changed x from unseen to accepted?
```

BFS also needs:

```text
who makes the accepted x available for every future expansion/output duty?
```

These are different events. A CAS or atomic bitmap operation can linearize
novelty while leaving the winner unable to publish the child. Notes 173--174
identify this as the `PB` obligation and an orphaned-publication failure. This
note isolates the state machine and the sufficient repair patterns.

## 2. Minimal semantic state machine

For one semantic state `x` accepted from completed frontier `F_d`, distinguish:

```text
ABSENT
  -> CLAIMED(x, d+1, obligation_id)
  -> PUBLISHED(x, d+1, obligation_id)
  -> EXPANDED(x, d+1, obligation_id)
```

`CLAIMED` means exact visited authority has selected one novelty winner.
`PUBLISHED` means an authoritative frontier record, durable log record, or
equivalent recoverable expansion duty exists. `EXPANDED` means all required
successor occurrences and output contributions have retired.

The crucial invariant is not merely

```text
x in Visited.
```

It is

```text
x in Visited
=> PUBLISHED_OR_EXPANDED(x) or live_publication_obligation(x).
```

The right side must remain true across every allowed interleaving and failure
model. This is a causal invariant, not necessarily one physical transaction.

## 3. Small counterexample

Take the path

```text
s -> a -> b -> c.
```

While expanding `a`, worker `W1` marks `b` visited and stops before enqueueing
it. A retry by `W2` sees `b` visited and discards the occurrence. Then:

- graph generation was sound and complete for every actual call;
- identity and novelty linearization were exact;
- `b` belongs to the recorded reached set;
- `b` is never expanded, so `c` is falsely unreachable.

Thus exact visited membership does not imply traversal coverage. In a
failure-free model, fairness of `W1` may eventually close the gap; under crash,
cancellation, overflow, or permanent device loss, fairness is unavailable and
the protocol needs recoverable responsibility.

## 4. Four sufficient coupling patterns

### 4.1 Atomic joint commit

Atomically install both visited membership and a recoverable frontier duty.
This gives the simplest proof but may be impractical across separate tables,
queues, devices, or machines. “Atomic” must cover the semantic objects, not
only two host instructions that can fail between them.

### 4.2 Helpable claim descriptor

The visited entry points to a stable descriptor containing state/depth/epoch
and publication status. Any observer of `CLAIMED` may finish the idempotent
publication and advance the descriptor to `PUBLISHED`. Losers therefore do not
blindly drop the record; they either observe proof of publication or help it.

Necessary conditions:

- descriptor identity is exact and epoch-scoped;
- publication is idempotent or duplicate-tolerant for the requested output;
- reclamation cannot remove the descriptor while a helper can still see it;
- `PUBLISHED` becomes visible only after the frontier payload is visible;
- capacity failure becomes explicit failure, never fake publication.

This is a proof pattern, not a recommendation for one lock-free data structure.

### 4.3 Intent/log before membership

First persist or replicate an idempotently replayable candidate/publication
intent; then claim visited; then mark the intent complete. Recovery can replay
an incomplete intent. This moves the vulnerable interval rather than deleting
it, so the log itself needs exact identity, epoch, completeness, and replay
rules.

### 4.4 Conserved publication obligation

After novelty acceptance, create a separately counted logical publication
obligation before retiring the candidate obligation. Termination cannot pass
while this credit exists. This supplies detection, but detection alone does not
guarantee recovery: some live/recovering agent must own a way to finish the
publication.

These patterns can be combined. Their common theorem is responsibility
continuity, not a particular queue design.

### May a loser publish its own witness?

Return to the diamond

```text
s -> a -> t
s -> b -> t.
```

Let `(a,t)` win the novelty claim and then lose its unpublished payload. A
losing observer holding `(b,t)` has another valid depth-two witness. What it may
publish depends on the output:

- **reached set / distance / future expansion:** publishing state `t` at depth
  two restores the missing frontier duty;
- **one arbitrary replayable path:** publishing parent edge `(b,t)` is a valid
  substitution, although it does not reproduce the original winner;
- **canonical path:** `(b,t)` may be a provisional contender, but a helper
  cannot finalize it without the declared complete minimum reduction;
- **all parents / path counts:** publishing `b` does not recreate the lost
  contribution from `a`; contender identity or a way to regenerate complete
  depth-one expansion is still required;
- **exact execution replay:** substituting `b` for the original winner changes
  the recorded execution even when the mathematical arbitrary-path contract is
  satisfied.

Thus helpability need not mean byte-for-byte recovery of the novelty winner.
It means recovery of a payload sufficient for the requested semantic output.
The descriptor must state which substitutions are allowed; otherwise a helper
can make traversal live while silently weakening the output.

## 5. Why ordering visibility matters

Suppose the status `PUBLISHED` becomes observable before the state payload,
depth, parent, or label metadata. A consumer can expand an uninitialized or
wrong-epoch record. Therefore the abstract protocol requires:

```text
publish payload happens-before expose PUBLISHED,
observe PUBLISHED happens-before consume payload.
```

The concrete release/acquire, stream-event, collective, DMA, or persistent-log
mechanism depends on the runtime. A host atomic does not automatically order a
device buffer or remote transport.

## 6. Level and output boundaries

Publication carries depth `d+1` and graph/search/ownership epoch. Making a
record physically visible early does not authorize expansion before the
schedule contract permits it.

For reached-set/distance output, one idempotent frontier duty per state may be
enough. Richer outputs alter the commit payload:

- canonical parent needs all equal-depth contenders before closure;
- all-parent DAG needs the contender set;
- path counts need contribution identities so retry is not double addition;
- labeled paths need edge/label occurrence identity;
- bidirectional output needs connector publication and closure on both sides.

Thus “helpably publish `x`” is output-relative. Set-union idempotence does not
make numeric addition or first-winner parent choice retry-safe.

## 7. Termination and consistent cuts

Chandy--Lamport treats a global state as process state plus channel state and
uses consistent snapshots to detect stable properties. BFS completion is such
a global property only when accepted-but-unpublished duties are included in the
recorded state. A snapshot of visited and visible queues that omits claim
descriptors or device append buffers can certify a state that never existed as
a causally closed BFS prefix.

Dijkstra--Scholten termination detection superimposes signalling on a diffusing
computation. The BFS-specific interpretation is: novelty can create a child
publication/expansion duty, and the parent signal cannot return until causal
responsibility has either completed or transferred. A zero counter without
this creation-before-retirement rule is not the same theorem.

## 8. GPU and multi-GPU interpretation

Possible locations of `CLAIMED -> PUBLISHED` include:

- visited bit/table versus device append buffer;
- owner claim kernel versus compaction kernel;
- producer GPU versus authoritative owner GPU;
- device buffer versus NCCL/MPI send;
- volatile frontier versus checkpoint/replay log.

For every boundary, ask:

1. What exact event owns publication responsibility?
2. Can the owner stop permanently after novelty wins?
3. Can another agent detect and help/replay the incomplete state?
4. Does duplicate physical publication preserve the requested merge algebra?
5. Which event proves payload visibility and durability?
6. Is the outstanding duty present in termination and checkpoint evidence?

These questions study correctness. They do not select an optimal GPU protocol.

## 9. Rejected implications

- Linearizable visited insertion implies eventual frontier publication.
- A loser that observes `visited=1` may always discard its full record.
- One host atomic orders device, network, and persistent payloads.
- A visible queue tail proves every preceding payload is consumable.
- Duplicate publication is harmless for every BFS output.
- Termination credit detects an orphan and therefore automatically repairs it.
- Queue persistence without visited/epoch consistency is sufficient recovery.
- Exact reached membership proves exact reachable-set traversal.

## 10. Evidence boundary and next gate

This note promotes the discovery/publication gap from a counterexample mention
to a conceptual state-machine contract. REF-046 now supplies bounded evidence:
a test-first Rust model exhaustively enumerates six local event orders and
eighteen single-stop placements along `s->a->b->c`. Blind drop strands three of
six local obligations and reaches `c` in only nine of eighteen path schedules;
helpable descriptors and logged intents publish in every declared schedule.

The fixture does not validate an implementation or memory model. Recovery is
guaranteed to run; queue capacity, memory ordering, transport, epoch reuse,
crash/restart, termination detection, and multi-GPU failure are absent. Its
duplicate physical attempts are idempotent only for set output. No C++ or GPU
implementation is justified by this bounded result. The earlier Docker access
failures remain preserved in the `not-run` precursor report.

## Sources

- K. Mani Chandy and Leslie Lamport, *Distributed Snapshots: Determining Global
  States of Distributed Systems*, ACM TOCS 3(1), 1985, pp. 63--75,
  DOI 10.1145/214451.214456. Primary author-hosted entry and PDF:
  <https://www.microsoft.com/en-us/research/people/lamport/publications/>.
- Edsger W. Dijkstra and C. S. Scholten, *Termination Detection for Diffusing
  Computations*, Information Processing Letters 11(1), 1980, pp. 1--4,
  DOI 10.1016/0020-0190(80)90021-6. Author archive transcription:
  <https://www.cs.utexas.edu/~EWD/transcriptions/EWD06xx/EWD687a.html>.
- Scott Beamer, Krste Asanovic, and David Patterson,
  *Direction-Optimizing Breadth-First Search*, SC 2012. Used only as a concrete
  parallel-BFS reference for frontier/visited work separation, not as support
  for the failure protocol derived here:
  <https://www.scottbeamer.net/pubs/beamer-sc2012.pdf>.
