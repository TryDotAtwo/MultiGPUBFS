# BFS logical obligations, credit conservation, and termination cuts

Distributed BFS completion is about causal work, not queue snapshots. A logical
successor obligation can move from a producer to a device kernel, message,
owner, visited decision, and publication step while disappearing from every
one queue that a local observer happens to inspect.

A safe termination proof must conserve logical obligations across these
locations. Physical retries are copies of one obligation, not new graph work.

This note defines an accounting model, not a termination implementation.

## 1. Root obligations for a strict level

For complete frontier `F_d` and labeled successor manifest `S(x)`, define

```text
O_d = { (x,label) : x in F_d, label in S(x) }.
```

Every member represents one logical duty to determine the declared endpoint
outcome. In a fixed-generator Cayley action,

```text
|O_d| = |S| |F_d|.
```

For an explicit graph it is the sum of expanded outdegrees. The obligation
identity must include the graph/action epoch and enough occurrence identity to
distinguish parallel labels when the output contract requires them.

## 2. A semantic lifecycle

At a consistent observation cut, every root obligation is in exactly one
semantic state such as

```text
pending at producer,
active in expansion,
in transport,
active at authoritative owner,
waiting for accepted-state publication,
retired with final endpoint outcome.
```

Hence

```text
|O_d|
= pending + active_expand + in_transport
  + active_owner + waiting_publication + retired.
```

The categories describe logical ownership of the duty. A physical message can
have several copies because of retry while all copies refer to one
`in_transport` obligation.

## 3. What “retired” must mean

An obligation is not retired merely because:

- its producer kernel emitted a record;
- a send call returned;
- the source deleted its queue entry;
- the owner received bytes;
- a visited bit changed;
- an acknowledgement packet was created.

It is retired only when its endpoint outcome can no longer create missing work
or required metadata. Typical terminal outcomes are:

```text
old state, with required duplicate metadata handled;
same-level duplicate, with required parent/count contribution handled;
new state, with durable/equivalent next-frontier publication complete;
explicit declared failure, which prevents an exact-success claim.
```

Coupling occurrence retirement to publication is one accounting contract. An
alternative can create a separate publication obligation, but then that child
must enter the conserved outstanding-work total before its parent retires.

## 4. Transfer must not create a false zero

A naive global counter can be unsafe. Suppose a sender decrements “local work”
after sending, while the receiver increments “remote work” only after receipt.
Between those events the sum can transiently be zero even though the message is
in flight.

Sampling sender and receiver counters at different logical times creates the
same inconsistent cut. Safe accounting needs an atomic/equivalent transfer,
channel-state inclusion, credit handoff, acknowledgement dependency, or a
consistent snapshot that never loses the obligation between locations.

This is a semantic requirement. It does not select a collective, counter, or
termination-detection algorithm.

## 5. Retries and acknowledgements

Let `id(o)` be the stable logical identity of obligation `o`.

- A retry preserves `id(o)` and does not increase `|O_d|`.
- Multiple physical receives must converge to one semantic application where
  the update is idempotent, or to one application per stable non-idempotent
  contribution identity.
- An acknowledgement may release the sender's responsibility only after the
  protocol's declared receiver/durability condition holds.
- A duplicate acknowledgement must not retire the obligation twice.

Counting retries as new logical obligations can prevent termination forever.
Retiring on the first unreliable send can instead produce premature
termination and lost BFS states.

## 6. Safety and liveness failures have opposite token shapes

The obligation view makes note 173's termination decomposition concrete:

- **lost/destroyed credit:** outstanding work exists but accounting says zero;
  termination can be unsafe;
- **leaked/never-returned credit:** semantic work is complete but accounting
  remains positive; termination is safe but not live;
- **duplicated retirement:** one obligation is subtracted twice; unsafe zero or
  underflow can occur;
- **missing retirement:** a completed obligation remains pending; liveness
  fails.

An exact detector must prevent both false zero and permanent false nonzero.

## 7. GPU work can be causally active without a host record

Examples of non-queue-resident outstanding work include:

- a running kernel whose threads can still emit candidates;
- a device-side append buffer not yet published to the host/runtime;
- an asynchronous copy or collective carrying records;
- a stream-ordered callback not yet executed;
- an owner kernel that claimed states but has not completed compaction;
- a spill/replay batch whose durable status is unresolved.

An API call returning asynchronously changes where the obligation lives; it
does not retire it. Device events become completion evidence only under a
protocol that connects them to the semantic lifecycle.

## 8. Completed level theorem

A strict level `d` is semantically complete when:

1. every root obligation in `O_d` is retired exactly once;
2. every created child obligation, including accepted-state publication and
   required metadata, is retired;
3. no failed/overflow outcome is being relabeled success;
4. every accepted next state is durably/equivalently present in `F_(d+1)`;
5. authoritative visited and the requested output reductions agree with that
   frontier.

Only then may the prefix claim advance from exact `B_d` to exact `B_(d+1)`.
An empty local queue is neither necessary nor sufficient for this theorem.

## 9. Outstanding depth matters for target stopping

A scalar outstanding count is sufficient for full quiescence only when zero is
soundly detected. Early target stopping additionally needs the lower-bound key
of unfinished work.

Let

```text
U(k) = number of outstanding logical obligations capable of producing a path
       with lower-bound distance k.
```

If the best target upper bound is `mu`, target finalization requires exclusion
of every unfinished obligation that could produce a result `<mu` under the
declared stopping theorem. A positive total outstanding count at keys
`>=mu` may be irrelevant to scalar target optimality while still relevant to a
full frontier, canonical parent, or all-path output.

Thus

```text
total outstanding,
minimum unfinished depth/key,
output-specific unfinished metadata
```

are separate termination summaries.

## 10. Asynchronous label correction creates obligations dynamically

For strict level expansion, root work is known from `F_d`. In label-correcting
execution, a successful decrease creates or reactivates downstream propagation
work. Termination accounting must therefore support dynamic causal creation:

```text
created work = retired work + outstanding work
```

at a consistent cut, with no creation permitted after the parent credit has
been returned unnoticed.

Quiescence means no active/in-flight obligation and no enabled improvement that
the fairness contract still owes. A static queue-length sum cannot express this
alone.

## 11. Recovery and epoch changes

A checkpoint must either:

- capture the consistent lifecycle location of every outstanding obligation
  and channel effect; or
- roll back to a completed ancestor boundary and recreate the root obligations.

After repartition or world-size change, stable logical IDs remain tied to the
same graph/search epoch while physical owners change. Old acknowledgements or
credits from another epoch must not retire current work.

This is the termination analogue of stale visited replicas: monotonicity and
idempotence do not cross an undeclared epoch boundary.

## 12. Telemetry for a consistent cut

For each level or termination epoch, retain:

```text
root obligations created,
pending producer obligations,
active kernels,
sent/unacknowledged logical IDs,
received/not-applied logical IDs,
owner decisions pending,
publication/metadata obligations pending,
retired success/old/duplicate/failure outcomes,
physical retry and duplicate counts,
minimum unfinished depth/key,
cut/snapshot identifier.
```

The semantic totals should conserve at one cut. Summing independently sampled
per-rank counters can appear balanced or zero while never representing a real
global state.

## 13. Minimal failure fixtures

### Message gap

One rank sends the only candidate `b` to its owner. Source queue becomes empty;
receiver has not yet observed the message. Local-empty reduction falsely ends
the layer.

### Orphaned publication

Owner claims `b`, returns credit, then fails before `b` becomes pending frontier
work. Membership exists, causal expansion does not.

### Duplicate acknowledgement

Two acknowledgements for one logical ID decrement an outstanding counter twice,
creating a false zero while another obligation remains.

### Lost acknowledgement

All semantic work completes, but one returned credit is lost. The detector
waits forever: safe, not live.

## 14. Rejected implications

- All local queues empty means no work exists.
- A returned send call retires the successor obligation.
- Physical retry creates a new logical graph occurrence.
- A visited claim permits credit return before publication.
- A zero sum of counters sampled at different times is a consistent cut.
- No host work means no active GPU work.
- One scalar outstanding count is enough for every target/output stop.
- Conservative overcount is harmless because it cannot return a wrong result.
- Checkpoint files from every rank automatically contain all channel work.

## 15. Current synthesis

Distributed termination is conservation of causal responsibility. A logical
obligation may move, split into explicitly tracked child duties, or acquire
physical retry copies, but it must never disappear before its semantic effects
are complete and must not remain forever after they are complete.

This view connects correctness and progress: missing credit threatens safety;
leaked credit threatens liveness. Queue emptiness is only one local observation
inside that larger invariant.

This note extends notes 09, 18, 30, 52, 56, 160, 162, 164, 172, and 173.

