# BFS, flooding, rumor spreading, and message time

Synchronous flooding looks like BFS because, under a precise communication
contract, it is BFS: every newly informed vertex forwards to every neighbor and
one edge consumes one reliable round. The time of first receipt is then graph
distance. Small changes to that contract produce weighted, temporal, random, or
incomplete exploration instead.

This note identifies those boundaries. It proposes no network or GPU protocol.

## 1. Exact synchronous flooding contract

Assume a fixed unweighted graph, one source `s`, and discrete global rounds:

1. `s` is informed at time zero;
2. a vertex first informed at time `r` sends the message to every neighbor in
   round `r+1`;
3. every sent message is delivered by the end of that round;
4. duplicate copies carry the same message identity and do not create new
   semantic states.

Then vertex `v` is first informed at time

```text
tau(v) = dist(s,v).
```

For the upper bound, a shortest path of length `d` forwards the message once per
round and informs `v` by time `d`. For the lower bound, any time-`r` delivery
traces back through `r` graph edges to `s`, so it witnesses a path of length at
most `r`. The two inequalities give equality.

## 2. Flooding layers are BFS frontiers

Let `I_r` be all vertices informed by the end of round `r`. Under the exact
contract:

```text
I_r = B_r(s),
I_r \ I_(r-1) = S_r(s).
```

Choosing the first sender as parent produces a BFS tree. Simultaneous senders
create the usual valid-parent tie; arrival order inside one logical round must
not be mistaken for a smaller distance.

If each vertex forwards only once after first receipt, an undirected graph sends
at most one message in each orientation of an edge, hence at most `2m` message
occurrences. Continuous retransmission is a different protocol with potentially
unbounded duplicates.

## 3. Broadcast time and eccentricity

Under exact flooding, the last first-receipt round is

```text
max_v dist(s,v) = ecc(s)
```

over the reachable component. Thus single-source broadcast time equals source
eccentricity in this model. It is not automatically graph diameter unless the
source is peripheral or Cayley homogeneity supplies the needed theorem, as in
note 21.

A local vertex knowing that it has forwarded once cannot infer that this last
round has occurred globally. Time-to-completion and knowledge-of-completion are
separate.

## 4. Fixed delays change the metric

Suppose edge `e` has a fixed nonnegative delivery latency `w(e)` and vertices
forward immediately. The earliest possible arrival time becomes the minimum
sum of latencies along a path: a weighted shortest-path metric, not hop-count
BFS.

A two-hop route with latency `1+1` can beat a one-hop edge of latency `10`.
Freezing the first arrival still gives a correct earliest-arrival label under
this fixed-latency wave model, but not an unweighted distance.

Queues, congestion, time-dependent availability, and adversarial scheduling
can make arrival time depend on departure time and execution history. Then the
appropriate object is a temporal earliest-arrival path or asynchronous
relaxation contract, not a static BFS layer.

## 5. Message loss destroys one-shot completeness

If a one-shot transmission can be lost, a reachable vertex may never be
informed. Duplicate suppression cannot repair a copy that was never delivered.

Retransmission, acknowledgements, or randomized gossip can raise delivery
probability, but introduce a new contract:

- eventual success may be probabilistic rather than deterministic;
- receipt time no longer equals hop distance;
- timeout failure does not prove graph unreachability;
- termination needs delivery/acknowledgement evidence, not silence alone.

The graph may be static and connected while the realized communication trace is
not a complete edge expansion.

## 6. Randomized rumor spreading is not BFS

In random-phone-call rumor spreading, a participant contacts a randomly chosen
partner in each round, using push, pull, or push-pull exchange. On a complete
network, classical protocols spread a rumor in logarithmic-order rounds with
high probability while using far fewer contacts per round than all-neighbor
flooding.

The round in which a node hears the rumor is a random protocol time, not its
distance in the underlying complete graph, which is one for every non-source.
Different runs create different infection trees and completion times.

Randomized rumor spreading optimizes a communication objective under sparse
contacts. It does not compute BFS layers unless the random contact trace itself
is declared as the temporal graph being measured.

## 7. First receipt is safe only for the promised output

Under exact unit-round flooding, first receipt finalizes hop distance. Under
fixed nonnegative latency it finalizes earliest arrival when the continuous wave
assumptions hold. Under arbitrary asynchronous delivery, the first received
proposal can be longer than a later proposal.

For reachability-only output, retaining any successful receipt may suffice. For
minimum distance, a system with out-of-order proposals must either enforce a
distance-safe schedule or allow label improvements and reactivation as in note
18. A visited bit that permanently rejects later improvements changes the
result.

## 8. Termination and acknowledgements

Exact flooding finishes semantically after round `ecc(s)`, but participants may
not know `ecc(s)` or even the component size. Safe termination can rely on a
declared bound, an acknowledgement/echo structure, or a global distributed
termination detector that accounts for in-flight messages.

The following are insufficient by themselves:

- one node has no new messages;
- every local outgoing queue is momentarily empty;
- no new vertex was observed during one timeout;
- all currently known vertices acknowledged, if unknown vertices may still be
  reached by in-flight work.

This is the same knowledge boundary as distributed BFS quiescence, expressed in
message-passing language.

## 9. Cayley and GPU interpretation

Exact push BFS on a Cayley graph resembles flooding: each frontier state emits
all declared generator transitions. But implicit states are not usually one
physical process each. GPUs batch many logical vertices, and owner routing
delivers generated states to the authoritative visited shard.

To preserve the flooding/BFS equivalence:

- every legal generator occurrence must be expanded or otherwise accounted for;
- remote discoveries from logical level `r` must participate in level `r+1`;
- coalescing may remove duplicates but not the only copy of a semantic state;
- delayed cross-device messages cannot be silently inserted at a later level;
- strict bulk-synchronous barriers may be replaced only by an exact
  asynchronous finalization protocol.

Sampling one generator or one remote peer per state is closer to random
exploration or rumor spreading. It may be useful, but it is not exact BFS under
the original all-generator graph.

## 10. Evidence checklist

1. Static unit graph, fixed weighted latencies, or temporal communication graph.
2. Global rounds, bounded delay, or arbitrary asynchronous scheduling.
3. All-neighbor flooding or randomized sparse contacts.
4. Reliable delivery, loss model, retransmission, and acknowledgements.
5. First receipt used for reachability, hop distance, or earliest arrival.
6. Duplicate suppression and message identity.
7. Semantic completion versus distributed knowledge of completion.
8. Logical Cayley state versus physical GPU owner/process.

## Sources

- J. Aspnes, [*Notes on Theory of Distributed
  Systems*](https://www.cs.yale.edu/homes/aspnes/classes/465/notes.pdf),
  distributed graph algorithms chapter. Synchronous flooding/BFS distance and
  asynchronous shortest-path distinctions.
- R. M. Karp, C. Schindelhauer, S. Shenker, and B. Vöcking,
  [*Randomized Rumor Spreading*](https://doi.org/10.1109/SFCS.2000.892324),
  FOCS 2000, 565-574. Random phone-call epidemic dissemination and its
  probabilistic round/message objectives.
- Notes 03, 07, 13, 18, 21, 22, 25, 51, 56, 57, and 95 provide synchronous
  layers, multi-GPU, multisource, asynchronous relaxation, eccentricity,
  temporal graphs, fixed points, ownership, termination, finalization, and
  random-walk boundaries.

## Takeaway

Reliable all-neighbor flooding in unit synchronous rounds is BFS: first-receipt
time equals distance and informed deltas are frontiers. Random contacts,
weighted or variable latency, loss, and local silence change either the metric,
the certainty, or the completion proof. Similar-looking message waves are not
interchangeable without their timing and delivery contracts.
