# BFS dovetailing, infinite branching, and distance finality

## 1. Three meanings of success

On a countably branching implicit graph, distinguish:

1. **Witness discovery:** eventually emit some finite path to a reachable target.
2. **Pointwise distance convergence:** the maintained target upper bound
   eventually equals the true finite distance.
3. **Certified finality:** halt and prove that no shorter target path exists.

Finite-branching BFS normally obtains all three together because every shallower
layer contains finitely many occurrences/states and can be exhausted. Infinite
branching separates them.

## 2. Effective countable successors

Assume each state `x` has a successor enumerator

```text
E_x(0), E_x(1), E_x(2), ...
```

that eventually emits every successor occurrence, although it may never announce
that the list is complete. Every particular edge has a finite enumeration index.
Identity and each emitted transition remain exact.

This is weaker than finite branching with terminating successor enumeration.
It gives positive edge witnesses but generally no finite negative certificate
that no additional successor exists.

## 3. Strict BFS can stall at the first layer

Let root `s` have successors `c_0,c_1,...`, and let `c_0->t`. A strict
level-closed BFS insists on completing all of `F_1` before expanding `c_0`.
Because root enumeration never completes, it never reaches the depth-two target.

The mathematical statement `dist(s,t)<=2` remains true. The executable
layer-setting schedule cannot advance its closure proof.

## 4. Dovetailing finite paths

A dovetailing scheduler interleaves successor enumerators and path depths. One
abstract construction enumerates all finite tuples of successor indices

```text
(i_1,...,i_d)
```

by increasing a fair pairing/diagonal rank and simulates the corresponding
path. Every fixed finite tuple appears after finite scheduler work. Therefore
every reachable target with one finite indexed witness is eventually found.

This is reachability semi-decision. It is not a completed metric-ball traversal:

- no infinite layer is ever exhausted;
- deeper paths run while shallower enumeration remains open;
- ordinary first-discovery distance finality is lost;
- unreachable targets still need not terminate.

## 5. First hit need not be shortest

Let root enumeration be

```text
E_s(0)=c_0,
E_s(M)=t
```

for a large finite `M`, and let `c_0->t`. A fair dovetailing schedule can expand
`c_0` and discover word/path `s,c_0,t` of length two before it reaches root
successor index `M`, even though the true distance is one.

Thus:

```text
fair finite-witness enumeration
does not imply
first-hit shortest-path ordering.
```

An irrevocable visited claim at distance two can also suppress the later direct
edge, reproducing note 164's label-setting versus label-correcting boundary.

## 6. Label correction gives pointwise convergence

Maintain upper bounds `D(x)` initialized to infinity and relax every fairly
enumerated edge:

```text
D(v) <- min(D(v), D(u)+1).
```

Assume every finite causal path is eventually scheduled and decreases are
reactivated fairly. For a target of true finite distance `d*`, choose one
shortest finite path. Its finitely many indexed edges eventually relax in causal
order, so eventually `D(t)<=d*`. Every relaxed label is the length of a real
path, hence `D(t)>=d*`. Therefore eventually

```text
D(t)=d*.
```

The value stabilizes pointwise, but the algorithm need not know when. A still
unenumerated shallower edge/path may always remain consistent with the finite
transcript.

## 7. Convergence is not certified finality

Suppose current incumbent is `mu`. Exact finality requires excluding every
target path of length `<mu`. Under infinite branching, even depth one can
contain infinitely many not-yet-enumerated alternatives. Fairness says each
particular alternative eventually appears; it does not produce a finite time at
which all alternatives have appeared.

Hence the execution may emit a decreasing stream of valid upper bounds and
eventually stop changing while possessing no finite “this is final” event.
This is analogous to a convergent computation without a computable modulus of
convergence.

## 8. Indistinguishable finite prefixes

After any finite enumeration transcript that has not exposed a direct edge
`s->t`, two effective graphs remain possible:

1. no such direct edge ever appears, and the known length-two path is shortest;
2. the same enumerator emits `t` at some later unused index, making distance one.

Any algorithm that finalizes distance two from that finite transcript answers
one of these extensions incorrectly. The impossibility is about the presentation
and negative information, not merely insufficient patience.

## 9. What can restore a finite shortest-distance certificate

Certified finality becomes possible with additional evidence such as:

- finite branching through every depth below `mu`;
- a terminating complete bounded-depth successor enumerator;
- a decidable predicate for existence of a target path shorter than `mu`;
- a finite exact quotient/abstraction whose negative result lifts;
- a proved finite state/rank bound and exhaustive exact identity;
- domain-specific lower bounds matching the incumbent;
- an independently decidable adjacency/word problem strong enough to close all
  smaller lengths.

These are stronger oracles than fair positive enumeration.

## 10. Graph search and duplicate identity

Exact visited still helps merge repeated semantic states, but it cannot make an
infinite layer finite. Under non-layered dovetailing:

- first claim is not necessarily minimum distance;
- permanent rejection of later proposals is unsafe;
- label-correcting updates need versioned/reactivated work;
- termination accounting must include every live enumerator and possible
  decrease obligation;
- all-parent/count/canonical outputs need still stronger equal-distance closure.

The graph may contain finitely many distinct states yet expose them through a
nonterminating redundant successor enumerator. Finite semantic state count alone
does not make that presentation operationally finite.

## 11. Cayley graphs with infinite alphabets

Ordinary puzzle Cayley graphs use a finite move alphabet and are locally finite,
so this failure mode does not apply to their declared move graph.

If the generator alphabet `S={g_0,g_1,...}` is countably infinite, then `F_1`
may be infinite. A state reached early by `g_0 g_1` may also equal a late
one-letter generator `g_M`. Fair word enumeration finds witnesses, but the first
word need not be geodesic in the infinite-alphabet word metric.

Exact group equality merges the endpoints; it does not prove that all shorter
letters/words have been enumerated. Generator enumeration order becomes part of
the effective presentation, while the abstract word metric depends on the whole
alphabet.

## 12. GPU and multi-GPU interpretation

Finite GPU batches can process prefixes of countable enumerators, but no batch
boundary is a completed infinite BFS layer. A fair distributed dovetailer would
need to expose at least:

- parent/enumerator index and depth scheduling fairness;
- active and paused enumerator obligations;
- versioned distance decreases and reactivation;
- minimum unfinished depth/index bounds;
- resource truncation versus semantic invalidity;
- incumbent upper bounds separately from finalized distances.

Adding GPUs increases the processed prefix rate; it does not create a finite
closure event for an infinite set. No optimization or implementation follows
from this conceptual observation.

## 13. Rejected implications

- A target at finite depth is always reached by strict level-synchronous BFS.
- Fair dovetailing preserves first-hit shortest distance.
- Eventual convergence of `D(t)` supplies a detectable finalization time.
- Exact visited turns infinite branching into finite work.
- A finite state set guarantees terminating successor enumeration.
- A valid incumbent path plus scheduler fairness excludes shorter paths.
- Infinite GPU throughput would turn an infinite layer into a finite barrier.
- Exact group equality makes first-found words geodesic for an infinite alphabet.
- Countable branching is operationally equivalent to finite branching.

## 14. Evidence boundary and next gate

The first-hit and indistinguishable-prefix counterexamples and the pointwise
convergence proof are conceptual. No experiment is needed to establish them.
A bounded scheduler model could illustrate finite prefixes but cannot validate
the universal infinite-enumeration claim by exhaustion. Docker remains
irrelevant to the proof and was not retried.

## Sources

- David Poole and Alan Mackworth, *Artificial Intelligence: Foundations of
  Computational Agents*, section on pruning/search completeness, explicitly
  stating the finite-branching condition for BFS shortest-arc completeness:
  <https://www.cs.ubc.ca/~poole/aibook/3e/html/ArtInt3e.Ch3.S7.html>.
- Berkeley CS188, *Uninformed Search*, for the classical shallowest-frontier BFS
  model and the separation of completeness and optimality:
  <https://inst.eecs.berkeley.edu/~cs188/textbook/search/uninformed.html>.
- Florian Lehner, *Algorithmic Traversals of Infinite Graphs*, for formal
  extensions of search/BFS to well-ordered infinite graphs:
  <https://arxiv.org/abs/1810.09974>.
- Note 09 provides the locally finite and infinite-branching baseline; notes 18
  and 164 provide the fair label-correcting schedule used here.
