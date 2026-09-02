# BFS support graphs, probabilistic reachability, and MDPs

BFS can forget transition probabilities and traverse only the **support graph**:
an edge exists whenever its transition probability is positive. This preserves
one qualitative question—whether some finite positive-probability path reaches
the target—but it does not preserve the probability of reaching it, almost-sure
reachability, expected hitting time, or an optimal policy.

This note studies those semantic boundaries. It adds no implementation,
optimizer, benchmark, or GPU code.

## 1. Fixed finite Markov-chain contract

Let `P(x,y)` be a transition matrix on a finite state set `S`, and let `T` be a
target set. The support graph has

```text
x -> y  iff  P(x,y) > 0.
```

Support distance is the ordinary unweighted graph distance

```text
d_sup(s,T) = min{k : a support path of k edges goes from s to T}.
```

It ignores every positive probability's magnitude. Consequently, changing an
edge probability from `1/2` to `10^-100` leaves support BFS unchanged.

## 2. Exactly what finite support reachability proves

For a fixed finite Markov chain,

```text
d_sup(s,T) < infinity
iff
Pr_s(tau_T < infinity) > 0,
```

where `tau_T` is the first hitting time. A finite support path has a finite
product of strictly positive transition probabilities, hence positive
probability. Conversely, any finite target-hitting trajectory uses support
edges and contains such a path.

The distance also gives the earliest time with nonzero hit probability:

```text
d_sup(s,T) = min{k : Pr_s(tau_T = k) > 0}.
```

This statement concerns *possibility*, not likelihood.

## 3. Positive probability is not probability one

Consider

```text
P(s,t) = 1/2,   t in T and absorbing,
P(s,c) = 1/2,   P(c,c) = 1.
```

BFS reports `d_sup(s,T)=1`, yet the eventual hitting probability is only
`1/2`. On the other hand,

```text
P(s,t) = 1/2,   P(s,s) = 1/2
```

also has support distance one, but reaches `t` almost surely: the probability
of remaining at `s` for `n` trials is `2^-n`, which tends to zero.

Thus the same BFS distance is compatible with different qualitative answers.

## 4. BSCC criterion for almost-sure reachability

Make target states absorbing without changing the first-hit event. In a finite
Markov chain, a run eventually enters a bottom strongly connected component
(BSCC) with probability one. Therefore `T` is reached almost surely from `s`
exactly when every BSCC reachable from `s` intersects `T`.

BFS supplies the reachable region used by this test, but does not replace the
SCC/BSCC analysis. One target path proves positive probability; excluding every
reachable target-free recurrent class proves probability one.

## 5. Hitting probability is a harmonic quantity

Let

```text
h(x) = Pr_x(tau_T < infinity).
```

With `h=1` on `T`, non-target states satisfy

```text
h(x) = sum_y P(x,y) h(y),
```

plus the appropriate zero boundary on states that cannot reach `T`. This is a
weighted fixed-point or linear-system problem, not a level recurrence. BFS
cannot recover `h` after discarding the weights.

Two chains with identical support can have identical BFS frontiers and very
different `h` values.

## 6. Expected hitting time is a separate question

Under the usual convention `tau_T=infinity` on non-hitting runs,

```text
Pr_s(tau_T < infinity) < 1  implies  E_s[tau_T] = infinity.
```

Conditional expected hitting time given eventual success is a different
quantity and must be named explicitly. When the target is reached almost surely
in a finite chain, expected hitting time is finite and satisfies

```text
e(x) = 1 + sum_y P(x,y)e(y),   e(t)=0.
```

For infinite-state chains, almost-sure hitting alone need not imply finite
expectation. The finiteness assumption is substantive.

In the self-loop example above, `E_s[tau_T]=2` although support distance is one.

## 7. Bounded reachability exposes what BFS omits

Define `h_k(x)=Pr_x(tau_T <= k)`. Then

```text
h_0(x) = 1[x in T],
h_(k+1)(x) = 1                         if x in T,
             sum_y P(x,y) h_k(y)       otherwise.
```

Support BFS only records the first `k` for which `h_k(x)>0`. It discards the
entire sequence of values thereafter. This makes the boundary particularly
clear: boolean frontier growth is the support shadow of a numerical dynamic
program, not a numerical solution itself.

## 8. MDPs add a choice before randomness

In a finite Markov decision process (MDP), a controller selects an enabled
action `a`, then Nature samples `y` according to `P(x,a,y)`. Maximum and minimum
reachability values quantify over policies:

```text
V_max(x) = sup_pi Pr_x^pi(tau_T < infinity),
V_min(x) = inf_pi Pr_x^pi(tau_T < infinity).
```

For non-target states, the Bellman form is

```text
V_max(x) = max_a sum_y P(x,a,y)V_max(y),
V_min(x) = min_a sum_y P(x,a,y)V_min(y),
```

with graph preprocessing needed to identify the zero/one regions and avoid
spurious self-supporting numerical fixed points. For finite reachability MDPs,
memoryless deterministic policies suffice for optimal values.

## 9. Three quantifiers that must not be collapsed

For one chosen action with support successors, compare:

```text
exists successor reaches T     support possibility,
random successor reaches T     probabilistic reachability,
forall successors reach T      adversarial/sure reachability.
```

Ordinary BFS uses the first. Note 131's universal game uses the third. An MDP
uses the middle quantifier after a controlled action.

The self-loop example `P(s,t)=P(s,s)=1/2` is almost-sure winning, but loses if
Nature is replaced by an adversary allowed to choose the self-loop forever.
Treating all random successors as adversarial is therefore a sound conservative
model for **sure** reachability, but is too strong for almost-sure reachability.

## 10. Support BFS inside an MDP

If an MDP support graph contains an edge whenever some action gives it positive
probability, BFS answers whether some policy has a positive-probability finite
target path. It does not answer whether the maximum value is one, which action
maximizes the value, or whether every policy succeeds.

A shortest support path may use an action whose target branch has tiny
probability and whose other branch enters a trap. A longer-looking action can
have much larger or even unit eventual success probability. Optimizing support
distance and optimizing reachability probability are different objectives.

## 11. Cycles change meaning across the three models

- In ordinary BFS, a cycle adds no shorter simple path after visited-state
  suppression.
- In an adversarial reachability game, a universal self-loop can support an
  infinite losing play.
- In a Markov chain, a self-loop with a fixed probability below one can be left
  almost surely, although it increases expected time.

The adjacency is insufficient to choose among these interpretations. Ownership
and transition probabilities are part of the problem state.

## 12. Cayley and Schreier interpretation

Choosing generators uniformly or with declared weights produces a random walk
on a Cayley or Schreier graph. Support BFS gives the word metric generated by
the positive-probability labels. It does not give target hitting probability,
expected hitting time, mixing, or cover time.

If a controller chooses a generator family and a random perturbation selects
the actual transition, the exact state is an MDP state such as

```text
(group/orbit element, control phase, stochastic mode).
```

Parallel labels that reach the same endpoint still contribute probability
mass. Deduplicating endpoints is safe for support reachability but can corrupt
the transition matrix unless their masses are combined.

## 13. Missing tiny edges can reverse qualitative answers

For support distance, any omitted positive-probability edge may remove the only
path. For almost-sure analysis, even a tiny omitted edge into an absorbing trap
can turn a true probability below one into a reported probability one.
Conversely, a spurious trap edge can create a false failure.

Numerical tolerance is therefore not automatically a semantic permission to
round support to zero. A model may deliberately declare a cutoff, but then it
is analyzing a different transition system and should report that contract.

## 14. GPU and multi-GPU boundary

A probabilistic reachability study should separate:

- support construction and ordinary BFS reachability;
- SCC, BSCC, and MDP end-component preprocessing;
- zero/one qualitative classification;
- bounded-horizon dynamic programming;
- unbounded probability computation and its residual/error certificate;
- expected-time equations and the convention for non-hitting runs;
- policy extraction and independent policy evaluation;
- probability-mass preservation across duplicate labels and owners;
- communication, reductions, convergence rounds, and end-to-end time.

On sharded transitions, local row sums do not prove that a global probability
distribution sums to one. A complete support epoch and an explicit reduction
of all probability mass are semantic prerequisites. Floating-point convergence
evidence is not BFS level-synchronization evidence, and vice versa.

These are conceptual correctness boundaries, not an implementation plan.

## Sources

- C. Baier and J.-P. Katoen,
  [*Principles of Model Checking*](https://mitpress.mit.edu/9780262026499/principles-of-model-checking/),
  MIT Press, 2008, Chapters 10–11, for finite Markov-chain and MDP reachability,
  BSCC/end-component analysis, and numerical solution methods.
- D. Parker,
  [*Probabilistic Model Checking* tutorial](https://www.cs.ox.ac.uk/people/david.parker/talks/dave-dagstuhl23pmctut.pdf),
  University of Oxford, 2023, for probabilistic reachability queries, MDP
  strategies, and the separation of graph analysis from numerical solution.
- M. Kwiatkowska, G. Norman, and D. Parker,
  [*Probabilistic Model Checking: Advances and Applications*](https://www.cs.ox.ac.uk/David.Parker/papers/fsv-pmc.pdf),
  for optimal probabilistic reachability, memoryless deterministic strategies,
  and graph-plus-numerical computation.
- Notes 45, 85, 95, 108, and 131 supply this repository's weighted-transition,
  stochastic-process, random-walk, electrical-network, and adversarial-game
  boundaries.

## Takeaway

Support-graph BFS answers the earliest step at which target hitting has positive
probability. It does not say how large that probability is, whether it is one,
how long a hit takes on average, or which MDP policy is optimal. Almost-sure
probabilistic reachability also differs from universal adversarial reachability:
a random self-loop can be escaped with probability one even though an adversary
could choose it forever.
