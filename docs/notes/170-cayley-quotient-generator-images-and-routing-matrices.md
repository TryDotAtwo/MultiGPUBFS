# Cayley quotient generator images and routing matrices

For a normal-subgroup Cayley partition, the quotient image of a generator
determines the destination owner block independently of the concrete parent.
This yields an exact logical routing matrix from the frontier's coset histogram.

Several generators can share one quotient image. They are routing aliases: they
enter the same destination block, but they are not necessarily duplicate
concrete states.

This note derives semantic traffic counts only. It does not prescribe packing,
aggregation, or communication code.

## 1. Generator-image multiplicities

Let `H normal G`, let `Q=G/H`, and let the right Cayley action use a labeled
generator manifest `S`. For quotient element `q in Q`, define

```text
mu(q) = number of labels s in S with sH=q.
```

The multiplicities conserve the manifest:

```text
sum_(q in Q) mu(q) = |S|.
```

The identity image has

```text
mu(H) = |S intersect H|,
```

counting labels with multiplicity. These are exactly the always-local
occurrences from note 167.

Two labels `s,t` share a quotient image exactly when

```text
sH=tH
<=> t^-1 s in H.
```

This relation is independent of the current parent `g`.

## 2. Same quotient destination is not same concrete endpoint

From concrete parent `g`, labels `s` and `t` produce endpoints `gs` and `gt`.
If `sH=tH`, then

```text
gsH=gtH,
```

so both records have the same coset owner. But

```text
gs=gt <=> s=t
```

as group elements. Distinct generator elements therefore remain distinct
concrete endpoints even when their quotient images collide.

Collapsing quotient-image multiplicity is valid for a simple owner-transition
graph. It is invalid for concrete vertex BFS unless the individual states or
their required metadata are retained. Routing coalescence and state
deduplication are different operations.

If the manifest contains two labels denoting the same group element, those
labels do produce one concrete endpoint but may still be distinct move/output
occurrences. That is the label-multiplicity boundary of notes 157 and 160.

## 3. Exact coset-to-coset occurrence matrix

Let

```text
f_d(C) = |F_d intersect C|
```

for quotient coset `C in Q`. Define `M_d(C,D)` as the number of raw labeled
successor occurrences from frontier states in `C` whose endpoints lie in `D`.

Right multiplication gives

```text
M_d(C,D) = f_d(C) mu(C^-1 D).
```

Equivalently, for each `q in Q`,

```text
M_d(C,Cq) = f_d(C) mu(q).
```

Every row conserves raw work:

```text
sum_D M_d(C,D) = |S| f_d(C),
```

and the full matrix satisfies

```text
sum_(C,D) M_d(C,D) = |S| |F_d|.
```

The diagonal is

```text
M_d(C,C) = f_d(C) |S intersect H|.
```

Thus the frontier coset histogram and quotient generator-image histogram
determine the complete logical occurrence-routing matrix before state-level
visited or duplicate resolution.

## 4. A small quotient-alias fixture

Take `G=Z_8`, normal subgroup

```text
H={0,4},
```

and labeled generators

```text
S={+1,+5,+2}.
```

In `G/H`, `+1` and `+5` have the same image because their difference is `4 in
H`. Therefore

```text
mu(H+1)=2,
mu(H+2)=1.
```

From parent `g=0`, the first two labels produce concrete states `1` and `5`.
They are distinct, but both belong to owner block `H+1`. A quotient-simple edge
would show one destination; the raw routing matrix correctly carries
multiplicity two.

This is the smallest distinction needed for interpreting packed destination
traffic: one destination bin can contain several nonduplicate states per
parent.

## 5. From coset blocks to physical GPUs

Let

```text
phi : Q -> {0,...,P-1}
```

map quotient cosets onto available GPUs. The logical GPU-to-GPU occurrence
matrix is

```text
R_d(i,j)
 = sum_(C:phi(C)=i) f_d(C)
   sum_(q in Q) mu(q) 1[phi(Cq)=j].
```

Its total remains `|S||F_d|`. Its diagonal includes all subgroup-label
occurrences and any outside-image transitions whose source and destination
cosets were coalesced onto the same GPU.

This matrix describes where raw logical occurrences belong. Actual transmitted
record counts can differ because an implementation may:

- resolve some exact old states locally;
- combine duplicate concrete states before routing;
- keep several occurrences in one message;
- send keys, full states, or provenance with different widths;
- retry or replay physical messages.

Therefore `R_d` is a semantic routing baseline, not a wire-byte prediction.

## 6. Convolution structure and its limit

The formula

```text
M_d(C,Cq)=f_d(C)mu(q)
```

has the form of right convolution on the quotient group. It says that owner
destinations are translation-invariant at the coset level. This follows from
normality and the Cayley action, not from GPU layout.

The convolution determines occurrence traffic only. It does not determine:

- which endpoint states are already visited;
- which records from different parents converge to one state;
- accepted next-frontier counts;
- parent/DAG/path-count metadata;
- physical locality or timing inside a GPU.

Those require the concrete frontier and state action.

## 7. Forward and reverse directions

Reverse BFS uses inverse generator images. Its multiplicity profile is

```text
mu_rev(q) = mu(q^-1)
```

when reverse labels are obtained exactly by inversion of the forward manifest.
For an inverse-closed manifest, the image histogram is inversion-symmetric.
Even then, forward and reverse routing matrices can differ because their
frontier coset histograms `f_d(C)` differ.

For a directed non-inverse-closed action, using the same `mu` in both directions
silently traverses the wrong transpose graph.

## 8. Why the formula does not transfer automatically to Schreier owners

In a Schreier graph grouped into `H`-orbits, outside-label destination orbits
can depend on the concrete representative. There is then no single quotient
element `q` whose multiplication translates every state in a block to one
destination block.

A matrix can still be measured from concrete occurrences, but it cannot be
predicted solely from one block histogram and one global generator-image
histogram unless a transition-congruence theorem supplies that structure.

## 9. Validation consequences

A bounded validation should compare:

1. declared generator labels and their exact group elements;
2. computed quotient images and `mu(q)` totals;
3. observed parent-level destination cosets for every label;
4. predicted and observed `M_d(C,D)` on exact frontier levels;
5. concrete endpoint identity within each routing bin;
6. logical `R_d` versus physical records, messages, retries, and bytes;
7. forward and inverse-direction profiles separately.

A matching raw matrix validates the action/partition routing equation on the
tested domain. It does not validate successor completeness or accepted
frontiers without the stronger ladder from notes 55 and 163.

## 10. Rejected implications

- Equal quotient generator images imply equal concrete endpoints.
- One quotient edge means one routed state record.
- Quotient-simple degree equals raw generator work.
- The raw routing matrix predicts accepted-state traffic.
- Coalescing destination bins is state deduplication.
- The logical occurrence matrix equals wire messages or bytes.
- An inverse-closed generator histogram makes forward/reverse traffic equal.
- The Cayley convolution formula applies to arbitrary Schreier orbit blocks.

## 11. Current synthesis

Normal Cayley ownership turns generator labels into a fixed quotient-image
histogram. Combined with the current frontier's coset histogram, it determines
an exact raw occurrence matrix by group convolution. This is unusually strong
predictability for an implicit distributed graph.

The prediction stops at owner destinations. Concrete equality, visited status,
cross-parent convergence, accepted progress, physical records, and bytes remain
separate layers of evidence.

This note extends notes 16, 51, 75, 157, 159, 160, 165, 167, and 168.

