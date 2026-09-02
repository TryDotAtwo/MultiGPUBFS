# REF-022: Cayley girth and relation-onset probe

Date: 2026-08-28

Status: pass after an earlier Docker infrastructure failure

## Question

Can a tiny exact Rust BFS show where non-backtracking word growth first
diverges from unique-state sphere growth in three small Cayley graphs?

## Cases and implementation

1. `Z_31` with `{+1,-1}`: the Cayley graph is `C_31`.
2. `Z_8 x Z_8` with four axis generators: `ab=ba` closes squares.
3. `S_3` with adjacent involutions `{s_0,s_1}`: its Cayley graph is `C_6`
   and `s_0 s_1 s_0 = s_1 s_0 s_1`.

`experiments/ref022_girth_probe.rs` is a standalone educational Rust file, not
a reusable library, GPU workload, benchmark, or optimized search.

## Counter semantics

For every completed frontier `F_d`:

```text
occurrences
  = parent_returns + visited_nonparent + candidate_occurrences

candidate_occurrences
  = unique_candidates + convergence_duplicates

accepted_next = unique_candidates.
```

- `parent_returns` are edges to the selected BFS-tree parent: ordinary inverse
  backtracking, not a newly detected relation.
- `visited_nonparent` includes alternate predecessors, same-level boundary
  edges, older-ball edges, and saturation effects.
- `convergence_duplicates` are additional current-layer occurrences proposing
  the same new state.
- `nonbacktracking_words` is `1` at depth zero and `q(q-1)^(d-1)` afterward.

Parent choice can change the split between parent returns and other visited
hits, but not their sum, frontier sets, or new-candidate convergence.

## Test-first and Docker evidence

Tests preceded implementation. The RED Docker compile failed because
`LevelCounts`, `z31_rows`, `z8_square_rows`, and `s3_rows` did not yet exist.
After the minimal implementation, all three tests passed. The first GREEN
command exposed one rustfmt line-wrap difference; after that correction,
tests, `rustfmt --check`, compile, and run all passed.

```text
image tag: multigpubfs-rust-toolchain:dev
image id: sha256:764a443c2ddc39b28b8fbb0b1495656984ea5ee8dd82f7f435f2069a6574ce69
created: 2026-08-27T21:19:34.480108485Z
rustc: 1.75.0
workspace mount: read-only for compile/test/run
tests: 3 passed, 0 failed
```

All compilation, testing, formatting checks, and calculations ran in Docker.

## Exact observations

### `Z_31`

At every depth `1..14`, the two frontier states produce two parent returns and
two distinct new states with no other visited hit or candidate convergence.
At depth 15:

```text
frontier=2
parent_returns=2
visited_nonparent=2
candidate_occurrences=0.
```

The boundary vertices remain distinct and each has a unique length-15 root
geodesic, but they are adjacent. The two visited-nonparent occurrences are the
two directed views of that one undirected boundary-closing edge. This is the
sharp difference between unique radius-15 geodesics and the induced radius-15
ball being a tree.

### `Z_8 x Z_8`

At depth one, four frontier states generate 12 non-parent candidates but only
eight unique depth-two states, so `convergence_duplicates=4`. These are the
commuting-square coincidences such as `ab=ba`, appearing at half the
length-four relation.

```text
depth 4: nonbacktracking_words=108, frontier=14
depth 8: nonbacktracking_words=8748, frontier=1.
```

This is word-tree growth versus finite torus geometry, not a hardware result.

### `S_3`

While expanding `F_2`, two occurrences converge on the unique opposite element
in `F_3`: `candidate_occurrences=2`, `unique_candidates=1`. They are the braid
words `s_0 s_1 s_0` and `s_1 s_0 s_1`. Expanding the opposite vertex then sees
one selected parent return and one other visited edge back to `F_2`.

## Interpretation

| Geometry | First counter signal |
|---|---|
| odd cycle `C_(2r+1)` | same-level visited-nonparent edge while expanding `F_r` |
| even cycle / equal length-`r` words | candidate convergence while producing `F_r` |
| commuting `ab=ba` | convergence at depth two |
| inverse generator pair | parent returns at every positive depth |

A scalar duplicate count therefore hides inverse returns, relation convergence,
alternate predecessors, same-level edges, and finite closure.

## Artifacts and earlier failure

The complete 29-row output is retained in
`experiments/REF-022-girth-relation-onset.csv`: 16 `Z_31`, nine `Z_8 x Z_8`,
and four `S_3` rows.

The first attempt had correctly remained unexecuted when Docker Desktop's WSL
backend failed while initializing `dockerInference`; no host compilation was
substituted. The later healthy Docker retry supersedes that status while
preserving the infrastructure failure in this history.

## Scope

- exact only for the three declared finite fixtures;
- no timing, GPU, multi-GPU, Cube, or Megaminx quantitative claim;
- no claim that the shortest written presentation relator equals actual girth;
- no optimized implementation.
