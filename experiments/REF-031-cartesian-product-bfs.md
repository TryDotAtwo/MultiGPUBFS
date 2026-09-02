# REF-031: Cartesian-product BFS

Status: pass after Docker availability, shell-PATH, and missing-component
failures.

## Question

Does a tiny exact BFS witness the additive-distance, sphere-convolution, and
shortest-path-interleaving laws for a Cartesian product, while rejecting their
blind transfer to the strong product?

## Fixture

- Rust source: `ref031_cartesian_product_bfs.rs`
- factors: the three-vertex path `P3` and four-cycle `C4`;
- Cartesian adjacency changes exactly one coordinate;
- strong-product adjacency additionally permits simultaneous coordinate moves;
- exact vertex identity and ordinary FIFO BFS;
- no timings, GPU code, or optimization.

The tests cover all 12 product vertices for the distance identity, compare the
whole sphere vector with the factor convolution, check one path-multiplicity
witness, and retain a diagonal strong-product counterexample.

## Attempt log

1. The first Docker run failed with permission denied on the Docker API pipe.
2. The approved retry established that the Linux-engine pipe did not exist.
3. `docker desktop status` reported that Docker Desktop was not running.
4. `docker desktop start` did not reach readiness and was interrupted after
   repeated silent waits.
5. Read-only inspection found `com.docker.service` stopped. An approved
   `Start-Service` attempt failed because the current process could not open
   the service.
6. On a later continuation, Docker Engine `29.3.1` was available. The official
   `rust:1.85-bookworm` image was pulled at digest
   `sha256:e51d0265072d2d9d5d320f6a44dde6b9ef13653b035098febd68cce8fa7c0bc4`.
7. The first image run used `bash -lc`, which reset the image PATH and reported
   `rustc: command not found`.
8. Retrying with `sh -c` passed all four semantic tests, then stopped because
   the image toolchain did not include `rustfmt`.
9. The final one-shot container installed the `rustfmt` component, passed the
   format gate and all four tests, compiled the executable, and printed the
   retained observations.

No native Rust fallback was used because the project contract requires Docker.

## Observed results

The final Docker run passed four of four tests and observed:

```text
P3 spheres: [1,1,1]
C4 spheres: [1,2,1]
Cartesian predicted/observed: [1,3,4,3,1]
far endpoint (2,2): distance 4, shortest paths 12
diagonal (1,1): Cartesian distance 2, strong distance 1
```

The 12 paths are the two shortest `C4` routes times the six interleavings of
two `P3` steps and two `C4` steps. The strong-product diagonal is the retained
counterexample to transferring the additive metric to another graph product.

Raw command, image/toolchain identity, test output, and executable output are
retained in `REF-031-cartesian-product-bfs.txt`. A separate read-only Docker
hash gives source SHA-256
`54658d2fa6b6766178770d8b64a08af53523385feb20d2c64bf6b7d056a993c9`.

## Interpretation

The bounded observation agrees with the proof in note 69 for every vertex of
this 12-state fixture. It validates the Rust oracle and catches the intended
strong-product boundary. It does not establish product structure for any
puzzle, provide a performance measurement, or imply an implementation method.
