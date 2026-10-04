# Production warmup v104 — retained evidence

Worker COMPLETE, root report `TYPED_WARMUP_PASS`, exact checkout
`bd294070c77f8be27972d089fb5207ac1a838b73`. Two physical Tesla T4,
P2P allowed both ways; driver 580.178.04, CUDA compiler 12.9.86.
Root summary SHA256:
`14a8d96887359b64117a0bf35eaab69a4aa5b81f8c1453d1606bed442a4a2ac1`.

Retained small outputs: `build/kaggle-v104-observation/lsa-bfs-gate`.
Three summaries: root and DENSE/HASH_FIRST production RunConfigV1 suites.
Both profiles use CUCO_RANK, three source banks and completion credits,
pre-dedup ON, rank map 0,1, batch 1, full S4 archive CPU oracle.
Both healthy warmup+measurement cases pass. All 40 asymmetric fault
cases pass: both processes fail within the 45-second bound, no forced
cleanup and no false group COMPLETE. Four rank-local warmup-setting
disagreements terminate with return codes [1,1] in 0.653–0.654 seconds.

Nested replay scope still says diagnostic/not T4 acceptance; the source
patch digest is the empty SHA256, and root hardware admission is T4.
This report records the tested cases, not overall production acceptance.
It is not a performance result or a four-sanitizer gate. Full-runtime
initcheck remains unresolved; the unchanged-runtime host-versus-CUDA12.9
instrumenter comparison is running separately in typed-stress v8.
