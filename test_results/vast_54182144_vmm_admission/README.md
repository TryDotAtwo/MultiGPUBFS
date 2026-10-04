# Vast 54182144: actual peer-read and primary-alias admission

Two physical Tesla T4, driver 595.71.05. Probe source commit:
2bc9eff6a8f3f20516256598aa0346c2e126f59c. Compile: nvcc12.9, sm75.

Four independent two-rank plain controls passed: Driver API initialization,
runtime initialization, symmetric VA, and primary plus symmetric alias.
The local/import primary-alias matrix passed all 10 pairs: plain, memcheck,
racecheck, initcheck and synccheck, with both ranks reaching their required
PASS marker, exit [0,0], and no timeout. No suppression was used.
The recorded samples agree with the own and peer expected values.

Compute Sanitizer version: 2025.2.1.0 (build 35969825).
Launcher /usr/local/cuda/bin/compute-sanitizer SHA256:
5cba659e1cd603aa2370e103425f184368e6ed405da90ce85743c4399c324dbc.
Actual /usr/local/cuda/compute-sanitizer/compute-sanitizer SHA256:
979adf037f4e3434a976b7f0912b0a05ce3b062e31b2a73397a1bf1cd5f57547.
Probe source SHA256:
9e7686de090e6c184504bf2851dcf61da43f84310b9afe30a873f8e19e85d0f5.
Probe binary SHA256:
a3c566f1dc70a7f09906a348b955e1d08e5408040da18e61ac2e641312865dd1.
Alias summary SHA256:
ddb45fa020a4581ea6188a132108a76169980ba2e987899c0b5f92d253b7a4f1.
Admission summary SHA256:
10c80592c7a789e841427828a7c7ff112ec67b739d403fd144f748daef499240.

This admits this bounded CUDA VMM shape on this host, not NCCL registration,
full BFS, performance, or the complete four-tool acceptance gate. The separate
full-BFS source52bb77f replay is compiling. On this image both host and pinned
instrumenters are version2025.2.1, so it is not a different-version comparison.
