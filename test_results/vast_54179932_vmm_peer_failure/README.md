# Vast 54179932: failed actual VMM peer-read admission

Date: 2026-10-04. Two Tesla T4, driver 595.71.05.
`cuDeviceCanAccessPeer` returned success and allowed=1 in both directions.
This was insufficient: the unchanged `cuda_posix_import.cu` probe built from
52bb77fd8e4cf7ed4187ceeddce598ae5880af43 failed in two independent processes
without any sanitizer, using import --runtime-init --symmetric.

A diagnostic-only change split the device check result into own/peer bits
and printed D2H samples after completion. Both ranks returned result_bits=2:

- rank 0 own 64646464 == expected, peer ffffffff != 65656565;
- rank 1 own 65656565 == expected, peer ffffffff != 64646464.

The failure is present without the new primary alias. The added alias case
passed five local two-process pairs (plain and all four sanitizer tools),
but all imported pairs failed, as did the unchanged control. Therefore its
imported-alias acceptance remains unverified. This does not establish the
cause of the separate Kaggle initcheck/NCCL activation failures.

The production gate was cancelled during NCCL compilation: no full BFS was
run on this host, no COMPLETE was produced, and no performance claim applies.
Verified process group 1363 contained only this gate and its compilers; it
was terminated and an immediate process-group query confirmed it absent.
Raw rank logs and the partial build summary are retained alongside this file.
The host is unsuitable for the LSA acceptance path; no host-sized fallback
was silently substituted.
