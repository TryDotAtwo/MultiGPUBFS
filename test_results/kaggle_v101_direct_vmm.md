# Direct CUDA VMM probe v101: diagnostic PASS

Source `1a00b1da7df33870cd1127c19c11b316bd3ff86c`; two independent processes
on two physical T4s, P2P enabled. Driver 580.178.04, compiler 12.9 V12.9.86,
Compute Sanitizer 2025.1.0.0 build35583870. All ten local/import ×
plain/memcheck/racecheck/initcheck/synccheck pairs pass with exit codes 0/0
and no timeout. Both import-initcheck logs have the required completion
marker and ERROR SUMMARY zero. All twenty rank logs plus commands and
environment evidence retained in `build/kaggle-v101-observation`.

Root summary SHA256
`f7a664d71db71d4ac79176108638f82fad7d25caeb0d8fd91fa845f7035cf6aa`.

This narrows the diagnostic: simple initialized CUDA POSIX import does not
reproduce the full runtime initialization failure on this host. It does
not prove NCCL is faulty or close the BFS initcheck gate. The diagnostic
initializes via a Driver API memset before exchange; NCCL initialization
includes its own symmetric mapping/shadow resources and runtime async
memset/copies after import. Those differences remain to be tested.
