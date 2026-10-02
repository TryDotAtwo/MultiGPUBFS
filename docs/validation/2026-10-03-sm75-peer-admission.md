# Alternative sm75 host rejected before build/search

Own diagnostic rental53923333, machine95392, two NVIDIA GeForce RTX2080Ti.
Provider driver595.71.05, CUDA12.9.1 image; distinct GPU UUIDs and observed
compute_cap7.5. Each device reports22528MiB; topology NODE, no NVLink.

Fresh libcudart cudaDeviceCanAccessPeer(0,1): status0, allowed0. The admission
probe stops immediately at this rejection. Reverse direction was not tested;
one missing required direction already rejects this protocol configuration.
No build, NCCL communicator, BFS, HF token or state dataset upload occurred.
No fallback transport or kernel/hardware guard was introduced.

Actual rental quoted compute+40GB storage rate0.20844444444444443 USD/h.
Initial absolute watchdog deadline was2026-10-02 23:39:21UTC, PID37856,
identity-checked WAITING. Lease was deleted early after preflight failure:
DELETE_REQUESTED followed by GET-confirmed ABSENT, session41628 exit0.
Watchdog is the backup until it observes absence; no lease extension occurred.
Exact provider billing is not asserted. This consumed only the short startup/
admission interval from the separately authorized additional10USD budget.

Do not rent machine95392 again for unchanged LSA acceptance. This is a failed
sm75 hardware admission, not a BMMA/Tensor test or physical2T4 substitute.
Machine55752 and TitanRTX offers were only quoted, not rented or tested.
