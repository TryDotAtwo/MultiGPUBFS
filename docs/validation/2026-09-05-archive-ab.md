# Compact S11 archive diagnostic, v7

Source d9dc40fbd7adf7eef07be28c27580fe583ec9371, two T4s.
Same build and configuration, fresh processes; archive-on precedes archive-off.
Single observations, not repeated randomized A/B.

| Contract | Search seconds | Run completion seconds |
|---|---:|---:|
| Compact archive, final sync only | 2.203792095 | 9.242790204 |
| No archive, diagnostic | 1.948075663 | not applicable |

Layer counts match. Observed search overhead is about 13.1%.
Writer rank 0: SHA256 3.4999s, writes 0.3133s, final sync 5.0762s.
Writer rank 1: SHA256 3.7356s, writes 0.3495s, final sync 4.8122s.
These worker timings overlap search and must not be added to BFS time.
The earlier per-layer-sync run finished at 8.2017s: no durability speedup
is established by removing intermediate sync calls.

S13 HF editor v15 separately confirmed HF_AUTH_OK, then failed at
ARCHIVE_PIN_RING_FATAL after roughly 36 seconds. This is insufficient
archive consumer throughput/buffering, not evidence of invalid HF credentials.
No completed S13 catalog artifact was promoted. Upload logs report buffer-based
HTTP fallback from Xet; its contribution has not been isolated.

Evidence: test_results/s11-archive-final-sync-v7/ and test_results/hf-editor-v15/.
