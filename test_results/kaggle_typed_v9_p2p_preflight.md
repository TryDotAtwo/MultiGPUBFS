# Typed v9: unsupported host, not a runtime result

Exact source c221b93d739a7f0cafbc6cf79757c035dcd24677.
Worker COMPLETE, root UNSUPPORTED_HOST. Two physical T4s, but
cudaDeviceCanAccessPeer returned allowed=0 in both directions, status=0.
No typed replay executed. Do not count this as a failed sanitizer, BFS
correctness result or evidence about the instrumenter-selection fix.

Root SHA256 b1aa3ecc9a907b0f0f189211310b4fe59e6223c26139a0e00e98d9ebbc29a41f.
Summary retained under build/kaggle-typed-v9-summary, all small JSON/logs
under build/kaggle-typed-v9-observation before unchanged-source retry.

The unchanged-source v10 also completed with UNSUPPORTED_HOST/P2P0 both
ways, with different GPU UUIDs. No BFS or instrumentation replay executed.
Root retained under build/kaggle-typed-v10-summary; small logs are being
retained before retry. Neither attempt tests runtime correctness.
