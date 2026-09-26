# S13 HF footer audit: Kaggle v28 stopped before the audit

On 2026-09-27, private notebook `trydotatwo/mgbfs-s11-hf-stream` version 28
ran source `a5f0c80` with `AUDIT_EXISTING_S13_FOOTERS=True`. Kaggle reported
`KernelWorkerStatus.ERROR`. Its log shows `UserSecretsClient().get_secret("HF_TOKEN")`
failed at `/kaggle/src/script.py:44`: Kaggle's secret-service POST returned
HTTP 400, wrapped by `kaggle_web_client` as `ConnectionError`. This occurred
before HF authentication, auditor download, or any Parquet footer read. The
notebook produced no `footer-summary.json`.

The local footer auditor has an independent passing test and the complete
Python suite passed (176 tests, 6 skipped) at source `9f53edc`; neither result
proves the remote dataset inventory. The prior anonymous scan remains bounded
at 5,601/6,228 footers because 627 reads received HTTP 429. Full footer
integrity and independent BFS-state equality are still unverified.

Next investigation should distinguish a Kaggle Secrets service problem from
notebook-specific secret access, without exposing or copying the token. Do not
restart another GPU notebook merely because this one failed; check its access
configuration or the service first. Keep only one 2xT4 Kaggle notebook active
at a time per the current user instruction.
