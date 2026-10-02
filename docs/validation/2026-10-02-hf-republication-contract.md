# HF republication contract verification

Source: `bc641cd`; unrelated working-tree changes preserved.

Executed on Windows with local PyArrow:

```text
python -m unittest discover -s tests -p "test_*hf*.py"
Ran 36 tests in 7.652s: OK
python -m unittest discover -s tests -p test_promote_hf_stream.py
Ran 11 tests in 0.403s: OK (subset of the preceding suite)
```

The legacy-manifest regression accepts a repeat without files[].rows and
rejects a changed remote state SHA-256. Reconciliation pins metadata lookup
and bounded layer metadata download to an immutable revision. Both metadata
variants retain the state size and SHA-256 checks.

These are local contract tests with a fake Hub API, not a remote publication
or full dataset validation. No remote files were changed or state payloads
downloaded. No additional compatibility implementation was needed.

Physical two-rank T4 fault tests, registration initcheck, current two-rank
timeline and paired A/B remain open; these tests do not satisfy those gates.
