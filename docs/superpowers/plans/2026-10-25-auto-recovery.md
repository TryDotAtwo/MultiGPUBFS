# Automatic recovery profile activation

Goal: doctor verifies the ordinary CayleyPy API in a persistent session and installs an identity-bound profile that enable_native() uses automatically.
Architecture: immutable content-addressed cache bundles with checksummed proof and API acceptance; activation validates current host/runtime/environment, preserves explicit options, and restores only its own environment changes on disable.
Constraints: no automatic production restart, no subset fallback, no GPU reset, no local development, no subagents. Diagnostic remains bounded; production has no new deadline.

- [ ] Regression tests for missing/stale/corrupt cache, matching activation, explicit options, repeated calls, live workers and rollback.
- [ ] Implement profile cache and ordinary activation hook.
- [ ] Add real CayleyPy multi-graph persistent check and fresh-process automatic activation gate to doctor; publish cache only after success.
- [ ] Run CPU tests, full 2-GPU recovery and new-process installed-wheel acceptance.
- [ ] Publish source/wheel and evidence on GitHub; remove rental after saving evidence.


## Acceptance
Implemented and published as dev34. 75 release tests passed. Source and installed-wheel diagnostics PASS/READY on 2x3060; separate ordinary enable_native persistent S3/S4/S3 PASS with cached recipe. Evidence: docs/acceptance/recovery-dev34-installed.json. Other hardware remains untested for this change.
