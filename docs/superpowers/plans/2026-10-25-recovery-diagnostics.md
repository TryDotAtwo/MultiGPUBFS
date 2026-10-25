# Automated recovery diagnostic

User-approved scope: one invocation, 30 minute total diagnostic budget, all selected GPUs, one evidence archive and a validated launch recipe. Production BFS remains unlimited. No GPU resets, privilege changes, silent device exclusion or blind BFS restart.

1. Add an isolated recovery supervisor with independent environment/API evidence, safe-first transport candidates and fresh-process health gates after failures.
2. Validate candidate exchange, CPU-oracle BFS and measured UT(4,47) tuning jointly on the entire GPU set; retain every failed attempt.
3. Add bounded pair localization when CUDA remains usable; explicitly record blocked and untested coverage.
4. Produce identity-bound launch recipes, durable phase/log archives and actionable findings. Test the public recipe entry point before recommending it.
5. Fault-injection tests, real cheap GPU end-to-end checks, wheel/publication checks, then release on GitHub. Report hardware coverage honestly.
