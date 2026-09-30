# Forced cleanup is not bounded runtime success

The existing Linux subreaper now reaps terminated children before signalling
remaining descendants. If it sends SIGKILL it emits
PROCESS_SCOPE_FORCED_CLEANUP and returns99 (cleanup failure98 takes priority).
Oracle/sanitizer and fault harnesses reject this marker/code. This preserves
resource cleanup while preventing launcher exit0 from hiding live ranks.

Parser regression reproduced RED for plain and memcheck; parser tests5/5
GREEN. Windows Python suite185 tests OK,8 skipped. Two new real Linux
subreaper tests are among the skips, not counted as Linux verification.
The Linux process ownership/exit contract remains pending actual execution.

Running Kaggle v31 pins928c25e and does NOT contain this instrumentation.
Its SUPERVISED_NO_COMPLETE result cannot establish graceful bounded rank exit.
