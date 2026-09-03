#!/usr/bin/env bash
set -euo pipefail
mkdir -p test_results/generation-tiles-local
cargo fmt --all -- --check
cargo test --locked
cargo test --locked -p mgbfs-cuda --features cuda --test generate -- --test-threads=1
cargo test --locked -p mgbfs-runtime --features cuda --test ping_pong -- --test-threads=1
test_binary() {
  cargo test --locked -p "$1" --features cuda --test "$2" --no-run --message-format=json |
    python3 -c 'import sys,json; rows=[json.loads(line) for line in sys.stdin if line.startswith("{")]; paths=[x["executable"] for x in rows if x.get("reason")=="compiler-artifact" and x.get("executable")]; assert len(paths)==1; print(paths[0])'
}
generation_exe="$(test_binary mgbfs-cuda generate)"
feedback_exe="$(test_binary mgbfs-runtime ping_pong)"
for tool in memcheck racecheck initcheck synccheck; do
  compute-sanitizer --error-exitcode 99 --tool "$tool" "$generation_exe" --test-threads=1 --skip large_batch_crosses_old_grid_y_boundary 2>&1 | tee "test_results/generation-tiles-local/generate-$tool.log"
  compute-sanitizer --error-exitcode 99 --tool "$tool" "$feedback_exe" generation_variants_small_feedback --exact --test-threads=1 2>&1 | tee "test_results/generation-tiles-local/feedback-$tool.log"
done
