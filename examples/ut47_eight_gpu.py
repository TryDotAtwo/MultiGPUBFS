"""Ordinary UT(4,47) calculation on eight visible GPUs; no diagnostic sweep."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time
import traceback

def definition():
    import numpy as np
    from cayleypy import CayleyGraphDef, MatrixGenerator
    generators = []
    for value in (1, 46):
        for row in range(3):
            matrix = np.eye(4, dtype=np.int64)
            matrix[row, row + 1] = value
            generators.append(MatrixGenerator(matrix, modulo=47))
    return CayleyGraphDef.for_matrix_group(
        generators=generators, central_state=np.eye(4, dtype=np.int64), name="UT(4,47)")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=3600,
                        help="Total bounded work budget, including autotuning")
    args = parser.parse_args(argv)
    if args.seconds < 1:
        parser.error("--seconds must be positive")
    if importlib.metadata.version("multigpubfs") != "0.2.0.dev30":
        raise RuntimeError("This pinned example requires multigpubfs 0.2.0.dev30")
    root = Path(tempfile.mkdtemp(prefix="ut47-dev30-", dir=".")).resolve()
    print("RESULTS:", root, flush=True)
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,NET")
    os.environ["NCCL_DEBUG_FILE"] = str(root / "nccl-%h-%p.log")
    start = time.monotonic()
    try:
        import multigpubfs
        from multigpubfs import NativeOptions, persistent_graph_workers
        from cayleypy import CayleyGraph
        multigpubfs.enable_native(NativeOptions(
            devices=tuple(range(8)), output_root=root,
            max_seconds=args.seconds, autotune=True))
        with persistent_graph_workers():
            result = CayleyGraph(definition(), device="cpu").bfs()
        summary = dict(
            status=result.report["status"], full_graph_computed=result.bfs_completed,
            completed_depth=result.completed_depth, layer_sizes=result.layer_sizes,
            states_in_completed_layers=result.num_vertices(),
            diameter=result.diameter() if result.bfs_completed else None,
            seconds=time.monotonic() - start, output=str(result.output),
            report=result.report)
        (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "report"}, indent=2), flush=True)
        print("RESULTS:", root, flush=True)
    except BaseException:
        (root / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("FAILED; preserved files:", root, flush=True)
        raise

if __name__ == "__main__":
    main()
