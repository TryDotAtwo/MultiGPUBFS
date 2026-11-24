"""Default locations for generated files, independent of the working directory."""

from collections import ChainMap
import os
from pathlib import Path
from uuid import uuid4


def artifact_path(category, *, env=None, override=None):
    env = os.environ if env is None else ChainMap(env, os.environ)
    if override and env.get(override):
        return Path(env[override])
    if env.get("MGBFS_ARTIFACT_ROOT"):
        return Path(env["MGBFS_ARTIFACT_ROOT"]) / category

    if os.name == "nt":
        root = Path(env["LOCALAPPDATA"]) if env.get("LOCALAPPDATA") else Path.home() / "AppData" / "Local"
    else:
        root = Path(env["XDG_STATE_HOME"]) if env.get("XDG_STATE_HOME") else Path.home() / ".local" / "state"
    return root / "multigpubfs" / category


def new_benchmark_directory():
    return artifact_path("benchmarks") / uuid4().hex
