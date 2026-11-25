from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
import socket
import subprocess
import sys
import time

from multigpubfs import external_pilots
from multigpubfs.network_control import ControlStore


def test_delayed_launch_does_not_extend_peer_trial_past_coordinator(tmp_path, monkeypatch):
    first_trial = "segmented-profile/trial-0"
    delay = 0.90

    class DelayedPeer(ControlStore):
        def await_value(self, key, **kwargs):
            request = super().await_value(key, **kwargs)
            if key == f"{first_trial}/launch" and request is not None:
                time.sleep(delay)
            return request

        def try_get(self, key, **kwargs):
            value = super().try_get(key, **kwargs)
            if key == f"{first_trial}/stop" and value:
                if not hasattr(self, "stop_visible_at"):
                    self.stop_visible_at = time.perf_counter() + delay
                if time.perf_counter() < self.stop_visible_at:
                    return None
            return value

    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    coordinator = ControlStore("127.0.0.1", port, 0, "delayed-pilot", 7)
    peer = DelayedPeer("127.0.0.1", port, 1, "delayed-pilot", 7)
    baseline = {"plan": {"batch": 64}}
    config = {"graph_digest": "fixture", "additional_reserve_bytes": 0}
    real_popen = subprocess.Popen
    deadlines = {}

    def child(command, **kwargs):
        return real_popen([sys.executable, "-c", "import time; time.sleep(15)"], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", child)
    startup_deadline = time.monotonic() + 5

    def run(rank, store):
        folder = tmp_path / str(rank)
        folder.mkdir()
        native = partial(external_pilots.launch_trial, store, temporary=folder, native=sys.executable,
                         definition=folder / "unused.json", env=os.environ, rank=rank, world=2, device=0)

        def launch(label, received, deadline):
            deadlines[rank] = deadline
            return native(label, received, deadline)

        return external_pilots.run_batch_trials(store, rank, 2, baseline, startup_deadline,
                                               lambda *_: (baseline, config), lambda *_: None, launch)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(run, 1, peer)
            selected, profile = run(0, coordinator)
            assert pending.result(timeout=2) == (selected, profile)
        assert selected is baseline
        assert profile["attempted_batches"] == [64] and not profile["measured"]
        assert deadlines[1] <= deadlines[0]
    finally:
        coordinator.close()
        peer.close()
