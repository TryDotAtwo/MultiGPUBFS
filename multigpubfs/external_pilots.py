"""Coordinate optional segmented trials without borrowing resident workers."""

import json
import subprocess
import time

from .segmented_network import AgreementRelay
from .segmented_profile import select_batch_pilot


_PREFIX = "segmented-profile"
_CLEANUP_SECONDS = 0.75
_MIN_TRIAL_SECONDS = 0.25
_PEER_WATCHDOG_SECONDS = 6.5


def launch_trial(store, label, config, deadline, *, temporary, native, definition, env, rank, world, device):
    from .launch import _job_phase, _receipt

    stop_key = f"{label}/stop"
    try:
        if store.try_get(stop_key, deadline=deadline):
            return None
    except TimeoutError:
        return None
    folder = temporary / label.rsplit("/", 1)[-1]
    folder.mkdir()
    configuration = folder / "launch.json"
    configuration.write_text(json.dumps(config), encoding="utf-8")
    bootstrap = folder / "nccl-id"
    relay = AgreementRelay(store, label, bootstrap, rank, world)
    output = folder / f"rank-{rank}"
    command = [str(native), "graph-rank", str(definition), str(configuration), str(rank), str(bootstrap),
               str(output), "1"]
    child_env = dict(env, RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(device),
                     MGBFS_JOB_PHASE_PATH=str(folder / f"rank-{rank}.phase.json"),
                     MGBFS_GENERIC_VRAM_RESERVE_BYTES=str(config["additional_reserve_bytes"]))
    for key in ("MGBFS_JOB_SEQUENCE", "MGBFS_JOB_CANCEL_PATH", "MGBFS_PROFILE_TERMINAL_EVIDENCE_DEPTH"):
        child_env.pop(key, None)
    with (folder / f"rank-{rank}.log").open("w") as log:
        if time.perf_counter() >= deadline:
            return None
        try:
            proc = subprocess.Popen(command, env=child_env, stdout=log, stderr=subprocess.STDOUT)
        except OSError:
            return None
        delivered = False
        next_stop_poll = 0.0
        try:
            while proc.poll() is None:
                now = time.perf_counter()
                if now >= deadline:
                    return None
                if now >= next_stop_poll:
                    if store.try_get(stop_key, deadline=deadline):
                        return None
                    next_stop_poll = now + 0.05
                phase = (_job_phase(proc, folder, rank) or {}).get("phase", "")
                delivered = relay.bootstrap(bootstrap, phase, delivered, deadline=deadline)
                time.sleep(max(0.0, min(0.01, deadline - time.perf_counter())))
            if proc.returncode or time.perf_counter() >= deadline:
                return None
            return _receipt(output, config["graph_digest"])
        except TimeoutError:
            return None
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=0.35)
                    except subprocess.TimeoutExpired as error:
                        raise RuntimeError("EXTERNAL_PROFILE_CHILD_CLEANUP_UNCONFIRMED_NO_RESTART") from error


def run_batch_trials(store, rank, world, baseline, startup_deadline, admit_cold, admit_resident, launch,
                     *, reuse=None):
    """Choose at rank zero, after all ranks have reaped each trial process."""
    selection_key = f"{_PREFIX}/selection"
    if rank != 0:
        watchdog = time.perf_counter() + _PEER_WATCHDOG_SECONDS
        for index in range(3):
            label = f"{_PREFIX}/trial-{index}"
            ready_at = time.perf_counter()
            store.put(f"{label}/ready/{rank}", True, deadline=watchdog)
            trial = store.await_value(f"{label}/launch", stop_key=selection_key, deadline=watchdog)
            if trial is None:
                break
            local_end = min(watchdog, ready_at + trial["lease_seconds"])
            report = launch(label, trial["config"], local_end) if time.perf_counter() < local_end else None
            # Our conservative child cutoff can precede the coordinator's by
            # a round trip or more. Retry the immutable ACK while it can be read.
            store.put(f"{label}/reaped/{rank}", report, deadline=watchdog)
        selection = store.get(selection_key, deadline=watchdog)
        return selection["admission"], selection["profile"]

    remaining = 5.0 if startup_deadline is None else max(0.0, startup_deadline - time.monotonic())
    end = time.perf_counter() + min(5.0, remaining)
    index = 0

    def probe(batch):
        nonlocal index
        planning_end = end - _CLEANUP_SECONDS - _MIN_TRIAL_SECONDS
        if time.perf_counter() >= planning_end:
            return None
        label = f"{_PREFIX}/trial-{index}"
        ready = f"{label}/ready/"
        try:
            store.put(ready + "0", True, deadline=planning_end)
            while store.try_get(ready, ranks=world, deadline=planning_end) is None:
                time.sleep(min(0.01, max(0.0, planning_end - time.perf_counter())))
        except TimeoutError:
            return None
        # Each peer sampled its local clock before making its Ready visible.
        # The same interval from that earlier sample expires no later than ours.
        lease_reference = time.perf_counter()
        candidate = admit_cold(batch, planning_end)
        if candidate is None:
            return None
        admission, config = candidate
        now = time.perf_counter()
        # Leave time for another candidate, rank acknowledgements and readmission.
        seconds = min(1.75, (end - now - _CLEANUP_SECONDS) / 2)
        if seconds < _MIN_TRIAL_SECONDS:
            return None
        index += 1
        trial_end = now + seconds
        cleanup_end = min(end, trial_end + _CLEANUP_SECONDS)
        launch_key = f"{label}/launch"
        request = {"config": config, "lease_seconds": trial_end - lease_reference}
        stop_sent = False
        try:
            try:
                store.put(launch_key, request, deadline=trial_end)
            except TimeoutError:
                # The PUT may have committed before its reply timed out. Make
                # the retry visible only after the stop, then collect every ack.
                store.put(f"{label}/stop", True, deadline=cleanup_end)
                stop_sent = True
                store.put(launch_key, request, deadline=cleanup_end)
                local = None
            else:
                local = launch(label, config, trial_end)
            reports = [local]
            if local is not None:
                try:
                    for peer in range(1, world):
                        reports.append(store.get(f"{label}/reaped/{peer}", deadline=trial_end))
                except TimeoutError:
                    pass
            if len(reports) != world and not stop_sent:
                store.put(f"{label}/stop", True, deadline=cleanup_end)
            for peer in range(len(reports), world):
                reports.append(store.get(f"{label}/reaped/{peer}", deadline=cleanup_end))
        except TimeoutError as error:
            raise RuntimeError("EXTERNAL_PROFILE_RANK_CLEANUP_UNCONFIRMED_NO_RESTART") from error
        if any(report is None for report in reports):
            return None
        return admission, reports

    cached = reuse(end) if reuse is not None and time.perf_counter() < end else None
    if cached is not None:
        selected, profile = cached
    else:
        selected, profile = select_batch_pilot(baseline, probe)
        if selected is not baseline:
            candidate = admit_resident(selected["plan"]["batch"], end - 0.125)
            if candidate is None:
                profile = dict(profile, status="EXTERNAL_PROFILE_RESIDENT_READMISSION_UNAVAILABLE",
                               selected_batch=baseline["plan"]["batch"])
                selected = baseline
            else:
                selected = candidate
    result = {"admission": selected, "profile": profile}
    # This mandatory decision releases ranks that are waiting for production.
    store.put(selection_key, result, deadline=time.perf_counter() + _CLEANUP_SECONDS)
    return selected, profile
