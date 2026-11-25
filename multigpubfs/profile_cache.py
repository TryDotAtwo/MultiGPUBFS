"""Best-effort JSON profile cache; never hide failures in the computation itself."""
from concurrent.futures import Future, TimeoutError as FutureTimeout
from contextvars import copy_context
import errno
import json
import os
from pathlib import Path
import tempfile
from threading import BoundedSemaphore, Thread

from .process_control import ProfileBudgetExpired, profile_timeout


_cache_io_slot = BoundedSemaphore(1)


def _bounded_cache_io(operation, busy):
    if profile_timeout(float("inf")) == float("inf"):
        return operation()
    if not _cache_io_slot.acquire(blocking=False):
        return busy

    result = Future()

    def perform():
        error = None
        try:
            profile_timeout(float("inf"))
            value = operation()
        except BaseException as caught:
            error = caught
        finally:
            _cache_io_slot.release()
        if error is None:
            result.set_result(value)
        else:
            result.set_exception(error)

    # An unresponsive filesystem may not return after cancellation. Keep at
    # most one such call alive and let the startup thread resume on time.
    try:
        Thread(target=copy_context().run, args=(perform,), name="mgbfs-profile-cache", daemon=True).start()
    except RuntimeError:
        _cache_io_slot.release()
        raise
    try:
        value = result.result(timeout=profile_timeout(float("inf")))
    except FutureTimeout as error:
        raise ProfileBudgetExpired("AUTOTUNE_STARTUP_DEADLINE") from error
    profile_timeout(float("inf"))
    return value


def read_mapping(path):
    """Return an object or a miss, together with the operation's status."""
    return _bounded_cache_io(lambda: _read_mapping(path),
                             (None, {"status": "UNAVAILABLE", "stage": "read", "errno": errno.EBUSY}))


def _read_mapping(path):
    try:
        raw=Path(path).read_text(encoding='utf-8')
    except FileNotFoundError:
        return None,{'status':'MISS','stage':'read','errno':None}
    except OSError as error:
        return None,{'status':'UNAVAILABLE','stage':'read','errno':error.errno}
    except UnicodeError:
        return None,{'status':'INVALID','stage':'decode','errno':None}
    try:
        value=json.loads(raw)
    except (ValueError,UnicodeError):
        return None,{'status':'INVALID','stage':'parse','errno':None}
    if not isinstance(value,dict):
        return None,{'status':'INVALID','stage':'type','errno':None}
    return value,{'status':'HIT','stage':'read','errno':None}

def write_mapping(path,value,*,prefix):
    """Replace one cache file atomically. Only this attempt's tmp is cleaned."""
    try:
        return _bounded_cache_io(lambda: _write_mapping(path, value, prefix),
                                 {"status": "UNAVAILABLE", "stage": "write", "errno": errno.EBUSY})
    except ProfileBudgetExpired:
        return {"status": "TIMED_OUT", "stage": "write", "errno": None}


def _write_mapping(path, value, prefix):
    path=Path(path)
    fd=None
    temporary=None
    stream=None
    stage='mkdir'
    try:
        path.parent.mkdir(parents=True,exist_ok=True)
        stage='mkstemp'
        fd,temporary=tempfile.mkstemp(prefix=prefix,suffix='.tmp',dir=path.parent)
        stage='open'
        stream=os.fdopen(fd,'w',encoding='utf-8')
        fd=None
        stage='write'
        json.dump(value,stream)
        stage='close'
        stream.close()
        stream=None
        stage='replace'
        profile_timeout(float("inf"))
        os.replace(temporary,path)
        temporary=None
        return {'status':'WRITTEN','stage':'replace','errno':None}
    except OSError as error:
        return {'status':'UNAVAILABLE','stage':stage,'errno':error.errno}
    finally:
        if stream is not None:
            try:stream.close()
            except Exception:pass
        if fd is not None:
            try:os.close(fd)
            except OSError:pass
        if temporary is not None:
            try:os.unlink(temporary)
            except OSError:pass
