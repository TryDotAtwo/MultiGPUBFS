"""Work deadlines are optional; startup and retirement remain bounded."""
def normalize(seconds):
 if seconds is None: return None
 if type(seconds) is not int or seconds < 0: raise ValueError("INVALID_MAX_SECONDS")
 return seconds or None

def wire(seconds):
 return 0 if seconds is None else seconds

def remaining(seconds, elapsed):
 return None if seconds is None else max(1, seconds-int(elapsed+.999))

def phase_expired(phase, now, seconds):
 name=phase["phase"]
 limit=600 if name in ("preparing","retiring") else (None if seconds is None else seconds+60)
 return name in ("preparing","bfs","retiring") and limit is not None and now-phase["entered"][name]>limit
