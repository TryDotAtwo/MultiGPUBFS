"""Retry only transient stale NFS handles, never a submitted computation."""
import errno
import json
import time

def read_json(path):
 for attempt in range(3):
  try:
   return json.loads(path.read_text())
  except FileNotFoundError:
   return None
  except OSError as error:
   if error.errno != errno.ESTALE or attempt == 2: raise
   time.sleep(.01 * (attempt+1))
