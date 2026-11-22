"""Best-effort JSON profile cache; never hide failures in the computation itself."""
import json
import os
import tempfile
from pathlib import Path

def read_mapping(path):
    """Return an object or a miss, together with the operation's status."""
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
