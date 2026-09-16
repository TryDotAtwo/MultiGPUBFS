"""Load the user-approved, Windows-user-bound rental key without logging it."""
import ctypes
from pathlib import Path


def load_key():
    class Blob(ctypes.Structure):
        _fields_ = [('size', ctypes.c_ulong), ('data', ctypes.POINTER(ctypes.c_ubyte))]
    encrypted = (Path.home()/'.config/mgbfs/vast-watchdog.dpapi').read_bytes()
    buffer = ctypes.create_string_buffer(encrypted)
    incoming = Blob(len(encrypted), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    output = Blob()
    crypt = ctypes.WinDLL('crypt32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    if not crypt.CryptUnprotectData(ctypes.byref(incoming), None, None, None, None, 1, ctypes.byref(output)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(output.data, output.size).decode('ascii')
    finally:
        kernel.LocalFree(output.data)
