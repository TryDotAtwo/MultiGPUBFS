"""One-shot localhost receiver: store an explicitly approved Vast key with DPAPI.

Never logs the request body or secret. This is Windows-only setup, not part of
the BFS pipeline. The encrypted file is outside the repository.
"""
import http.server
import json
import re
import secrets
import ctypes
from pathlib import Path


def main():
    destination = Path.home()/'.config/mgbfs/vast-watchdog.dpapi'
    if destination.exists():
        raise RuntimeError('Refusing to overwrite existing credential')
    nonce = secrets.token_urlsafe(24)
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path != '/'+nonce:
                self.send_error(404); return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(b'<title>Local Vast credential storage</title><form method="post"><label>Vast API key <input name="key" type="password" autocomplete="off"></label><button>Store encrypted locally</button></form>')

        def do_POST(self):
            from urllib.parse import parse_qs
            size = int(self.headers.get('Content-Length', '0'))
            if self.path != '/'+nonce or not 0 < size <= 4096:
                self.send_error(400); return
            key = parse_qs(self.rfile.read(size).decode()).get('key', [''])[0]
            if not re.fullmatch(r'[A-Za-z0-9_-]{20,512}', key):
                self.send_error(400); return
            if destination.exists():
                self.send_error(409); return
            destination.parent.mkdir(parents=True, exist_ok=True)
            class Blob(ctypes.Structure):
                _fields_ = [('size', ctypes.c_ulong), ('data', ctypes.POINTER(ctypes.c_ubyte))]
            plaintext = key.encode('ascii')
            buffer = ctypes.create_string_buffer(plaintext)
            incoming = Blob(len(plaintext), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
            encrypted, recovered = Blob(), Blob()
            crypt = ctypes.WinDLL('crypt32', use_last_error=True)
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.LocalFree.argtypes = [ctypes.c_void_p]
            kernel.LocalFree.restype = ctypes.c_void_p
            if not crypt.CryptProtectData(ctypes.byref(incoming), None, None, None, None, 1, ctypes.byref(encrypted)):
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if not crypt.CryptUnprotectData(ctypes.byref(encrypted), None, None, None, None, 1, ctypes.byref(recovered)):
                    raise ctypes.WinError(ctypes.get_last_error())
                try:
                    if ctypes.string_at(recovered.data, recovered.size) != plaintext:
                        raise RuntimeError('DPAPI_ROUNDTRIP_FAILED')
                finally:
                    kernel.LocalFree(recovered.data)
                destination.write_bytes(ctypes.string_at(encrypted.data, encrypted.size))
            finally:
                kernel.LocalFree(encrypted.data)
            self.server.stored = True
            self.send_response(200)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(b'STORED_ENCRYPTED_LOCALLY')

    server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
    server.timeout = 1
    server.stored = False
    print(json.dumps({'url':f'http://127.0.0.1:{server.server_port}/{nonce}'}), flush=True)
    import time
    deadline = time.monotonic()+180
    while not server.stored and time.monotonic() < deadline:
        server.handle_request()
    server.server_close()
    if not server.stored:
        raise TimeoutError('Credential was not submitted')
    print('STORED_ENCRYPTED_LOCALLY', flush=True)


if __name__ == '__main__':
    main()
