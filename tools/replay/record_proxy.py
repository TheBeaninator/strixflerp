#!/usr/bin/env python3
"""A pass-through recorder for the FLERP-SWAP replay corpus (docs/REPLAY.md).

  record_proxy.py --listen HOST:PORT --upstream http://HOST:PORT --log FILE.jsonl [--target NAME]

Sits between a client (FreeFitty) and one serving node. Every request is forwarded unchanged: method, path, headers
(except the hop-by-hop ones a proxy must not pass on) and body bytes; the upstream's status, headers and body come
back unchanged, streamed as they arrive (SSE and 1 GB cache states are never held in memory). After each exchange one
JSON line is appended to the log, in the corpus's exchange schema (the one tools/replay/scrub.py and gate/replay.py
read): id, target, server, client, method, path, t_start, t_req_end, t_end, req_headers (minus authorization, cookie,
x-api-key, proxy-authorization), request (parsed JSON) or request_raw, status, resp_headers, response_raw,
in_flight_at_start, req_complete, resp_complete. Bodies of /cache/export and /cache/import are logged as size and
sha256 only (req_body_bytes / req_body_sha256, resp_body_bytes / resp_body_sha256).

The log holds user conversations: its directory is created mode 700 and the file mode 600. Python stdlib only.
"""
import argparse
import hashlib
import http.client
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import urllib.parse

BULK = ("/cache/export", "/cache/import")
HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authenticate"}
SECRET_HEADERS = {"authorization", "cookie", "x-api-key", "proxy-authorization", "set-cookie"}
CHUNK = 1 << 16


class Log:
    def __init__(self, path):
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, mode=0o700, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        os.fchmod(fd, 0o600)
        self.f = os.fdopen(fd, "a", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, rec):
        line = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        with self.lock:
            self.f.write(line + "\n")
            self.f.flush()


class State:
    def __init__(self, upstream, target, log):
        u = urllib.parse.urlsplit(upstream)
        if u.scheme != "http" or not u.hostname:
            raise SystemExit(f"--upstream must be http://host:port, not {upstream}")
        self.host, self.port = u.hostname, u.port or 80
        self.server = f"{self.host}:{self.port}"
        self.target, self.log = target, log
        self.lock = threading.Lock()
        self.seq = 0
        self.in_flight = 0

    def begin(self, counted):
        with self.lock:
            self.seq += 1
            n = self.in_flight
            if counted:
                self.in_flight += 1
            return f"{self.target}-{self.seq:06d}", n

    def end(self, counted):
        if counted:
            with self.lock:
                self.in_flight -= 1


class Body:
    """Reads a request body from the client (Content-Length or chunked) as an iterable of chunks, hashing it and
    keeping a copy unless the route is bulk."""

    def __init__(self, rfile, headers, keep):
        self.rfile, self.keep = rfile, keep
        self.length = headers.get("Content-Length")
        self.chunked = "chunked" in (headers.get("Transfer-Encoding") or "").lower()
        self.sha, self.n, self.copy, self.complete = hashlib.sha256(), 0, bytearray(), True

    def _take(self, b):
        self.sha.update(b)
        self.n += len(b)
        if self.keep:
            self.copy += b
        return b

    def __iter__(self):
        if self.chunked:
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return
                left = size
                while left:
                    b = self.rfile.read(min(left, CHUNK))
                    if not b:
                        self.complete = False
                        return
                    left -= len(b)
                    yield self._take(b)
                self.rfile.readline()
        elif self.length is not None:
            left = int(self.length)
            while left:
                b = self.rfile.read(min(left, CHUNK))
                if not b:
                    self.complete = False
                    return
                left -= len(b)
                yield self._take(b)


def parse_json(raw):
    try:
        return json.loads(raw.decode("utf-8")), None
    except (ValueError, UnicodeDecodeError):
        return None, raw.decode("utf-8", "replace")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    st = None  # State

    def log_message(self, fmt, *args):
        pass

    def _proxy(self):
        st = self.st
        route = self.path.split("?")[0]
        bulk = route in BULK
        counted = self.command == "POST" and route.startswith("/v1/")
        eid, in_flight = st.begin(counted)
        t_start = time.time()
        rec = {"id": eid, "target": st.target, "server": st.server, "client": f"{self.client_address[0]}:{self.client_address[1]}",
               "method": self.command, "path": self.path, "t_start": t_start, "in_flight_at_start": in_flight,
               "req_headers": {k.lower(): v for k, v in self.headers.items() if k.lower() not in SECRET_HEADERS}}
        try:
            body = Body(self.rfile, self.headers, keep=not bulk)
            has_body = body.chunked or body.length is not None
            up = http.client.HTTPConnection(st.host, st.port, timeout=3600)
            up.putrequest(self.command, self.path, skip_host=True, skip_accept_encoding=True)
            for k, v in self.headers.items():
                if k.lower() not in HOP:
                    up.putheader(k, v)
            if body.chunked:
                up.putheader("Transfer-Encoding", "chunked")
            up.endheaders()
            if has_body:
                for b in body:
                    up.send(b"%x\r\n%s\r\n" % (len(b), b) if body.chunked else b)
                if body.chunked:
                    up.send(b"0\r\n\r\n")
            rec["t_req_end"] = time.time()
            rec["req_complete"] = body.complete
            if bulk:
                rec["req_body_bytes"], rec["req_body_sha256"] = body.n, body.sha.hexdigest()
            elif body.n:
                rec["request"], raw = parse_json(bytes(body.copy))
                if raw is not None:
                    rec["request_raw"] = raw
                    del rec["request"]
            else:
                rec["request"] = None
            resp = up.getresponse()
            rec["status"] = resp.status
            rec["resp_headers"] = {k.lower(): v for k, v in resp.getheaders() if k.lower() not in SECRET_HEADERS}
            self.send_response_only(resp.status, resp.reason)
            length = resp.getheader("Content-Length")
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            relay_chunked = length is None and self.command != "HEAD" and resp.status not in (204, 304)
            if relay_chunked:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            sha, n, copy, done = hashlib.sha256(), 0, bytearray(), True
            keep = not (bulk and route == "/cache/export")
            while True:
                b = resp.read1(CHUNK) if hasattr(resp, "read1") else resp.read(CHUNK)
                if not b:
                    break
                sha.update(b)
                n += len(b)
                if keep:
                    copy += b
                self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b) if relay_chunked else b)
                self.wfile.flush()
            if relay_chunked:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            if length is not None and n != int(length) and self.command != "HEAD":
                done = False
            rec["resp_complete"] = done
            if keep:
                rec["response_raw"] = bytes(copy).decode("utf-8", "replace")
            else:
                rec["resp_body_bytes"], rec["resp_body_sha256"] = n, sha.hexdigest()
            up.close()
        except (OSError, http.client.HTTPException) as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            if "status" not in rec:
                try:
                    msg = json.dumps({"error": {"message": f"record_proxy: upstream: {e}"}}).encode()
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(msg)))
                    self.end_headers()
                    self.wfile.write(msg)
                except OSError:
                    pass
            self.close_connection = True
        finally:
            rec["t_end"] = time.time()
            st.end(counted)
            st.log.write(rec)

    do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = do_PATCH = do_OPTIONS = _proxy


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(listen, upstream, log_path, target):
    host, _, port = listen.rpartition(":")
    handler = type("H", (Handler,), {"st": State(upstream, target, Log(log_path))})
    return Server((host or "127.0.0.1", int(port)), handler)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", required=True, help="HOST:PORT to accept the client on")
    ap.add_argument("--upstream", required=True, help="http://HOST:PORT of the node")
    ap.add_argument("--log", required=True, help="JSONL file the exchanges are appended to (mode 600)")
    ap.add_argument("--target", default="node", help="name written into each exchange (strixite, pair, ...)")
    a = ap.parse_args(argv)
    srv = serve(a.listen, a.upstream, a.log, a.target)
    print(f"record_proxy: {a.listen} -> {a.upstream}, log {a.log}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
