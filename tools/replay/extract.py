#!/usr/bin/env python3
"""Passive corpus extractor: reads tcpdump pcap files (link type SLL/SLL2/EN10MB), reassembles TCP streams to the
strixite worker (:5300) and the StrixFold pair (:18080), splits HTTP/1.1 exchanges and writes one JSON per exchange.

  extract.py OUT_DIR PCAP...
Bulk bodies (/cache/export, /cache/import) are stored as size + sha256 only. Secrets are scrubbed by scrub.py rules.
"""
import hashlib, json, os, struct, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scrub

TARGETS = {5300: "strixite", 18080: "pair"}
CAP = int(os.environ.get("FLOW_CAP_MB", "48")) << 20
capped = set()


def packets(path):
    with open(path, "rb") as f:
        hdr = f.read(24)
        if len(hdr) < 24:
            return
        magic = struct.unpack("<I", hdr[:4])[0]
        end = "<" if magic in (0xa1b2c3d4, 0xa1b23c4d) else ">"
        nano = magic in (0xa1b23c4d, 0x4d3cb2a1)
        link = struct.unpack(end + "I", hdr[20:24])[0]
        while True:
            ph = f.read(16)
            if len(ph) < 16:
                return
            ts, tf, incl, _ = struct.unpack(end + "IIII", ph)
            data = f.read(incl)
            if len(data) < incl:
                return
            t = ts + tf / (1e9 if nano else 1e6)
            if link == 113:      # SLL
                proto, off = struct.unpack(">H", data[14:16])[0], 16
            elif link == 276:    # SLL2
                proto, off = struct.unpack(">H", data[0:2])[0], 20
            elif link == 1:
                proto, off = struct.unpack(">H", data[12:14])[0], 14
            else:
                continue
            if proto != 0x0800:
                continue
            ip = data[off:]
            ihl = (ip[0] & 15) * 4
            if ip[9] != 6:
                continue
            tot = struct.unpack(">H", ip[2:4])[0]
            src, dst = ip[12:16], ip[16:20]
            tcp = ip[ihl:tot]
            sp, dp, seq = struct.unpack(">HHI", tcp[:8])
            doff = (tcp[12] >> 4) * 4
            flags = tcp[13]
            yield t, ".".join(map(str, src)), sp, ".".join(map(str, dst)), dp, seq, flags, tcp[doff:]


def streams(paths):
    segs = collections.defaultdict(dict)   # (cli, cport, srv, sport, dir) -> seq -> (t, payload)
    stored = collections.Counter()
    for p in paths:
        for t, s, sp, d, dp, seq, flags, pl in packets(p):
            if dp in TARGETS:
                key, direction = (s, sp, d, dp), "c2s"
            elif sp in TARGETS:
                key, direction = (d, dp, s, sp), "s2c"
            else:
                continue
            if pl:
                k2 = (key, direction)
                if stored[k2] + len(pl) > CAP:   # bulk cache-transfer bodies: stop keeping bytes past CAP per flow
                    capped.add(k2)
                    continue
                if seq not in segs[k2]:
                    segs[k2][seq] = (t, pl); stored[k2] += len(pl)
    out = {}
    for (key, direction), m in segs.items():
        buf, times, nxt = bytearray(), [], None
        for seq in sorted(m, key=lambda x: (x - min(m)) % (1 << 32)):
            t, pl = m[seq]
            if nxt is not None and seq != nxt:
                rel = (seq - nxt) % (1 << 32)
                if rel < (1 << 31):     # gap: data lost
                    buf += b"\x00" * 0; times.append((len(buf), t, "GAP"))
                else:                   # overlap/retransmit
                    cut = (nxt - seq) % (1 << 32)
                    pl = pl[cut:]
                    if not pl:
                        continue
            times.append((len(buf), t, None))
            buf += pl
            nxt = (seq + len(m[seq][1])) % (1 << 32)
        if (key, direction) in capped:
            times.append((len(buf), times[-1][1] if times else 0, "GAP"))
        out.setdefault(key, {})[direction] = (bytes(buf), times)
    return out


def t_at(times, pos):
    best = times[0][1] if times else None
    for p, t, _ in times:
        if p <= pos:
            best = t
        else:
            break
    return best


def gaps(times, a, b):
    return any(g == "GAP" and a <= p < b for p, _, g in times)


def parse_msgs(buf, is_req):
    i, msgs = 0, []
    while i < len(buf):
        he = buf.find(b"\r\n\r\n", i)
        if he < 0:
            break
        head = buf[i:he].decode("latin1").split("\r\n")
        start = head[0]
        if not (start.startswith("HTTP/") if not is_req else start.split(" ")[0] in ("GET", "POST", "PUT", "DELETE", "HEAD")):
            nx = buf.find(b"HTTP/" if not is_req else b"POST ", i + 1)
            if nx < 0:
                break
            i = nx
            continue
        hdrs = {}
        for h in head[1:]:
            k, _, v = h.partition(":")
            hdrs[k.strip().lower()] = v.strip()
        bstart = he + 4
        if "content-length" in hdrs:
            n = int(hdrs["content-length"])
            body, end = buf[bstart:bstart + n], bstart + n
            complete = len(body) == n
        elif hdrs.get("transfer-encoding", "").lower() == "chunked":
            body, j, complete = bytearray(), bstart, False
            while True:
                le = buf.find(b"\r\n", j)
                if le < 0:
                    break
                sz = int(buf[j:le].split(b";")[0] or b"0", 16)
                if sz == 0:
                    j = le + 4; complete = True; break
                body += buf[le + 2:le + 2 + sz]; j = le + 2 + sz + 2
            body, end = bytes(body), j
        elif is_req or start.split(" ")[1] in ("204", "304"):
            body, end, complete = b"", bstart, True
        else:
            body, end, complete = buf[bstart:], len(buf), True
        msgs.append({"start": start, "headers": hdrs, "body": body, "pos": i, "bend": end, "complete": complete})
        i = end
    return msgs


BULK = ("/cache/export", "/cache/import")


def main():
    out_dir, pcaps = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    exch = []
    for key, d in streams(pcaps).items():
        cli, cport, srv, sport = key
        cbuf, ctimes = d.get("c2s", (b"", []))
        sbuf, stimes = d.get("s2c", (b"", []))
        reqs, resps = parse_msgs(cbuf, True), parse_msgs(sbuf, False)
        for k, rq in enumerate(reqs):
            rs = resps[k] if k < len(resps) else None
            method, path = rq["start"].split(" ")[:2]
            bulk = path.split("?")[0] in BULK
            e = {"target": TARGETS[sport], "server": f"{srv}:{sport}", "client": f"{cli}:{cport}",
                 "method": method, "path": path,
                 "t_start": t_at(ctimes, rq["pos"]), "t_req_end": t_at(ctimes, max(rq["bend"] - 1, rq["pos"])),
                 "req_headers": {k2: v for k2, v in rq["headers"].items() if k2 not in ("authorization", "cookie")},
                 "req_complete": rq["complete"] and not gaps(ctimes, rq["pos"], rq["bend"])}
            if bulk:
                e["req_body_bytes"] = len(rq["body"]); e["req_body_sha256"] = hashlib.sha256(rq["body"]).hexdigest()
            else:
                try:
                    e["request"] = json.loads(rq["body"]) if rq["body"] else None
                except Exception:
                    e["request_raw"] = rq["body"].decode("utf-8", "replace")
            if rs:
                e["status"] = int(rs["start"].split(" ")[1])
                e["t_end"] = t_at(stimes, max(rs["bend"] - 1, rs["pos"]))
                e["resp_headers"] = rs["headers"]
                e["resp_complete"] = rs["complete"] and not gaps(stimes, rs["pos"], rs["bend"])
                if bulk and path.startswith("/cache/export"):
                    e["resp_body_bytes"] = len(rs["body"]); e["resp_body_sha256"] = hashlib.sha256(rs["body"]).hexdigest()
                else:
                    e["response_raw"] = rs["body"].decode("utf-8", "replace")
            exch.append(e)
    exch.sort(key=lambda e: e["t_start"] or 0)
    # concurrency: requests in flight on the same server at this one's start
    for e in exch:
        e["in_flight_at_start"] = sum(1 for o in exch if o is not e and o["server"] == e["server"] and o["t_start"]
                                      and o["t_start"] <= e["t_start"] < (o.get("t_end") or 1e18)
                                      and o["method"] == "POST" and o["path"].startswith("/v1/"))
    n = collections.Counter()
    for e in exch:
        if not (e.get("req_complete") and e.get("resp_complete")):
            continue
        scrub.scrub_exchange(e)
        tgt = e["target"]
        n[tgt] += 1
        d = os.path.join(out_dir, tgt)
        os.makedirs(d, mode=0o700, exist_ok=True)
        eid = f"{tgt}-{n[tgt]:05d}"
        e["id"] = eid
        with open(os.path.join(d, eid + ".json"), "w") as f:
            json.dump(e, f, ensure_ascii=False, sort_keys=True)
    print(json.dumps({"exchanges": len(exch), "kept": dict(n), "capped_flows": len(capped),
                      "dropped_incomplete": sum(1 for e in exch if not (e.get("req_complete") and e.get("resp_complete")))}))


if __name__ == "__main__":
    main()
