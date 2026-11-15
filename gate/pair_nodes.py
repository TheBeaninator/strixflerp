"""Which workers are the StrixFold serving pair right now: rank 0 and rank 1, each {letter, mgmt, rail, rail_nic}, and
the serve URL, resolved from the pod's live serving conf (the one source of truth), never from a constant here.

    from pair_nodes import add_args, from_args
    pair = from_args(ap.parse_args())          # on control-node: reads LIVE_CONF and the dnsmasq worker map

Rules, in order of precedence:
  1. an explicit command-line override (--rank1 IP, --rank1-rail IP, --rank0 IP, --rank0-rail IP, --serve-url URL);
  2. the serving conf: rank 0 = SERVE_HOST (mgmt) and MASTER_IP (rail); rank 1 = NODE1 (letter), NODE1_IP (mgmt),
     NODE1_RAIL (rail); the URL = http://SERVE_HOST:SERVE_PORT;
  3. where the conf lacks NODE1*, exactly the pod launcher's own fallback (pod-infra 99999c2 tools/strixfold-pair.sh:27
     `F=10.10.0.50; A=10.10.0.54; FR=10.20.0.50; AR=10.20.0.54` and :32 `N1=${SP_N1:-$(cv NODE1)}; N1=${N1:-A};
     v=$(cv NODE1_IP); A=${v:-$A}; v=$(cv NODE1_RAIL); AR=${v:-$AR}`), each key independently, and every such
     field is named in the record's "fallback" list and printed: the pair is never A without saying so.
The letter is read from the pod's dnsmasq map (mgmt ip -> ltsp-<letter>); the rail NIC is not a constant either: it
is the interface that routes to the peer's rail address on the node (rail_nic_cmd), filled by probe_nics.
Stdlib only.
"""
import re
import subprocess

LIVE_CONF = "/data/pod/plans/strixfold/tp2-128k.conf"
WORKER_MAP = "/etc/dnsmasq.d/pod-netboot.conf"
LAUNCHER_REF = "pod-infra 99999c2 tools/strixfold-pair.sh:27,32"
# the launcher's fallback for rank 1 when the conf has no NODE1* keys (see the module docstring)
LAUNCHER_RANK1 = {"letter": "A", "mgmt": "10.10.0.54", "rail": "10.20.0.54"}
LAUNCHER_RANK0 = {"letter": "F", "mgmt": "10.10.0.50", "rail": "10.20.0.50"}     # fixed in the launcher (:27)
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]


def conf_values(text):
    """KEY -> value as the launcher's cv() reads it: the last `KEY=value` line, quotes stripped, up to a space."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"""^([A-Za-z_][A-Za-z0-9_]*)=['"]?([^'" ]*)""", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def worker_letters(text):
    """mgmt ip -> letter from dnsmasq `dhcp-host=<mac>,<ip>,ltsp-<letter>` lines."""
    out = {}
    for m in re.finditer(r"^dhcp-host=[^,\n]*,([0-9.]+),ltsp-([a-z])\b", text or "", re.M):
        out[m.group(1)] = m.group(2).upper()
    return out


def resolve(conf_text, map_text="", rank0=None, rank0_rail=None, rank1=None, rank1_rail=None, serve_url=None,
            conf_path=LIVE_CONF):
    """The pair record. Raises ValueError when the conf and the overrides do not name a whole pair."""
    v, letters = conf_values(conf_text), worker_letters(map_text)
    fallback, warnings, source = [], [], {}

    def pick(field, over, key, default=None):
        if over:
            source[field] = "override"
            return over
        if v.get(key):
            source[field] = f"conf {key}"
            return v[key]
        if default is not None:
            source[field] = f"launcher fallback ({LAUNCHER_REF})"
            fallback.append(field)
            return default
        raise ValueError(f"{conf_path} has no {key} and no override was given for {field}")

    r0 = {"mgmt": pick("rank0.mgmt", rank0, "SERVE_HOST"), "rail": pick("rank0.rail", rank0_rail, "MASTER_IP")}
    r1 = {"mgmt": pick("rank1.mgmt", rank1, "NODE1_IP", LAUNCHER_RANK1["mgmt"])}
    if rank1 and not rank1_rail and v.get("NODE1_IP") != rank1:
        raise ValueError(f"--rank1 {rank1} is not the conf's NODE1_IP ({v.get('NODE1_IP') or 'absent'}): give --rank1-rail too")
    r1["rail"] = pick("rank1.rail", rank1_rail, "NODE1_RAIL", LAUNCHER_RANK1["rail"])
    for r in (r0, r1):
        r["letter"], r["rail_nic"] = letters.get(r["mgmt"]), None
    # the letter the launcher itself uses for rank 1's cordon and hold checks (NODE1, else A)
    launcher_n1 = v.get("NODE1") or LAUNCHER_RANK1["letter"]
    if not v.get("NODE1"):
        fallback.append("rank1.launcher_letter")
    if r1["letter"] is None and not rank1:
        r1["letter"] = launcher_n1
        source["rank1.letter"] = "conf NODE1" if v.get("NODE1") else f"launcher fallback ({LAUNCHER_REF})"
    elif r1["letter"] and r1["letter"] != launcher_n1:
        warnings.append(f"the launcher checks cordon/holds on {launcher_n1} (NODE1) but rank 1's mgmt {r1['mgmt']} is {r1['letter']}")
    if r0["mgmt"] == r1["mgmt"] or r0["rail"] == r1["rail"]:
        raise ValueError(f"rank 0 and rank 1 resolve to the same node ({r0}, {r1})")
    if (r0["mgmt"], r0["rail"]) != (LAUNCHER_RANK0["mgmt"], LAUNCHER_RANK0["rail"]):
        warnings.append(f"rank 0 is not the launcher's fixed F ({LAUNCHER_RANK0['mgmt']}/{LAUNCHER_RANK0['rail']})")
    if serve_url:
        url, source["serve_url"] = serve_url, "override"
    else:
        port = v.get("SERVE_PORT")
        if not port:
            port = "18080"
            fallback.append("serve_url.port")
        url, source["serve_url"] = f"http://{r0['mgmt']}:{port}", "conf SERVE_HOST:SERVE_PORT"
    return {"rank0": r0, "rank1": r1, "serve_url": url, "conf": conf_path, "source": source,
            "fallback": fallback, "launcher_rank1_letter": launcher_n1, "warnings": warnings}


def load(conf=LIVE_CONF, worker_map=WORKER_MAP, **over):
    with open(conf) as f:
        text = f.read()
    try:
        with open(worker_map) as f:
            mtext = f.read()
    except OSError:
        mtext = ""
    return resolve(text, mtext, conf_path=conf, **over)


def add_args(ap):
    g = ap.add_argument_group("the pair (default: the live serving conf)")
    g.add_argument("--serving-conf", default=LIVE_CONF, help="the conf the pair is read from")
    g.add_argument("--rank0", metavar="IP", help="rank 0's mgmt address (overrides SERVE_HOST)")
    g.add_argument("--rank0-rail", metavar="IP", help="rank 0's rail address (overrides MASTER_IP)")
    g.add_argument("--rank1", metavar="IP", help="rank 1's mgmt address (overrides NODE1_IP)")
    g.add_argument("--rank1-rail", metavar="IP", help="rank 1's rail address (overrides NODE1_RAIL)")
    g.add_argument("--serve-url", metavar="URL", help="the server (overrides http://SERVE_HOST:SERVE_PORT)")
    return ap


def from_args(a):
    return load(a.serving_conf, rank0=a.rank0, rank0_rail=a.rank0_rail, rank1=a.rank1, rank1_rail=a.rank1_rail,
                serve_url=a.serve_url)


def rail_nic_cmd(peer_rail):
    """Shell, run on a node: the interface its route to the peer's rail address leaves by."""
    return f"ip -o route get {peer_rail} | sed -n 's/.* dev \\([^ ]*\\).*/\\1/p' | head -1"


def probe_nics(pair, timeout=30):
    """Fill each rank's rail_nic over read-only ssh (None if unreadable). Returns the pair."""
    for me, peer in (("rank0", "rank1"), ("rank1", "rank0")):
        try:
            p = subprocess.run(SSH + [f"pod@{pair[me]['mgmt']}", rail_nic_cmd(pair[peer]["rail"])],
                               capture_output=True, text=True, timeout=timeout)
            pair[me]["rail_nic"] = p.stdout.strip() or None
        except (subprocess.TimeoutExpired, OSError):
            pair[me]["rail_nic"] = None
    return pair


def describe(pair):
    """One line per rank, the URL, and every fallback and warning."""
    def one(n, r):
        return (f"rank {n}: {r.get('letter') or '?'} mgmt {r['mgmt']} rail {r['rail']}"
                + (f" nic {r['rail_nic']}" if r.get("rail_nic") else ""))
    lines = [f"pair from {pair['conf']}:", "  " + one(0, pair["rank0"]), "  " + one(1, pair["rank1"]),
             f"  serve {pair['serve_url']}"]
    if pair["fallback"]:
        lines.append(f"  FALLBACK ({LAUNCHER_REF}, the conf lacks the key): {', '.join(pair['fallback'])}")
    lines += [f"  WARNING: {w}" for w in pair["warnings"]]
    return "\n".join(lines)


def bind(pair, *modules):
    """Set the F (rank 0 mgmt), A (rank 1 mgmt) and URL globals the pair scripts use, in each module that has them."""
    for m in modules:
        for name, val in (("F", pair["rank0"]["mgmt"]), ("A", pair["rank1"]["mgmt"]), ("URL", pair["serve_url"]),
                          ("PAIR", pair)):
            if hasattr(m, name):
                setattr(m, name, val)
