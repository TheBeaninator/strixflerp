#!/bin/bash
# first_light.sh up | down | restart | status | plan | place: the TP2 server on both boxes from serve.conf, run on node1
# (rank 1's box; it reaches node0 by ssh). With WORLD=4 (tp4.conf) the four-node split on NODES (rank g = position g):
# rank 1's box drives, rank 0 is reached as node0 is and ranks 2 and 3 through the gate's rankf verbs (tfp-node0); every
# step below then covers all four boxes (four checks compared, four GPU locks all or nothing, three followers started
# before rank 0, each follower linked, each marked and stopped on down). Both boxes run one deployed commit (deploy_rank0.sh --native, to node0 and --host local);
# rank 0 uses node0's deployed copy of these tools (DEPLOY_DIR/tools/dsv41-hip), rank 1 this box's. TF_SERVE_CONF, when
# set, must name a file at the same place under each box's $HOME.
#   up       a run (RUN_ID, default the UTC time) on both boxes:
#            0. without RUN_ID, the latest run first: still up, up stops there and says so; its rank 0 gone without a
#               down (the OOM killer, a fault), what it left (rank 1, a GPU lock) is taken down (no pkill needed)
#            1. both ranks resolved and checked (check): the same deploy commit, kit manifest and config on both
#            2. the GPU locks, all or nothing: node0's (waiting up to LOCK_WAIT_S), then node1's within
#               LOCK_PAIR_S; if node1's is busy, node0's is released and the pair retried after 30 s (LOCK_TRIES times);
#               held for the run
#            3. rank 1 started (it loads, then waits for rank 0's link), then rank 0
#            4. ready: rank 0's /health and the probe chat request answered (PROBE_EXPECT in the reply), rank 1 linked
#            5. BENCH=1: the timed request inside quiet windows (the optional QUIET_BIN hook on: node0, then node1;
#               off on both after):
#               TTFT, prefill and decode tok/s, the server's /metrics, load on both boxes before and after
#            RUN/first_light.json: both resolved configurations, the probe, the timed run, uptime on both boxes.
#            Any failure takes the run down again (exit 1).
#   down     RUN/stopping written, then rank 0 stopped cleanly (SIGTERM: the engine closes, rank 1 hears done and
#            leaves with exit 0; without the marker a peer that goes away is exit 1), rank 1 stopped if it has not
#            left, both locks released
#   restart  down, then up (a new run)
#   status   every rank, and whether the run still holds each box's GPU lock (a rank's end lets its box's lock go)
#   plan     the rehearsal, on any box (control-node): every rank's resolved launch line (its whole environment and argv,
#            this conf, paths under PLAN_HOME, default /home/pod), the driver's control call for it, and each node's
#            place verify line; at WORLD=4 also which Engram half each node's placement holds (PLACE's ranges files,
#            read-only). Nothing is started, locked or reached over ssh; box-local files (RoPE tables, DEPLOY.json, the
#            rail interface) are named, not read
#   place    on control-node (place verify is a control verb): every placement each rank's box needs (ENGRAM_PLACE_IDS<g>,
#            RANK_PLACE_IDS<g>; WORLD=2: ENGRAM_PLACE_IDS on rank 0's and rank 1's node) re-hashed on its node, all at
#            once; a pass needs exit 0 and the node's own line. Run it before up
set -Eeuo pipefail  # (-E: the ERR trap reaches failures inside functions)
cd "$(dirname "$0")"
source ./serve_lib.sh
case ${1:-} in plan|place) export SERVE_PLAN=1;; esac   # (another box's files are named, not read)
serve_load 1
R0=$RANK0_HOST
# the followers this box reaches (WORLD=4: ranks 2 and 3) and every follower (1 is this box's)
REMOTE=(); for ((g = 2; g < WORLD; g++)); do REMOTE+=("$g"); done
L1=${FL_RANK1:-./serve-rank1.sh}   # (test hook: tp4_orch_test.sh stands in a stub for this box's rank 1, and for ssh)
if [ "$WORLD" = 4 ]; then R0=$RANK_HOST_0; fi
# node0 is reached with the shared worker key, whose forced command (pod-worker-ssh-gate, d74915b5d6) accepts only
# `tfp-node0 rank0 VERB RUN_ID DEPLOY CONF [ARG]`, `tfp-node0 quiet on|off TAG` and `tfp-node0 uptime`: no shell runs
# on node0, so every field is a plain word. DEPLOY and CONF are paths under node0's $HOME (the two homes can differ).
R0_DEPLOY=${DEPLOY_DIR_RAW#\$HOME/}; R0_DEPLOY=${R0_DEPLOY#\~/}
[[ $R0_DEPLOY =~ ^tfp/deploy/[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "DEPLOY_DIR $DEPLOY_DIR_RAW: rank 0's deploy must be \$HOME/tfp/deploy/<name> (the node0 gate allows nothing else)"
R0_CONF=-
if [ -n "${TF_SERVE_CONF:-}" ] && [ "${SERVE_PLAN:-0}" = 1 ]; then
  R0_CONF=tfp/run/conf/$(basename "$TF_SERVE_CONF")   # (plan, place: where the launcher puts it under each box's $HOME)
elif [ -n "${TF_SERVE_CONF:-}" ]; then
  R0_CONF=${TF_SERVE_CONF#$HOME/}
  [[ $R0_CONF =~ ^tfp/run/conf/[A-Za-z0-9][A-Za-z0-9._-]*\.conf$ ]] || die "TF_SERVE_CONF $TF_SERVE_CONF: must be \$HOME/tfp/run/conf/<name>.conf (the node0 gate allows nothing else)"
fi
R0_GATE=$R0_DEPLOY/tools/dsv41-hip/tfp-node0   # the deployed copy; on a gated node0 the gate reads this word itself
r0() { ssh "$R0" "$R0_GATE" rank0 "$1" "$RUN_ID" "$R0_DEPLOY" "$R0_CONF" ${2:+"$2"}; }
# rf G VERB [ARG]: follower G (2, 3) on its box through the same gate (rankf), the same deploy and conf paths
rf() { local h=RANK_HOST_$1; ssh "${!h}" "$R0_GATE" rankf "$1" "$2" "$RUN_ID" "$R0_DEPLOY" "$R0_CONF" ${3:+"$3"}; }
rhost() { local h=RANK_HOST_$1; echo "${!h}"; }

# quiet windows: an optional hook that holds other work on a box while a timed run goes (QUIET_BIN, default
# ~/bin/halo-quiet, called as QUIET_BIN on|off TAG; skipped on a box where it isn't installed). node0 first on, both off.
# (node0's side is always ~/bin/halo-quiet: the gate runs nothing else)
quiet_both() {
  local q='b=${QUIET_BIN:-$HOME/bin/halo-quiet}; [ -x "$b" ] && "$b"' g
  if [ "$1" = on ]; then ssh "$R0" "$R0_GATE" quiet on "$QUIET_TAG" || true; eval "$q on \"\$QUIET_TAG\" || true"
    for g in "${REMOTE[@]}"; do ssh "$(rhost "$g")" "$R0_GATE" quiet on "$QUIET_TAG" || true; done
  else for g in "${REMOTE[@]}"; do ssh "$(rhost "$g")" "$R0_GATE" quiet off "$QUIET_TAG" || true; done
    eval "$q off \"\$QUIET_TAG\" || true"; ssh "$R0" "$R0_GATE" quiet off "$QUIET_TAG" || true; fi
}
# uptimes JSON: every box's load ({"rank0": ..., "rank1": ..., ...})
uptimes() {
  local g s="{\"rank0\": \"$(ssh "$R0" "$R0_GATE" uptime)\", \"rank1\": \"$(uptime)\""
  for g in "${REMOTE[@]}"; do s+=", \"rank$g\": \"$(ssh "$(rhost "$g")" "$R0_GATE" uptime)\""; done
  echo "$s}"
}

down() {
  set +e
  # a quiet window left on by a failure (the requests also expire after 20 minutes)
  [ -n "${QUIET_TAG:-}" ] && quiet_both off
  # a follower reads this marker (its TF_TFP_RUN, serve_lib.sh) and leaves with 0 when rank 0 goes; without it, 1.
  # Every follower's box gets it before rank 0 stops (ranks 2 and 3: the gate's mark)
  : > "$RUN/stopping" 2>/dev/null || echo "first_light down: cannot write $RUN/stopping (rank 1 exits 1)" >&2
  local g
  for g in "${REMOTE[@]}"; do rf "$g" mark || echo "first_light down: no stop marker on rank $g's box (it exits 1)" >&2; done
  r0 stop
  local i
  for i in $(seq 1 "$STOP_TIMEOUT_S"); do [ -z "$(serve_pid "$RUN/rank1.pid")" ] && break; sleep 1; done
  "$L1" stop
  "$L1" unlock
  for g in "${REMOTE[@]}"; do rf "$g" down; done   # stop (SIGTERM, then SIGKILL) + unlock + its tfp-lock<g> units
  r0 down   # rank 0 stop (a no-op by now) + unlock + this run's and any orphaned tfp-lock0 unit
  set -e
}

# the latest run, before a new one: still up, it is said (no new run while it holds both GPUs); its rank 0 gone without a
# down (the kernel's OOM killer, a fault), what it left (a rank 1 that did not leave, a GPU lock still held) is taken down
# first, so the new run never waits out the old one's locks
prior_run() {
  local was st0 st1 g; was=$(serve_latest)
  [ -n "$was" ] || return 0
  st0=$(RUN_ID=$was r0 status 2>/dev/null || true); st1=$(RUN_ID=$was "$L1" status 2>/dev/null || true)
  for g in "${REMOTE[@]}"; do st1+=$'\n'$(RUN_ID=$was rf "$g" status 2>/dev/null || true); done
  if grep -q '^rank 0 running' <<< "$st0"; then die "run $was is up (rank 0 runs on $R0): ./first_light.sh down first, or restart"; fi
  grep -q '^rank 0 not running (ended)' <<< "$st0" || return 0  # (not started yet: another up's, still taking its locks)
  grep -q '^rank [1-7] running\|: held by this run' <<< "$st0"$'\n'"$st1" || return 0
  echo "run $was: rank 0 is gone without a down (its log's last line: $(tail -n1 <<< "$st0")); taking the rest down first"
  RUN_ID=$was RUN=$RUN_ROOT/$was down
}

up() {
  [ -n "${RUN_ID:-}" ] || prior_run
  RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
  export RUN_ID
  RUN=$RUN_ROOT/$RUN_ID
  mkdir -p "$RUN"
  rm -f "$RUN/stopping"   # (a reused RUN_ID: the marker was an earlier down's, not this run's)
  echo "run $RUN_ID: rank 0 on $R0, rank 1 on $(hostname)$(for g in "${REMOTE[@]}"; do printf ', rank %s on %s' "$g" "$(rhost "$g")"; done)"
  r0 check > "$RUN/resolved-rank0.json"
  "$L1" check > /dev/null
  local g
  for g in "${REMOTE[@]}"; do rf "$g" check > "$RUN/resolved-rank$g.json"; done
  python3 - "$WORLD" "$RUN" <<'PY'
import json, os, sys
world, run = int(sys.argv[1]), sys.argv[2]
r = [json.load(open(os.path.join(run, f"resolved-rank{g}.json"))) for g in range(world)]
a = r[0]
for g, b in enumerate(r):
    if b.get("rank") != g:
        sys.exit(f"resolved-rank{g}.json is rank {b.get('rank')}'s")
for g, b in enumerate(r[1:], 1):
    for k in ("deploy", "kit_manifest_sha256", "config_sha256"):
        x, y = (a[k]["commit"], b[k]["commit"]) if k == "deploy" else (a[k], b[k])
        if x != y:
            sys.exit(f"the ranks differ in {k}: rank 0 {x}, rank {g} {y}")
    # the transport by name: one config hash already says so, but a mixed launch (one rank's TF_COLL_KIND tcp proxy or
    # host, the other's rdma; one rank on RCCL) is named here before any rank starts, and the engines' handshake refuses it too
    ea, eb = a["env"], b["env"]
    # at world 4 each rank names its own RDMA device and rail interface (A carries rocep194s0, the others rocep195s0): only the rest must agree
    for k in ("TF_COLL_KIND", "TF_DS_COLL", "TF_TP_STRIPE", "TF_RDMA_GID_INDEX", "NCCL_IB_GID_INDEX") + (("TF_RDMA_DEVICES", "TF_RDMA_IFNAME") if world == 2 else ()):
        if ea.get(k) != eb.get(k):
            sys.exit(f"the ranks differ in {k}: rank 0 {ea.get(k, '(unset)')}, rank {g} {eb.get(k, '(unset)')}"
                     + (" (a mixed transport)" if k in ("TF_COLL_KIND", "TF_DS_COLL") else ""))
hosts = [b.get("host") for b in r]
if len(set(hosts)) != world:
    sys.exit(f"the ranks do not run on {world} different boxes: {hosts}")
ea = a["env"]
kind = ea.get("TF_COLL_KIND", "host")
if ea.get("TF_DS_COLL") == "rccl":
    wire = f"rccl (gid {ea.get('NCCL_IB_GID_INDEX')})"
elif kind == "rdma":
    wire = f"rdma (devices {ea.get('TF_RDMA_DEVICES') or ea.get('TF_RDMA_IFNAME') or 'resolved'}, gid {ea.get('TF_RDMA_GID_INDEX', '3')})"
else:
    wire = f"tcp ({kind})"
print(f"{'both' if world == 2 else f'all {world}'} ranks ({', '.join(hosts)}): deploy {a['deploy']['commit'][:12]}, kit {a['kit_manifest_sha256'][:12]}, config {a['config_sha256'][:12]}, collectives {wire}")
PY
  trap 'echo "first light failed: taking run $RUN_ID down" >&2; down; exit 1' ERR
  # the GPU locks, all or nothing: rank 0's (waiting), then rank 1's and each further follower's within LOCK_PAIR_S;
  # one busy box releases every lock taken (the latest first) and the whole set is retried
  local try held busy
  for try in $(seq 1 "${LOCK_TRIES:-20}"); do
    r0 lock
    held=() busy=
    if "$L1" lock "${LOCK_PAIR_S:-60}"; then
      held=(1)
      for g in "${REMOTE[@]}"; do
        if rf "$g" lock "${LOCK_PAIR_S:-60}"; then held+=("$g"); else busy=$g; break; fi
      done
    else busy=1; fi
    [ -z "$busy" ] && break
    for ((i = ${#held[@]} - 1; i >= 0; i--)); do
      if [ "${held[i]}" = 1 ]; then "$L1" unlock; else rf "${held[i]}" unlock; fi
    done
    r0 unlock
    [ "$try" -lt "${LOCK_TRIES:-20}" ] || { trap - ERR; die "rank $busy's GPU lock stayed busy (${LOCK_TRIES:-20} tries)"; }
    sleep "${LOCK_RETRY_S:-30}"
  done
  # the followers first (each loads, then waits for rank 0's link), rank 0 last
  "$L1" start --locked
  for g in "${REMOTE[@]}"; do rf "$g" start --locked; done
  r0 start --locked
  # (not a pipe: a failure in a pipeline's subshell would take the run down twice)
  r0 ready > "$RUN/rank0-ready.txt" 2>&1 || { cat "$RUN/rank0-ready.txt"; false; }
  cat "$RUN/rank0-ready.txt"
  "$L1" wait-linked
  for g in "${REMOTE[@]}"; do rf "$g" wait-linked; done
  trap - ERR
  grep -m1 '^{' "$RUN/rank0-ready.txt" > "$RUN/probe.json"
  if [ "$BENCH" = 1 ]; then
    # quiet windows on both boxes (node0 first), always paired with off
    QUIET_TAG=first-light-$RUN_ID
    quiet_both on
    uptimes > "$RUN/load-before.json"
    r0 bench > "$RUN/bench-rank0.txt" 2>&1 || echo "the timed run failed (RUN/bench-rank0.txt)" >&2
    uptimes > "$RUN/load-after.json"
    quiet_both off; QUIET_TAG=
    grep -m1 '^{' "$RUN/bench-rank0.txt" > "$RUN/bench.json" || true
    cat "$RUN/bench-rank0.txt"
  fi
  python3 - "$RUN" "$RUN_ID" "$WORLD" "$(uptimes)" <<'PY'
import json, os, sys
run, run_id, world, ups = sys.argv[1], sys.argv[2], int(sys.argv[3]), json.loads(sys.argv[4])
j = lambda n: json.load(open(os.path.join(run, n))) if os.path.exists(os.path.join(run, n)) else None
rec = {"schema": "tf-port-first-light/1", "run": run_id, "world": world}
rec.update({f"rank{g}": j(f"resolved-rank{g}.json") for g in range(world)})
rec.update({"probe": j("probe.json"), "bench": j("bench.json"), "load_before_bench": j("load-before.json"),
            "load_after_bench": j("load-after.json"), "uptime": ups})
json.dump(rec, open(os.path.join(run, "first_light.json"), "w"), indent=1)
PY
  echo "ready: run $RUN_ID ($RUN/first_light.json)"
}

# ---- the rehearsal and the placements (TP4-ORCH a1baab5e2d) ----

# node_of IP: the pod worker whose addresses end in IP's last octet (F .50, B .51, C .52, D .53, A .54)
node_of() { case ${1##*.} in 50) echo F;; 51) echo B;; 52) echo C;; 53) echo D;; 54) echo A;; esac; }
# rank_node G: rank G's box letter (WORLD=4: NODES; a pair: NODE0 / MASTER_IP's box for rank 0, NODE1 for rank 1)
rank_node() {
  local -a n
  if [ "$WORLD" = 4 ]; then read -ra n <<< "$NODES"; echo "${n[$1]}"
  elif [ "$1" = 0 ]; then echo "${NODE0:-$(node_of "$MASTER_IP")}"; else echo "${NODE1:-}"; fi
}
# place_ids G: the placements rank G's box needs, space separated (ENGRAM_PLACE_IDS<G> and RANK_PLACE_IDS<G>; a pair's
# conf names one ENGRAM_PLACE_IDS for both boxes)
place_ids() {
  local e=ENGRAM_PLACE_IDS$1 r=RANK_PLACE_IDS$1 v
  v="${!e:-}"; [ "$WORLD" = 2 ] && v=${v:-${ENGRAM_PLACE_IDS:-}}
  v="$v,${!r:-}"; tr ',' '\n' <<< "$v" | grep -v '^ *$' | tr '\n' ' ' | sed 's/ $//'
}
PLACE=${PLACE_BIN:-/data/models/.swarm/bin/loom-swarm.place}

# plan: the rehearsal. Each rank is resolved in a subshell under SERVE_PLAN=1 with PLAN_HOME as its $HOME, exactly as
# serve-rank0.sh / serve-rank1.sh start resolves it on its box (serve_load, serve_env, serve_cmd); nothing is started.
plan() {
  local ph=${PLAN_HOME:-/home/pod} g rid=${RUN_ID:-plan-rehearsal} rc=0
  echo "plan: $SERVE_CONF, WORLD=$WORLD, driver = rank 1's box, \$HOME = $ph on every box; nothing started, locked or reached"
  for ((g = 0; g < WORLD; g++)); do
    ( export HOME=$ph; RUN_ID=$rid; serve_load "$g"; RUN=$RUN_ROOT/$rid; serve_env; serve_cmd
      wv=WRAP$g; wrap=(); pin=()
      [ -n "${!wv:-}" ] && read -ra wrap <<< "${!wv}"
      [ -n "${ENGINE_CPUS:-}" ] && pin=(taskset -c "$ENGINE_CPUS")
      where="$(rank_node "$g") (${NODE_IP:-$( [ "$g" = 0 ] && echo "$RANK0_HOST" || echo "$RANK1_HOST")}, rail ${NODE_RAIL:-$( [ "$g" = 0 ] && echo "$MASTER_IP" || echo "?")})"
      [ "$WORLD" = 4 ] && where+=", TP2 rank $HALF of pair $PAIR"
      printf 'launch rank %s on %s in %s:' "$g" "$where" "$RUN"; printf ' %q' "${pin[@]}"; printf ' env -i <env rank %s>' "$g"
      printf ' %q' "${wrap[@]}" "${SERVE_CMD[@]}"; echo
      printf 'env rank %s:' "$g"; printf ' %q' "${SERVE_ENV[@]}"; echo
      if [ "$g" = 1 ]; then echo "control rank 1: ./serve-rank1.sh start --locked (on the driver, $(rank_node 1))"
      elif [ "$g" = 0 ]; then echo "control rank 0: ssh $R0 $R0_GATE rank0 start $rid $R0_DEPLOY $R0_CONF --locked"
      else echo "control rank $g: ssh $(rhost "$g") $R0_GATE rankf $g start $rid $R0_DEPLOY $R0_CONF --locked"; fi
    ) || { echo "plan: rank $g does not resolve (above)"; rc=1; }
  done
  local ids n id line
  for ((g = 0; g < WORLD; g++)); do
    n=$(rank_node "$g"); ids=$(place_ids "$g")
    if [ -z "$ids" ]; then echo "place rank $g $n: no placement named (place refuses: ENGRAM_PLACE_IDS$g / RANK_PLACE_IDS$g)"; rc=1; continue; fi
    line="place rank $g $n:"; for id in $ids; do line+=" $PLACE place verify --id $id --node $n;"; done; echo "${line%;}"
  done
  [ "$WORLD" = 4 ] && { engram_halves || rc=1; }
  echo "plan: $([ "$rc" = 0 ] && echo "every rank resolved" || echo "FAILED (above)")"
  return "$rc"
}

# engram_halves (WORLD=4, read-only): node g = 2p + r reads TP2 rank r's Engram half, so each node's Engram placement
# must hold half r = g % 2. From PLACE's queue (requests.json) and its ranges files: per shard, the nodes of one half
# hold the same ranges, the two halves differ, and half 0's data starts lower in the shard (the columns of TP2 rank 0).
engram_halves() {
  local -a args=() ; local g
  for ((g = 0; g < WORLD; g++)); do local e=ENGRAM_PLACE_IDS$g; args+=("$g:$(rank_node "$g"):${!e:-}"); done
  python3 - "${PLACE_STATE:-/data/pod/place}/requests.json" "${args[@]}" <<'PY'
import hashlib, json, os, sys
path, specs = sys.argv[1], sys.argv[2:]
if not os.path.exists(path):
    print(f"engram: not checked: no PLACE queue at {path} (run plan on control-node)"); sys.exit(1)
q = json.load(open(path)); q = q.get("requests", q) if isinstance(q, dict) else q
reqs = {r["id"]: r for r in (q.values() if isinstance(q, dict) else q) if isinstance(r, dict) and "id" in r}
seen, bad = {}, []   # (shard, half) -> {sha: [ranks]}; first data offset per (shard, half)
first = {}
for s in specs:
    g, node, ids = s.split(":", 2); g = int(g); half = g % 2
    for pid in filter(None, ids.split(",")):
        r = reqs.get(pid)
        if r is None or node not in r.get("nodes", []) or node not in r.get("ranges", {}):
            bad.append(f"rank {g} {node}: request {pid} {'is not queued' if r is None else 'does not place ranges on ' + node}"); continue
        rf = r["ranges"][node]; body = open(rf, "rb").read()
        rows = [tuple(map(int, l.split())) for l in body.decode().split("\n") if l.strip()]
        shard = r["items"][0]["file"]
        sha = hashlib.sha256(body).hexdigest()[:12]
        seen.setdefault((shard, half), {}).setdefault(sha, []).append(f"{g}{node}")
        first.setdefault((shard, half), set()).add(rows[1][0] if len(rows) > 1 else rows[0][0])
        print(f"engram rank {g} {node}: {pid} {shard} ranges {sha} ({sum(n for _, n in rows) / 1e9:.1f} GB), wants TP2 half {half}")
for (shard, half), by in sorted(seen.items()):
    if len(by) != 1:
        bad.append(f"{shard} half {half}: the nodes hold different ranges {by}")
for shard in sorted({k[0] for k in seen}):
    h0, h1 = seen.get((shard, 0)), seen.get((shard, 1))
    if not h0 or not h1:
        bad.append(f"{shard}: no node of half {0 if not h0 else 1}"); continue
    if set(h0) & set(h1):
        bad.append(f"{shard}: half 0 and half 1 nodes hold the same ranges")
    if not min(first[(shard, 0)]) < min(first[(shard, 1)]):
        bad.append(f"{shard}: half 0's nodes ({h0}) do not hold the lower columns: they hold TP2 rank 1's half")
for b in bad:
    print(f"engram: MISMATCH {b}")
print("engram: every node holds its TP2 rank's half" if not bad else "engram: FAILED")
sys.exit(1 if bad else 0)
PY
}

# place_verify (control-node): every rank's placements re-hashed on its node at once; a pass needs exit 0 and the
# node's own line (verify prints nothing for a node the request does not name)
place_verify() {
  local d g n id rc=0 o
  [ -x "$PLACE" ] || die "no place binary at $PLACE (place runs on control-node)"
  d=$(mktemp -d)
  for ((g = 0; g < WORLD; g++)); do n=$(rank_node "$g")
    [ -n "$n" ] && [ -n "$(place_ids "$g")" ] || { echo "place: rank $g (${n:-no node}): no placement named"; rc=1; continue; }
    for id in $(place_ids "$g"); do
      ( timeout "${PLACE_VERIFY_S:-600}" "$PLACE" place verify --id "$id" --node "$n" > "$d/$id.$n" 2>&1; echo $? > "$d/$id.$n.rc" ) &
    done
  done; wait
  for ((g = 0; g < WORLD; g++)); do n=$(rank_node "$g")
    for id in $(place_ids "$g"); do
      o=$(tr '\n' ' ' < "$d/$id.$n" | cut -c1-200)
      if [ "$(cat "$d/$id.$n.rc" 2>/dev/null)" != 0 ]; then rc=1; echo "place: rank $g $n: verify --id $id FAILED (rc $(cat "$d/$id.$n.rc" 2>/dev/null)): $o"
      elif ! grep -q "^$n: " "$d/$id.$n"; then rc=1; echo "place: rank $g $n: request $id does not name $n"
      else echo "place: rank $g $n: $id verified ($o)"; fi
    done
  done
  rm -rf "$d"; return "$rc"
}

case ${1:-} in
  up) up;;
  down)
    RUN_ID=${RUN_ID:-$(serve_latest)}; [ -n "$RUN_ID" ] || die "no run"; RUN=$RUN_ROOT/$RUN_ID; down;;
  restart)
    RUN_ID=${RUN_ID:-$(serve_latest)}
    if [ -n "$RUN_ID" ]; then RUN=$RUN_ROOT/$RUN_ID; down; fi
    unset RUN_ID; up;;
  status)
    RUN_ID=${RUN_ID:-$(serve_latest)}; RUN=$RUN_ROOT/${RUN_ID:-none}
    r0 status || true; "$L1" status
    for g in "${REMOTE[@]}"; do rf "$g" status || true; done;;
  plan) plan;;
  place) place_verify;;
  *) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2;;
esac
