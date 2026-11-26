#!/bin/bash
# linger_guard_test.sh: serve_unit_start refuses to start a unit on a box where the user has no Linger (TP4-LINGER
# 7aa8487768, after TP4-PEERDOWN ec91e930f8). Host-only: loginctl, systemd-run and systemctl are stubs on PATH, nothing
# real is started.
#   1  Linger=no: refused (rc != 0), the message names 'loginctl enable-linger' and the user, systemd-run is not called
#   2  Linger=yes: systemd-run is called with the unit and the command, rc 0
#   3  loginctl cannot look the user up (no user manager yet): refused, says Linger=unknown
#   4  loginctl missing from PATH: refused
#   5  LINGER_USER is the user asked about
set -uo pipefail
cd "$(dirname "$0")"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
pass=0 fail=0
ok() { echo "ok   $*"; pass=$((pass + 1)); }
no() { echo "FAIL $*"; fail=$((fail + 1)); }
check() { local what=$1; shift; if "$@"; then ok "$what"; else no "$what"; fi; }
mkdir -p "$T/bin" "$T/log"
cat > "$T/bin/systemd-run" <<'SH'
#!/bin/bash
echo "systemd-run $*" >> "$STUB_LOG"
SH
cat > "$T/bin/systemctl" <<'SH'
#!/bin/bash
exit 0
SH
cat > "$T/bin/loginctl" <<'SH'
#!/bin/bash
echo "loginctl $*" >> "$STUB_LOG"
case $LINGER_STUB in
  yes|no) echo "$LINGER_STUB";;
  fail) echo "Failed to look up user pod: No such process" >&2; exit 1;;
esac
SH
chmod +x "$T/bin/"*
start() {  # start LINGER_STUB [LINGER_USER]: serve_unit_start in a subshell, output in $T/out, calls in $T/calls
  : > "$T/calls"
  ( export STUB_LOG=$T/calls LINGER_STUB=$1 PATH=$T/bin:$PATH; [ -n "${2:-}" ] && export LINGER_USER=$2
    source ./serve_lib.sh; serve_unit_start tfp-test-r1 "$T/log/x.log" sleep 1 ) > "$T/out" 2>&1
}

start no; rc=$?
check "1 Linger=no refused" [ "$rc" != 0 ]
check "1 message names the fix and the user" grep -q "loginctl enable-linger $(id -un)" "$T/out"
check "1 message says Linger=no" grep -q 'Linger=no' "$T/out"
check "1 systemd-run not called" bash -c "! grep -q '^systemd-run' $T/calls"

start yes; rc=$?
check "2 Linger=yes starts the unit" bash -c "[ $rc = 0 ] && grep -q '^systemd-run .*--unit tfp-test-r1 .*-- sleep 1' $T/calls"

start fail; rc=$?
check "3 lookup failure refused as Linger=unknown" bash -c "[ $rc != 0 ] && grep -q 'Linger=unknown' $T/out && ! grep -q '^systemd-run' $T/calls"

( export STUB_LOG=$T/calls LINGER_STUB=yes PATH=$T/nobin; mkdir -p "$T/nobin"; ln -sf "$T/bin/systemd-run" "$T/nobin/"; ln -sf "$T/bin/systemctl" "$T/nobin/"
  for c in id hostname dirname basename; do ln -sf "$(command -v $c)" "$T/nobin/"; done
  source ./serve_lib.sh; serve_unit_start tfp-test-r1 "$T/log/x.log" sleep 1 ) > "$T/out4" 2>&1; rc=$?
check "4 no loginctl: refused, no unit started" bash -c "[ $rc != 0 ] && ! grep -q ^systemd-run $T/calls && grep -q Linger=unknown $T/out4"

start yes other; rc=$?
check "5 LINGER_USER is the user asked about" grep -q 'loginctl show-user other -p Linger' "$T/calls"

echo "linger_guard_test: $pass passed, $fail failed"
[ "$fail" = 0 ]
