#!/bin/bash
# Makes the directory pod-kbench k_run stages on K for FLERP-SF-CLOSE-K (card 91a9cda544), on control-node, and puts
# the job's arguments where the card names them. It submits nothing: the run is
#   pod-kbench k_run --args /data/pod/plans/91a9cda544/k_run.json
# usage: gate/k_close/stage.sh CRASH_HOST [FIXED_HOST]
#   CRASH_HOST  a flerp-executor-conformance.strixfold built before the fix
#   FIXED_HOST  one built from this branch (default: build/dsv41-flash.strixfold-tp2-abi/flerp-executor-conformance.strixfold)
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
crash=${1:?usage: stage.sh CRASH_HOST [FIXED_HOST]}
fixed=${2:-$root/build/dsv41-flash.strixfold-tp2-abi/flerp-executor-conformance.strixfold}
plan=${FLERP_K_CLOSE_PLAN:-/data/pod/plans/91a9cda544}
line='the process leaves by _exit'
grep -qa "$line" "$fixed" || { echo "stage.sh: $fixed is not a fixed build (no '$line')" >&2; exit 1; }
if grep -qa "$line" "$crash"; then echo "stage.sh: $crash is a fixed build, not the crash build" >&2; exit 1; fi
src=$plan/k-src
mkdir -p "$src/fixed" "$src/crash"
install -m 755 "$(readlink -f "$fixed")" "$src/fixed/flerp-executor-conformance.strixfold"
install -m 755 "$(readlink -f "$crash")" "$src/crash/flerp-executor-conformance.strixfold"
install -m 755 "$root/gate/k_close/run.sh" "$src/run.sh"
install -m 644 "$root/tests/cpp/hipblas_exit_shim.c" "$src/hipblas_exit_shim.c"
cc -shared -fPIC -Wall -Werror "$src/hipblas_exit_shim.c" -ldl -o "$src/shim.so"
sed "s#/data/pod/plans/91a9cda544/k-src#$src#" "$root/gate/k_close/k_run.json" > "$plan/k_run.json"
(cd "$src" && sha256sum fixed/* crash/* shim.so run.sh) | tee "$plan/k-src.sha256"
echo "staged $src; args $plan/k_run.json"
