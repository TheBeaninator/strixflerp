#!/bin/bash
# FLERP-SF-CLOSE-K (card 91a9cda544): the StrixFold adapter host's exit on bench box K, one box, no rank 1, no weights.
# Run by pod-kbench k_run in the staged directory (gate/k_close/k_run.json; gate/k_close/stage.sh makes it):
#   fixed/flerp-executor-conformance.strixfold   the host built from the card's branch (leaves by _exit)
#   crash/flerp-executor-conformance.strixfold   the host built before it (returns from main)
#   shim.so                                      tests/cpp/hipblas_exit_shim.c, built on control-node
# Two legs a build:
#   shim     no device: hipBLAS opened and closed before main, the driver's usage (status 2). The crash build dies
#            here of SIGSEGV in libomp.so's fini (139); the fixed build must exit 2.
#   world1   K's GPU: TF_TP_WORLD=1 and a full command line, so flerp_open opens the HIP driver and a device context,
#            native.open refuses the world (BadWorld, before any file of the model is read), the adapter destroys the
#            context and closes the driver, and the driver returns 2. No weights are opened: model_dir and kit are
#            names that do not exist. flerp_close itself does not run on this leg (there is no executor to close).
# What K cannot show: a served request and flerp_close after it need rank 1 (the pair); that is FLERP-SF-PAIRPASS.
# Exit status 0: both legs of the fixed build ended with the host's last line and status 2. run.txt has everything.
set -u
ROCM=${FLERP_ROCM_ROOT:-/data/models/tools/therock-10.2}
export LD_LIBRARY_PATH=$ROCM/lib:$ROCM/lib/llvm/lib
here=$(cd "$(dirname "$0")" && pwd); cd "$here"
: > run.txt; : > status.txt
say() { echo "$*" | tee -a run.txt; }
printf '1\n2\n3\n' > ids.txt
bad=0
leg() {  # build, leg name, then the environment and command
    local build=$1 name=$2; shift 2
    say "=== $build $name: $* $build/flerp-executor-conformance.strixfold"
    timeout 300 env "$@" "$here/$build/flerp-executor-conformance.strixfold" ${ARGS[@]+"${ARGS[@]}"} > leg.txt 2>&1
    local rc=$?
    tail -n 12 leg.txt | cut -c1-400 | tee -a run.txt
    say "exit status $rc ($build $name)"
    echo "$build $name $rc" >> status.txt
    if [ "$build" = fixed ]; then
        [ "$rc" = 2 ] || { say "FAIL: fixed $name: exit status $rc, wanted the driver's 2"; bad=1; }
        tail -n 1 leg.txt | grep -q '^host: main returned 2; ' || { say "FAIL: fixed $name: the last line is not the host's"; bad=1; }
    fi
}
say "host $(hostname) $(date -u +%FT%TZ); ROCm $ROCM; sha256 fixed $(sha256sum fixed/flerp-executor-conformance.strixfold | cut -c1-16) crash $(sha256sum crash/flerp-executor-conformance.strixfold | cut -c1-16)"
for build in crash fixed; do
    ARGS=()
    leg $build shim LD_PRELOAD="$here/shim.so"
    grep -q 'shim: dlclose: 0' leg.txt || { say "FAIL: $build shim: hipBLAS was not opened and closed"; bad=1; }
    ARGS=(--out "$here/out-$build.json" model_dir=/nonexistent/91a9cda544-no-weights kit=/nonexistent/91a9cda544-no-kit --case close --ids "$here/ids.txt" --paths plain --new 2)
    leg $build world1 TF_TP_WORLD=1
    grep -q 'BadWorld' leg.txt || { say "FAIL: $build world1: flerp_open did not get as far as the world check (no device?)"; bad=1; }
done
grep -q '^crash shim 139$' status.txt || say "NOTE: the crash build did not die of SIGSEGV on the shim leg: the fault is not shown on this box"
say "result: $([ $bad = 0 ] && echo PASS || echo FAIL)"
exit $bad
