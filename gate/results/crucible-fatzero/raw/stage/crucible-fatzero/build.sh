#!/bin/bash
# Builds the three trees in parallel on K (each one hipcc process, as the Crucible gate does). Logs in <tree>/build/.
TR=/data/models/tools/therock-10.1
b() {
  mkdir -p $1/build
  t0=$(date +%s)
  (cd $1 && env -u HIPCC_COMPILE_FLAGS_APPEND -u HIPCC_LINK_FLAGS_APPEND -u MAKEFLAGS make -j1 ROCM_PATH=$2 HIPCC=$2/bin/hipcc \
     SPLIT=0 STATE_EPOCH=0 GPU_ARCH=gfx1151 build/engine/crucible.mono) > $1/build/kb-build.log 2>&1
  rc=$?
  echo "build $1 toolchain $2 rc=$rc in $(( $(date +%s) - t0 )) s"
  [ $rc = 0 ] && { sha256sum $1/build/engine/crucible.mono; readelf -d $1/build/engine/crucible.mono | grep RUNPATH; } || tail -20 $1/build/kb-build.log
}
b stock101 $TR & b fatzero101 $TR & b stock724 /opt/rocm-7.2.4 &
wait
