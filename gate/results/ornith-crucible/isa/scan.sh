set -u
cd /data/projects/strixflerp-ocr/build/isa-kg
export FLERP_ROOT=/data/projects/strixflerp-ocr
for v in 0d5c1fd ec08ae9 wip; do
  for tc in rocm-7.2.4 therock-10.1 therock-10.2; do
    (cd $v && ionice -c3 nice -n 19 python3 $FLERP_ROOT/tools/isa-scan.py compile --toolchain $tc --out ../out/$v/$tc \
      "cmd:moe_kgrouped:/opt/rocm-7.2.4/lib/llvm/bin/clang++ -x hip --offload-arch=gfx1151 -O3 -std=c++17 -DNDEBUG -I src -c src/flerp/moe_kgrouped/moe_kgrouped.hip -o x.o")
    python3 $FLERP_ROOT/tools/isa-scan.py table out/$v/$tc/*.s > out/$v-$tc.tsv
  done
done
ls -la out/*.tsv
