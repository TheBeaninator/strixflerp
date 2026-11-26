#!/usr/bin/env python3
import subprocess
import os
import sys
import re
import tempfile

def audit_binary(bin_path):
    if not os.path.exists(bin_path):
        print(f"ERROR: binary not found: {bin_path}")
        sys.exit(1)

    print(f"Auditing kernels in: {bin_path}")
    with tempfile.TemporaryDirectory() as tmpdir:
        fatbin_path = os.path.join(tmpdir, "fatbin.bin")
        cmd_objcopy = ["llvm-objcopy", f"--dump-section=.hip_fatbin={fatbin_path}", bin_path]
        res = subprocess.run(cmd_objcopy, capture_output=True)
        if res.returncode != 0 or not os.path.exists(fatbin_path):
            print(f"Failed to dump .hip_fatbin: {res.stderr.decode('utf-8', errors='ignore')}")
            sys.exit(1)

        with open(fatbin_path, "rb") as f:
            data = f.read()

        magic = b"__CLANG_OFFLOAD_BUNDLE__"
        offsets = []
        idx = 0
        while True:
            pos = data.find(magic, idx)
            if pos == -1:
                break
            offsets.append(pos)
            idx = pos + len(magic)
        offsets.append(len(data))

        all_kernels = {}
        total_kernels = 0
        spilled_kernels = []

        for i in range(len(offsets) - 1):
            chunk = data[offsets[i]:offsets[i+1]]
            b_path = os.path.join(tmpdir, f"b_{i}.bin")
            dev_path = os.path.join(tmpdir, f"dev_{i}.o")
            with open(b_path, "wb") as bf:
                bf.write(chunk)

            cmd_unbundle = [
                "clang-offload-bundler",
                "--type=o",
                "--unbundle",
                f"--inputs={b_path}",
                "--targets=hip-amdgcn-amd-amdhsa--gfx1151",
                f"--outputs={dev_path}"
            ]
            res = subprocess.run(cmd_unbundle, capture_output=True)
            if res.returncode != 0 or not os.path.exists(dev_path):
                continue

            cmd_readelf = ["llvm-readelf", "--notes", dev_path]
            try:
                out = subprocess.check_output(cmd_readelf).decode("utf-8", errors="ignore")
            except Exception as e:
                continue

            kernels = out.split("  - .args:")
            for k in kernels:
                m_name = re.search(r"\.name:\s+([^\n]+)", k)
                if not m_name:
                    continue
                name = m_name.group(1).strip()
                vgpr = re.search(r"\.vgpr_count:\s+(\d+)", k)
                spill = re.search(r"\.vgpr_spill_count:\s+(\d+)", k)
                priv = re.search(r"\.private_segment_fixed_size:\s+(\d+)", k)
                v = int(vgpr.group(1)) if vgpr else 0
                s = int(spill.group(1)) if spill else 0
                p = int(priv.group(1)) if priv else 0
                total_kernels += 1

                cat = "other"
                if "attention" in name: cat = "attention"
                elif "linear_q4_experts" in name: cat = "moe_experts (q4/q5)"
                elif "grouped_wmma" in name: cat = "moe_grouped"
                elif "hc_down" in name: cat = "hc_down"
                elif "linear_q4_kernel" in name: cat = "linear_q4"
                elif "linear_q5_kernel" in name: cat = "linear_q5"
                elif "linear_q6_kernel" in name: cat = "linear_q6"
                elif "linear_q8_kernel" in name: cat = "linear_q8"
                elif "linear_wmma" in name: cat = "linear_wmma"
                elif "delta_rule" in name: cat = "gdn"
                elif "mtp" in name: cat = "mtp"

                if cat not in all_kernels:
                    all_kernels[cat] = {"count": 0, "spills": 0, "max_vgpr": 0, "min_vgpr": 999}
                st = all_kernels[cat]
                st["count"] += 1
                st["max_vgpr"] = max(st["max_vgpr"], v)
                st["min_vgpr"] = min(st["min_vgpr"], v)
                if p > 0:
                    st["spills"] += 1
                    spilled_kernels.append((cat, name, v, s, p))

        print(f"\n=== COMPREHENSIVE KERNEL AUDIT REPORT ({total_kernels} KERNELS) ===")
        header = f"{'Module / Family':<25} | {'Kernels':>7} | {'Spills':>6} | {'VGPR Range':>12}"
        print(header)
        print("-" * len(header))
        for cat, st in sorted(all_kernels.items()):
            vgpr_str = f"{st['min_vgpr']}..{st['max_vgpr']}"
            print(f"{cat:<25} | {st['count']:>7} | {st['spills']:>6} | {vgpr_str:>12}")

        print("\n--- ALL REMAINING SPILLED KERNELS IN ENTIRE BINARY ---")
        if not spilled_kernels:
            print("NONE! 100% Zero Spills across all compiled kernels!")
        else:
            for cat, name, v, s, p in spilled_kernels:
                print(f"[{cat}] VGPR: {v:>3} | Spill: {s:>2} | Private: {p:>3} B | {name[:75]}")

if __name__ == "__main__":
    target_bin = sys.argv[1] if len(sys.argv) > 1 else "/home/cheblan/fastasfuck/build/strix/fastasfuck_server"
    audit_binary(target_bin)
