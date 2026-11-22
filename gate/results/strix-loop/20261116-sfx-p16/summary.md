# STRIX-LOOP run 20261116-sfx-p16: recipe:qwen38-flash-next.strixite-p16 -> **FAIL** (corrected 2026-11-21: the isa scan did not run)

2026-11-17T00:33:12Z. Builds and isa on control-node, kernels and the model on bench box K through pod-kbench (jobs: <bench-runs>/20261117T000124Z-run-pmodel). Base `qwen38-flash-next.strixite-pf`, variant `qwen38-flash-next.strixite-p16`. Prediction: {'identity': 'logits sha256 and greedy ids equal to base on every prompt', 'speed': 'identity on all 8 prompts; prefill ratio 1.02-1.05 at 2048-32768, ~1.00 at 128-512; decode 1.00-1.03 (gate/criteria/sfx-pkg.toml [panel.predict])'}.

- **PASS** full model on K: logits sha256 (last 16 rows) and greedy ids equal to base on every prompt [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]
- **PASS** every bench process on K exited 0
- **FAIL** isa: no kernel of the changed units uses scratch in the variant build (the isa scan did not run: base: isa-scan compile exited 1; base: no assembly (.s) produced; base: base.tsv has no kernel rows; variant: isa-scan compile exited 1; variant: no assembly (.s) produced; variant: variant.tsv has no kernel rows)

Swaps: format.device-loader/strixw/-: format.strixite.strixw_loader -> format.flerp.strixw_loader_p; linear.decode/strixw-q4/bf16: kernel.strixite.linear_q4 -> kernel.flerp.linear_q4p; linear.decode/strixw-q8/bf16: kernel.strixite.linear_q8 -> kernel.flerp.linear_q8p; linear.prefill/strixw-q4/bf16: kernel.strixite.linear_wmma -> kernel.flerp.wmma_q4p; linear.prefill/strixw-q8/bf16: kernel.strixite.linear_wmma -> kernel.flerp.wmma_q4p; moe.experts-decode/strixw-q4/bf16: kernel.strixite.linear_q4 -> kernel.flerp.linear_q4p; moe.experts-prefill/strixw-q4/bf16: kernel.flerp.moe_grouped_pf -> kernel.flerp.moe_grouped_p; moe.experts-prefill/strixw-q5/bf16: kernel.flerp.moe_grouped_pf -> kernel.strixite.moe_grouped; format.container/strixw/-: format.strixite.strixw -> format.strixite.strixw overlay format.flerp.strixw_v5p

Full model on K, process order A B A B, 2 rounds a process, chunk 8192, weights `/data/models/strix-infer/converted/qwen38-fnext-tpn/weights.strixw`:

| prompt | logits = base | greedy = base (first diff) | prefill tok/s base med (min-max) | variant | ratio | outside noise | decode tok/s base / variant | ratio | device GiB |
|---|---|---|---|---|---|---|---|---|---|
| 128 | True | True (None) | 715.621 (714.728-716.515) | 787.65 (786.885-788.415) | 1.1007 | True | 33.916 / 34.026 | 1.0032 | 76.58 / 76.62 |
| 512 | True | True (None) | 1206.34 (1199.011-1213.669) | 1341.499 (1335.751-1347.247) | 1.112 | True | 33.755 / 33.919 | 1.0049 | 76.72 / 76.76 |
| 1024 | True | True (None) | 1371.913 (1370.723-1373.102) | 1471.904 (1468.404-1475.405) | 1.0729 | True | 33.648 / 33.77 | 1.0036 | 76.86 / 76.9 |
| 2048 | True | True (None) | 1465.543 (1464.439-1466.648) | 1533.041 (1532.905-1533.177) | 1.0461 | True | 33.154 / 33.294 | 1.0042 | 77.0 / 77.04 |
| 4096 | True | True (None) | 1530.722 (1530.005-1531.438) | 1636.406 (1634.459-1638.353) | 1.069 | True | 33.128 / 33.241 | 1.0034 | 77.19 / 77.23 |
| 8192 | True | True (None) | 1618.363 (1617.528-1619.198) | 1700.411 (1699.427-1701.395) | 1.0507 | True | 33.116 / 33.24 | 1.0038 | 77.43 / 77.47 |
| 16384 | True | True (None) | 1605.009 (1604.47-1605.547) | 1686.109 (1685.262-1686.955) | 1.0505 | True | 33.104 / 33.229 | 1.0038 | 77.75 / 77.79 |
| 32768 | True | True (None) | 1588.725 (1588.099-1589.351) | 1669.983 (1669.483-1670.483) | 1.0511 | True | 33.064 / 33.19 | 1.0038 | 78.29 / 78.33 |

Processes (K): A1 base rc 0 wall 448.1 s, load <= 323.5 s, read 78.9 GB, device 78.29 GiB, Tctl 20.85->23.85 C; B1 variant rc 0 wall 386.7 s, load <= 279.0 s, read 78.88 GB, device 78.33 GiB, Tctl 25.85->24.85 C; A2 base rc 0 wall 409.0 s, load <= 288.2 s, read 78.71 GB, device 78.29 GiB, Tctl 25.85->24.85 C; B2 variant rc 0 wall 449.7 s, load <= 304.6 s, read 78.72 GB, device 78.33 GiB, Tctl 25.85->23.85 C

K MemAvailable (GB): free_gb_start 93.7; free_gb_after_A1 93.4; free_gb_after_B1 93.3; free_gb_after_A2 93.3; free_gb_after_B2 93.3

ISA scan (units /linear_q4\.hip|linear_q4p\.hip|linear_q8\.hip|linear_q8p\.hip|linear_wmma\.hip|moe_grouped\.hip|moe_grouped_p\.hip|strixw\.cpp|strixw_loader\.cpp|wmma_q4p\.hip/): NOT RUN. The toolchain clang++ path was missing (isa/isa-scan.log), so base.tsv and variant.tsv are empty. The run first recorded this as "0 changed rows; no scratch in either build" and the isa criterion as PASS. That was wrong: the scan did not run, so it says nothing about scratch. The identity and speed rows above stand.

Canary pin (a human runs it, only if every criterion passed and the gain is wanted): `pod-engine pin <letter> strixite_loop_20261116-sfx-p16`

  before: register the serving: a SERVE_strixite_loop_20261116-sfx-p16 block in engines.conf (pod-engine register) whose binary is <checkout>/build/qwen38-flash-next.strixite-p16/ninja/strixflerp-qwen38-flash-next.strixite-p16 copied to /data/models/strix-infer/runtime/bin/ (NFS, replace in place, never mv over a path workers exec)
  before: one worker, the canary letter a human picks; the keeper converges it at its next pass
  before: pod-engine pin <letter> - removes the pin
