| class | ms | share | TFLOPS | % of 59 |
|---|---|---|---|---|
| dense_proj | 434.7 | 30.8% | 24.2 | 41% |
| experts_routed | 411.3 | 29.2% | 20.1 | 34% |
| gdn_core | 183.6 | 13.0% | 2.5 | 4% |
| attention | 92.5 | 6.6% | 14.9 | 25% |
| norms_residual | 72.3 | 5.1% | - | - |
| other (MoeCombineVec4Kernel, the fast MoE combine: G's moe_combine row, 280 ms, is gone) | 70.5 | 5.0% | - | - |
| experts_shared | 54.1 | 3.8% | 19.1 | 32% |
| router | 47.3 | 3.4% | 3.6 | 6% |
| convert_group | 40.0 | 2.8% | - | - |
| embed_head_misc | 3.1 | 0.2% | - | - |
| total GPU | 1409.9 | | 15.5 | |
