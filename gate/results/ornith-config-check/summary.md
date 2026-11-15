# Ornith serving defaults: identity on the 35B proxy (config ornith397.serve)

2026-11-14, control-node GPU, world 4 over loopback TCP (4 processes), 3 greedy chats of 96 tokens, one load an arm. Config applied by
flerplib.configs (settings: drafts 1 on the serve path, --delivery tokens, FLERP_MOE_GROUPED=1).

**PASS.** token ids and text of the `default` arm (the named config, no flags) equal the `explicit` arm (the same settings spelled out by
hand) and the `nodrafts` arm (speculation off, grouped prefill on) on 3/3 prompts, and also equal the `old` arm (drafts off, per-token
prompt path, auto delivery) on 3/3 prompts in a second run (old-run/). Every process of every arm exited 0.

Not shown: depth 3. At --drafts 3 the serve front end fails at the first decode step ("flerp_draft_hold: 3 drafts asked, the last stage
holds 1"); card 3f6d20b703. The first run's old arm failed to bind a port (40530, runner port plan), rerun in old-run/.

Seconds a chat: default [3.66, 1.6, 1.51], explicit [2.68, 1.59, 1.51], nodrafts [1.85, 5.24, 1.52]; second run default [1.91, 1.55, 1.47], old [5.08, 1.69, 1.66]
(one GPU shared by four ranks and other pod work: timing is not judged).
