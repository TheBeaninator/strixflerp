# format.strixite.strixw: design record

## What it is for
The single container the engine loads: a 128-byte hashed header, a hashed index (metadata plus one record per tensor:
name, encoding, shape, group size, experts, source parts, components), then a data region in which every tensor is
already in the layout its kernel reads. Nothing is converted at load except the Q8 to Q4 draft-head copy
(`format.strixite.q4_from_q8`).

## Format choices it embodies (origin's code)
- Encodings: Q4/Q8 row-major, Q4/Q8 chunk-major, Q5/Q6 row-major, BF16, F32, I64 (`src/formats/strixw.hpp:17-18`).
  Components per tensor: Data, or Q + Scale + Min (`:19`).
- Group size is a per-tensor field of the index, not a file-wide constant (`src/formats/strixw.hpp:42`;
  index encoding `src/formats/strixw.cpp:288-321`).
- Alignment: 4096 bytes per tensor, 256 per component (`src/formats/strixw.hpp:26`; layout `strixw.cpp:265-270`;
  the reader refuses a component that is not 256-aligned `strixw.cpp:490-494`). The loader keeps file offsets mod
  4096, so file alignment becomes device-pointer alignment and the kernels' 16-byte vector-load checks hold
  (`kernel.strixite.linear_q4` `src/kernels/linear_q4.hip:407-408`).
- Hashes: header (`strixw.cpp:428`), index (`:440-441`), and every component (`verify_hashes`, `:541-552`). The
  reader checks header and index on open; component hashes only on `inspect_strixw --verify` and in the writer.
- Metadata strings carry the converter's `layout`, `source_fingerprint`, `shared_expert` and similar; the prompt
  cache keys saved states on `source_fingerprint` plus a hash of `layout`
  (`executor.strixite.qwen4exp_backend` `src/serve/qwen4exp_backend.cpp:225-241`).

## What the fleet file contains
Fleet file, read 2026-11-11 from the header and index of `/cache/models/qwen38-fnext-tpn/weights.strixw` on worker C (the file `strix_server.v070tg` serves; version 4, 74,579,656,704 bytes, converter_git ed4c1f4, created 2026-10-05, `shared_expert = separate`):

| class (converter name) | encoding, group | shape (per tensor) | tensors | bytes |
|---|---|---|---|---|
| routed experts gate/up (`exp_gu`) | Q4 row-major G128 | [655360, 2560], E 512 | 48 + 1 MTP | 42.78 + 0.89 GB |
| routed experts down (`exp_down`) | Q4 row-major **G32** | [1310720, 640], E 512 | 48 + 1 MTP | 25.17 + 0.52 GB |
| shared expert gate/up (`sh_gu`) | Q8 G64 | [1280, 2560] | 48 + 1 | 0.17 GB |
| shared expert down (`sh_down`) | Q8 G32 | [2560, 640] | 48 + 1 | 0.09 GB |
| GDN in_proj, merged (`gdn_in`) | Q4 G128 | [16480, 2560] | 36 | 0.81 GB |
| GDN out_proj (`gdn_out`) | Q8 G64 | [2560, 6144] | 36 | 0.60 GB |
| attention qkv + indexer, merged (`attn_qkv`) | Q8 G64 | [13952, 2560] | 12 + 1 | 0.49 GB |
| attention o_proj (`attn_o`) | Q8 G64 | [2560, 6144] | 12 + 1 | 0.22 GB |
| HC mix_down (`hc_down`) | Q8 row-major G64 | [324, 10240] (mixer [320, 10240]) | 97 + 3 | 0.35 GB |
| HC mix_up (`hc_up`) | **Q8 chunk-major** G64 | [10240, 320] | 97 + 3 | 0.35 GB |
| PLE kv_proj (`ple_kv`) | Q8 G64 | [12800, 2560] | 1 | 0.035 GB |
| LM head (`lm_head`) | Q8 G64 | [248320, 2560] | 1 | 0.68 GB |
| MTP fc (`mtp_fc`) | Q8 G64 | [2560, 2560] | 2 | 0.014 GB |
| embedding | BF16 | [248320, 2560] | 1 | 1.27 GB |
| router + shared gate, merged | BF16 | [513, 2560] | 48 + 1 | 0.13 GB |

The `layout` metadata string is exactly what the live `/health` reports. No Q5, Q6 or Q4 chunk-major tensor is in it.

**Conflict resolved (fleet vs origin docs).** The origin's `strixite docs/weights.md:19-20` describes the published file with
routed-expert down at "4-bit, groups of 128" and the MTP input projections at 4-bit G64, 65.9 GiB in all. The fleet
file has expert down at G32 (5.0 bits per weight against 4.25: +3.8 GB over G128) and `mtp_fc` at Q8 G64; it is
69.5 GiB. Both are valid strixw files; the fleet runs a different `--layout` from the published one. Why G32 was
chosen for the fleet is not recorded in the file (open question).

## Costs and limits (inferred)
- One layout per tensor, fixed at conversion: a prefill-friendlier layout means re-conversion (or a load-time copy).
- Alignment padding is small (4096 per tensor over ~900 tensors, < 4 MB).
- Component hashes are not checked on the serving path; a bit flip in the data region is found only by
  `inspect_strixw --verify`.

## Coupled to
`format.strixite.strixw_loader` (reads it onto the device), `format.strixite.convert_qwen4exp` (writes it),
the per-encoding format modules (`q4`, `q5`, `q6`, `q8`: component byte sizes), `tool.strixite.inspect_strixw`.

## Prefill remarks
The container is neutral; prefill is shaped by the encodings it stores (row-major Q4 with separate code / scale /
min planes). A new encoding id (e.g. a WMMA-tile Q4) would be a format version bump plus a converter class.
