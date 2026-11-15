"""hipfire builds every kernel from a Rust string constant: `pub const X_SRC: &str = include_str!(...)` or
`concat!("#define ...\n", include_str!(...), ...)` in crates/rdna-compute/src/*.rs. This module recovers those compile
units (name, ordered parts) and classifies kernel sources into families. Standard library only."""
import os, re

CONST = re.compile(r'(?:pub(?:\([a-z]+\))?\s+)?const ([A-Z0-9_]+): &str\s*=\s*(.*?);\n', re.S)
PART = re.compile(r'include_str!\(\s*"(?:\.\./)+([^"]+)"\s*\)|"((?:[^"\\]|\\.)*)"|\b([A-Z][A-Z0-9_]{3,})\b', re.S)
FOREIGN = re.compile(r'(?<![a-z0-9])gfx(1030|1100|1101|1102|11_dgpu|12|1200|1201|906|908|90a|940|942)(?![0-9])')


def arch_ok(name):
    """True for sources that are generic or gfx1151 / gfx11 (RDNA3 family, which gfx1151 belongs to)."""
    return not FOREIGN.search(name.lower().replace('.', '_').replace('gfx11_dgpu', 'gfx11_dgpu'))


def unescape(s):
    return s.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\').replace('\\t', '\t')


def parse_units(root):
    """{const name: [parts]} over rdna-compute/src; a part is {'file': origin-relative} or {'text': str}."""
    raw = {}
    d = os.path.join(root, 'crates/rdna-compute/src')
    for dp, _, fs in os.walk(d):
        if '/cdna' in dp:
            continue
        for f in sorted(fs):
            if not f.endswith('.rs'):
                continue
            s = open(os.path.join(dp, f), errors='replace').read()
            for m in CONST.finditer(s):
                name, body = m.group(1), m.group(2)
                if 'include_str!' not in body and 'concat!' not in body and not re.fullmatch(r'\s*[A-Z0-9_]+\s*', body):
                    if not body.lstrip().startswith(('"', 'r#"', 'r"')):
                        continue
                parts = []
                rm = re.match(r'\s*r(#*)"(.*)"\1\s*$', body, re.S)
                if rm:
                    parts.append({'text': rm.group(2)})
                else:
                    for t in PART.finditer(body):
                        if t.group(1):
                            p = t.group(1)
                            parts.append({'file': p if p.startswith(('kernels/', 'third_party/')) else None, 'raw': p})
                        elif t.group(2) is not None:
                            parts.append({'text': unescape(t.group(2))})
                        elif t.group(3) not in ('SRC',):
                            parts.append({'const': t.group(3)})
                if parts:
                    raw.setdefault(name, (parts, os.path.relpath(os.path.join(dp, f), root)))
    out = {}

    def resolve(name, depth=0):
        parts, where = raw[name]
        res = []
        for p in parts:
            if 'const' in p:
                if p['const'] in raw and depth < 4:
                    r = resolve(p['const'], depth + 1)
                    if r is None:
                        return None
                    res += r
                else:
                    return None
            elif 'file' in p:
                if p['file'] is None:
                    return None
                res.append({'file': p['file']})
            else:
                res.append(p)
        return res
    for name in raw:
        r = resolve(name)
        if r and any('file' in p and p['file'].endswith('.hip') for p in r):
            out[name] = {'parts': r, 'registry': raw[name][1]}
    return out


# (module name, regex on the kernel source basename); first match wins
FAMILIES = [
    ('LEFTOUT', r'^(vae_|vit_|vl_|attention_flux|gemma4|raon|cross_|modulate|bench_|emu_|calib|quick|tape|test_|patch|imagedec|flux|pflash|rope_2d_flux|apply_rope_2d_vision|layernorm_modulate|fused_gemma4|attention_encoder|.*_FAILED)'),
    ('speculation', r'^(dflash|attention_dflash|dspark|spec_|verify_|draft|sampled|ddtree|chain)'),
    ('ds4_sparse_attn', r'^(deepseek4_attn|indexed_attention|triattn|compressor_)'),
    ('indexer_topk', r'^(indexer_|deepseek4_topk|topk8|qsa)'),
    ('moe', r'^(moe_|deepseek4_moe|hash_router|fp8b128_moe|ep_gather|ep_|qwen4_moe|qwen4_shared|qwen4_slab|ffn_|grouped)|.*_moe_'),
    ('gemv_hfq', r'^(gemv|fused_(qkv|qkvza|qk|gate)).*(hfq|hfp|paro)'),
    ('gemv_mq', r'^(gemv|fused_(qkv|qkvza|qk|gate)|qwen4_gemv).*mq'),
    ('gemv_mfp4', r'^(gemv|fused_(qkv|qkvza|qk|gate)).*mfp\d'),
    ('gemv_ggml', r'^(gemv|fused_(qkv|qkvza|qk|gate)|qwen4_gemv).*(q4k|q8|q4f16|q6k|q5k|q4as8|q4lut|q4wave|tq2|bq1)|^ggml_'),
    ('gemv_float', r'^(gemv|fused_(qkv|qkvza|qk|gate)|qwen4_gemv)'),
    ('gemm_hfq', r'^(gemm|qwen4_gemm).*(hfq|hfp|paro)'),
    ('gemm_mq', r'^(gemm|qwen4_gemm).*mq|^(act_quant|block_i8|mq4v2_fp8|requant)'),
    ('gemm_mfp4', r'^(gemm|qwen4_gemm).*mfp4|^dequantize_mfp4'),
    ('gemm_ggml', r'^(gemm|qwen4_gemm).*(q8|q4k|tq2|bq1)'),
    ('gemm_float', r'^(gemm|qwen4_gemm)'),
    ('kv_cache', r'^(kv_|swa_ring)'),
    ('attention', r'^(attention[._]|qwen35_fa|latent_attention|flash_attn|causal_|wo_per|qk_|qkv_)'),
    ('hc', r'^(hc_|hyper_|stream_mix|qwen4_hc)'),
    ('gdn', r'^(gated_|gdn_|conv1d_silu|dynamic_|alpha|state_|softplus)'),
    ('norm_rope', r'^(rope[._]|rmsnorm|fused_rmsnorm|givens|mq_rotate|rotate|l2|layernorm|hfq4g256)'),
    ('sampling', r'^(sample|argmax|topk|logit|softmax|greedy|max_|select)'),
    ('embedding', r'^embedding'),
    ('tp', r'^(tp_|tp4|slowlink|railgun|device_)'),
    ('tensor_ops', r'.*'),
]
_FAM = [(n, re.compile(r)) for n, r in FAMILIES]


def family(basename):
    for n, r in _FAM:
        if r.match(basename):
            return n
