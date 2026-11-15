# SPDX-License-Identifier: Apache-2.0 AND MIT
# flerp (runtime.flerp.vision_sidecar): an adapted copy of Strixfold pull request #6 (766df75, aercanozcan),
# zig/src/families/dsv41/vision_cpu.py (Apache-2.0), which follows DeepSeek V4.1 Flash 1017 inference/vision.py at
# 1bae0e10 (MIT, Copyright (c) 2023 DeepSeek; tools/dsv41-hip/assets/DEEPSEEK_VISION_LICENSE). The origin's file is
# catalog/strixfold_pr6/runtime/vision_sidecar, unchanged. Every change here is marked with this module's id: the
# interpreter / torch pin is a setting that is checked, in place of one string. The tower's arithmetic is not touched.
"""On-demand DeepSeek vision with BF16 image rows and optional HIP attention."""

import hashlib
import json
import os
import signal
import struct
import sys


def read_weights(path, expected):
    if len(expected) != 64:
        raise ValueError('a pinned vision SHA256 is required')
    with open(path, 'rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise ValueError('vision weights failed SHA256')
        stream.seek(0)
        n, = struct.unpack('<Q', stream.read(8))
        if n > 1 << 20:
            raise ValueError('oversized vision header')
        header = json.loads(stream.read(n))
        base = 8 + n
        tensors = {}
        for name, spec in header.items():
            if not name.startswith(('vision.', 'aligner.', 'image_')):
                continue
            if spec['dtype'] != 'BF16':
                raise ValueError('unexpected vision dtype')
            start, end = spec['data_offsets']
            stream.seek(base + start)
            raw = bytearray(stream.read(end - start))
            if len(raw) != end - start:
                raise ValueError('truncated vision tensor')
            tensors[name] = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(spec['shape'])
        if len(tensors) != 266:
            raise ValueError('incomplete vision tower')
        return tensors


def rms(x, weight):
    f = x.float()
    return (weight.float() * (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6))).to(x.dtype)


def rotary(x, cos, sin):
    a, b = x.float().chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(x.dtype)


def encode(patches, nh, nw, weights, *, math_attention=False, probe=None, attention=None):
    """The checkpoint's own ViT and aligner; no caption or OCR substitution."""
    w = weights
    def linear(x, prefix, bias=True):
        return F.linear(x, w[prefix + '.weight'], w[prefix + '.bias'] if bias else None)
    n = nh * nw
    x = linear(patches.flatten(1), 'vision.patch_embed.proj')
    inv = 1 / (10000 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
    hp = torch.arange(nh).unsqueeze(1).expand(nh, nw)
    wp = torch.arange(nw).unsqueeze(0).expand(nh, nw)
    angles = (torch.stack((hp, wp), -1).reshape(-1, 2, 1).float() * inv).flatten(1)
    cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
    if probe:
        probe('embed', x)
    for i in range(32):
        p = 'vision.blocks.' + str(i)
        h = rms(x, w[p + '.norm1.weight'])
        q, k, v = (t.reshape(n, 16, 64) for t in linear(h, p + '.attn.wqkv').chunk(3, -1))
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        # Independent query blocks retain all keys and the reference math attention backend.
        if attention is not None:
            out = attention(q, k, v)
        else:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                if math_attention:
                    out = F.scaled_dot_product_attention(q, k, v)
                else:
                    out = torch.cat([F.scaled_dot_product_attention(piece, k, v)
                                     for piece in q.split(128, dim=-2)], dim=-2)
        x = x + linear(out.squeeze(0).transpose(0, 1).reshape(n, 1024), p + '.attn.wo')
        h = rms(x, w[p + '.norm2.weight'])
        gate, up = linear(h, p + '.mlp.w1', False).chunk(2, -1)
        x = x + linear(F.silu(gate) * up, p + '.mlp.w2', False)
        if probe:
            probe('block-' + str(i), x)
    x = rms(x, w['vision.norm.weight'])
    x = x.reshape(nh, nw, 1024).permute(2, 0, 1)
    x = F.pad(x, (0, -nw % 3, 0, -nh % 3))
    x = F.unfold(x.unsqueeze(0), 3, stride=3).squeeze(0).transpose(0, 1)
    x = linear(F.gelu(linear(x, 'aligner.w1')), 'aligner.w2')
    if probe:
        probe('rows', x)
    return x


def span(rows, nh, nw, weights):
    lh, lw = (nh + 2) // 3, (nw + 2) // 3
    newline = weights['image_newline'].reshape(1, 1, 5120).expand(lh, 1, 5120)
    middle = torch.cat((rows.reshape(lh, lw, 5120), newline), 1).reshape(-1, 5120)
    return torch.cat((weights['image_start'].reshape(1, 5120), middle,
                      weights['image_end'].reshape(1, 5120)), 0).contiguous()


def decode_grid(header):
    if len(header) != 32:
        raise ValueError('missing picture grid')
    bh, bw, nh, nw, lh, lw = struct.unpack('<6I', header[:24])
    n = nh * nw
    tokens = lh * (lw + 1) + 2
    if (not 1 <= n <= 9216 or not 1 <= tokens <= 1024 or bh != nh * 14
            or bw != nw * 14 or lh != (nh + 2) // 3 or lw != (nw + 2) // 3):
        raise ValueError('invalid or oversized picture grid')
    return nh, nw, n, tokens


def timeout_seconds():
    timeout = int(os.environ.get('TF_DS_VISION_TIMEOUT', '240'))
    if timeout < 1:
        raise ValueError('vision worker timeout must be positive')
    return timeout


# flerp (runtime.flerp.vision_sidecar): the origin refuses any torch but 2.9.0+cpu (its line 158). Here the pin is
# the setting FLERP_VISION_ENV, 'python=3.12,torch=2.11.0+cpu,numpy=2.3.5': every key it names is compared with
# what this process runs, and any difference refuses. torch must be named. Unset, the setting is the origin's pin.
# Which settings have gate evidence is recorded in the module's vision-env.json and enforced by the admission
# (tools/dsv41-hip/verify_vision.py): this child is started with `-c` and reads no file but the weights.
ORIGIN_PIN = 'torch=2.9.0+cpu'


def environment_setting():
    setting = {}
    for item in os.environ.get('FLERP_VISION_ENV', ORIGIN_PIN).split(','):
        key, _, value = item.partition('=')
        if key not in ('python', 'torch', 'numpy') or not value or key in setting:
            raise ValueError('FLERP_VISION_ENV: expected python=,torch=,numpy= entries, got ' + repr(item))
        setting[key] = value
    if 'torch' not in setting:
        raise ValueError('FLERP_VISION_ENV must name torch')
    return setting


def check_environment(setting):
    import numpy
    actual = {'python': '%d.%d' % sys.version_info[:2], 'torch': torch.__version__, 'numpy': numpy.__version__}
    for key, value in setting.items():
        if actual[key] != value:
            raise RuntimeError('vision environment: %s is %s, the setting requires %s' % (key, actual[key], value))


def main():
    timeout_seconds()
    threads = os.environ.get('TF_DS_VISION_THREADS')
    if threads is not None:
        if int(threads) < 1:
            raise ValueError('vision worker thread count must be positive')
        torch.set_num_threads(int(threads))
    torch.set_num_interop_threads(1)
    inp = sys.stdin.buffer
    nh, nw, n, tokens = decode_grid(inp.read(32))
    backend = sys.argv[3:]
    if len(backend) not in (0, 2):
        raise ValueError('expected a code object and SHA256 for HIP vision')
    raw = bytearray(inp.read(n * 588 * 2 + 1))
    if len(raw) != n * 588 * 2:
        raise ValueError('invalid patch payload length')
    attention = None
    if backend:
        code = load_vision_code(*backend)
        attention = lambda q, k, v: hip_attention(q, k, v, code)
    weights = read_weights(sys.argv[1], sys.argv[2])
    patches = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(n, 588)
    with torch.inference_mode():
        result = span(encode(patches, nh, nw, weights, attention=attention), nh, nw, weights)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('nonfinite vision rows')
    payload = result.view(torch.uint16).numpy().tobytes()
    sys.stdout.buffer.write(struct.pack('<4sIQ', b'VCPU', tokens, len(payload)) + payload)
    sys.stdout.buffer.flush()


if __name__ == '__main__':
    signal.alarm(timeout_seconds())
    setting = environment_setting()  # flerp (runtime.flerp.vision_sidecar): a malformed setting fails before torch loads
    import torch
    import torch.nn.functional as F
    check_environment(setting)  # flerp (runtime.flerp.vision_sidecar): in place of `torch.__version__ != '2.9.0+cpu'`
    main()
