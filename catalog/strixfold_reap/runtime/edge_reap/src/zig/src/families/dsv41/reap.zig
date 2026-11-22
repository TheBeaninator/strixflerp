//! Edge-REAP routing statistics, host side. GPU-free: the device kernels are
//! reap.hip, launched from reap_dev.zig. f64 sums and i64 counts, one writer per (layer, expert),
//! so the same traffic yields the same bits. Routing counters cover every committed row. Norms cover
//! a deterministic sample; the export rescales so reap_saliency = raw_sum / max(norm_count, 1) and
//! reap_weighted_norm_sum = reap_saliency * token_count. `sampled` keeps the raw cells.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const moe_layers: u32 = 40;
pub const n_experts: u32 = 384;
pub const topk: u32 = 6;
pub const early_pos: i32 = 64;
pub const hidden: u32 = 5120;
pub const dn_tp2: u32 = 5120;
pub const dn_tp4: u32 = 2560;
pub const rows_cap_decode: u32 = 64;
/// Decode down-y rows. `round.max_rows`, not the pick stride (`rows_cap_decode`).
pub const rows_cap_y: u32 = 48;
/// fillSplit runs two prompt chunks at once. Each has its own pack region; y does not.
pub const pack_banks: u32 = 2;
pub const rows_cap_prompt: u32 = 2048;
pub const max_domains: u32 = 64;
pub const max_name: usize = 32;
/// build.zig.zon `.version`, used when TF_DS_DEPLOY is unset.
pub const zon_version = "1.0.2";
pub const invalid_domain = "reap_domain must match [a-z0-9_.-]{1,32}";

pub const report_keys = [_][]const u8{
    "model_id",                 "num_layers",     "num_experts", "reap_saliency", "reap_weighted_norm_sum",
    "gate_mass_sum",            "token_count",    "max_l2_norm", "early_pos_count", "domain",
    "modality",                 "tokens",         "norm_rows",   "sampled",       "build",
};

pub const Modality = enum(u8) { text = 0, vision = 1, all = 2 };

pub const Config = struct {
    stats: bool = false,
    every: u32 = 8,
    prefill_rows: u32 = 4,
    /// Distinct domains. Each owns two modality slots. Includes "default".
    buckets: u32 = 16,
    dir: []const u8 = "",
    snapshot_min: u32 = 10,
    served_name: []const u8 = "model",
    build: []const u8 = zon_version,
    dir_owned: bool = false,
    name_owned: bool = false,
    build_owned: bool = false,

    pub fn deinit(c: *Config, a: Allocator) void {
        if (c.dir_owned) a.free(c.dir);
        if (c.name_owned) a.free(c.served_name);
        if (c.build_owned) a.free(c.build);
        c.dir_owned = false;
        c.name_owned = false;
        c.build_owned = false;
    }

    pub fn nbucket(c: Config) u32 {
        return c.buckets * 2;
    }
};

pub const EnvIn = struct {
    stats: ?[]const u8 = null,
    every: ?[]const u8 = null,
    prefill_rows: ?[]const u8 = null,
    buckets: ?[]const u8 = null,
    dir: ?[]const u8 = null,
    snapshot_min: ?[]const u8 = null,
    served_name: ?[]const u8 = null,
    deploy: ?[]const u8 = null,
};

fn parseU32(s: ?[]const u8) ?u32 {
    const t = s orelse return null;
    if (t.len == 0) return null;
    return std.fmt.parseInt(u32, t, 10) catch null;
}

/// `stats` is on only for the exact text "1". The other keys apply only then. `every` 0 and
/// `buckets` 0 fall back to the defaults; `buckets` above 64 clamps to 64.
pub fn configFrom(a: Allocator, in: EnvIn) !Config {
    const on = if (in.stats) |s| std.mem.eql(u8, s, "1") else false;
    if (!on) return .{};
    var c: Config = .{ .stats = true };
    errdefer c.deinit(a);
    c.every = parseU32(in.every) orelse 8;
    if (c.every == 0) c.every = 8;
    c.prefill_rows = parseU32(in.prefill_rows) orelse 4;
    c.buckets = parseU32(in.buckets) orelse 16;
    if (c.buckets == 0) c.buckets = 16;
    if (c.buckets > max_domains) c.buckets = max_domains;
    c.snapshot_min = parseU32(in.snapshot_min) orelse 10;
    if (in.dir) |d| if (d.len > 0) {
        c.dir = try a.dupe(u8, d);
        c.dir_owned = true;
    };
    if (in.served_name) |n| if (n.len > 0) {
        c.served_name = try a.dupe(u8, n);
        c.name_owned = true;
    };
    if (in.deploy) |d| if (d.len > 0) {
        c.build = try a.dupe(u8, d[0..@min(d.len, 64)]);
        c.build_owned = true;
    };
    return c;
}

fn envSlice(name: [*:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    const s = std.mem.span(v);
    return if (s.len == 0) null else s;
}

pub fn fromEnv(a: Allocator) !Config {
    return configFrom(a, .{
        .stats = envSlice("TF_DS_REAP_STATS"),
        .every = envSlice("TF_DS_REAP_EVERY"),
        .prefill_rows = envSlice("TF_DS_REAP_PREFILL_ROWS"),
        .buckets = envSlice("TF_DS_REAP_BUCKETS"),
        .dir = envSlice("TF_DS_REAP_DIR"),
        .snapshot_min = envSlice("TF_DS_REAP_SNAPSHOT_MIN"),
        .served_name = envSlice("TF_DS_SERVED_NAME"),
        .deploy = envSlice("TF_DS_DEPLOY"),
    });
}

pub fn statsOn() bool {
    const v = std.c.getenv("TF_DS_REAP_STATS") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// Leading space included, so planText can append it. Empty when stats are off (the stock plan).
pub fn planSuffix(buf: []u8, c: Config) []const u8 {
    if (!c.stats) return "";
    return std.fmt.bufPrint(buf, " reap={d}/{d}/{d}", .{ c.every, c.prefill_rows, c.buckets }) catch "";
}

pub fn domainOk(s: []const u8) bool {
    if (s.len < 1 or s.len > max_name) return false;
    for (s) |ch| {
        const ok = (ch >= 'a' and ch <= 'z') or (ch >= '0' and ch <= '9') or ch == '_' or ch == '.' or ch == '-';
        if (!ok) return false;
    }
    return true;
}

pub fn formatCap(buf: []u8, cap: u32, name: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "reap_domain cap {d}: refusing new domain \"{s}\"", .{ cap, name }) catch buf[0..0];
}

pub const AdmitError = error{ Invalid, Cap };

pub const Domains = struct {
    mu: std.atomic.Mutex = .unlocked,
    cap: u32,
    n: u32 = 0,
    names: [max_domains][max_name]u8 = undefined,
    lens: [max_domains]u8 = @splat(0),

    pub fn init(cap: u32) Domains {
        var d: Domains = .{ .cap = @min(cap, max_domains) };
        _ = d.addAssume("default");
        return d;
    }

    pub fn lock(d: *Domains) void {
        while (!d.mu.tryLock()) std.Thread.yield() catch {};
    }

    pub fn unlock(d: *Domains) void {
        d.mu.unlock();
    }

    fn find(d: *const Domains, raw: []const u8) ?u32 {
        var i: u32 = 0;
        while (i < d.n) : (i += 1) {
            if (d.lens[i] == raw.len and std.mem.eql(u8, d.names[i][0..raw.len], raw)) return i;
        }
        return null;
    }

    fn addAssume(d: *Domains, raw: []const u8) u32 {
        const i = d.n;
        @memset(&d.names[i], 0);
        @memcpy(d.names[i][0..raw.len], raw);
        d.lens[i] = @intCast(raw.len);
        d.n += 1;
        return i;
    }

    pub fn name(d: *const Domains, i: u32) []const u8 {
        return d.names[i][0..d.lens[i]];
    }

    /// Empty is "default". A known name returns its index. A new name past `cap` is `Cap`.
    pub fn admit(d: *Domains, raw: []const u8) AdmitError!u32 {
        const n = if (raw.len == 0) "default" else raw;
        if (!domainOk(n)) return error.Invalid;
        d.lock();
        defer d.unlock();
        if (d.find(n)) |i| return i;
        if (d.n >= d.cap) return error.Cap;
        return d.addAssume(n);
    }

    pub fn lookup(d: *Domains, raw: []const u8) ?u32 {
        const n = if (raw.len == 0) "default" else raw;
        d.lock();
        defer d.unlock();
        return d.find(n);
    }

    /// Names as of this call. The snapshot writer keeps the copy; live admits do not touch it.
    pub fn cloneInto(src: *Domains, dst: *Domains) void {
        src.lock();
        defer src.unlock();
        dst.cap = src.cap;
        dst.n = src.n;
        var i: u32 = 0;
        while (i < src.n) : (i += 1) {
            dst.names[i] = src.names[i];
            dst.lens[i] = src.lens[i];
        }
    }
};

pub fn modalityOf(text: []const u8) ?Modality {
    if (text.len == 0 or std.mem.eql(u8, text, "all")) return .all;
    if (std.mem.eql(u8, text, "text")) return .text;
    if (std.mem.eql(u8, text, "vision")) return .vision;
    return null;
}

pub const ImageSpan = struct { at: u32, len: u32 };

/// 1 when `pos` sits in a span (the router used gate.bias_vl), else 0 (gate.bias).
pub fn modalityAt(pos: u32, spans: []const ImageSpan) u8 {
    for (spans) |s| if (pos >= s.at and pos - s.at < s.len) return 1;
    return 0;
}

pub fn bucketIndex(domain_index: u32, modality: u8) u32 {
    return domain_index * 2 + modality;
}

/// Decode rows are text: the decode router never calls routeSpans.
pub fn fillDecodeBuckets(out: []u8, domain_index: u32) void {
    @memset(out, @intCast(bucketIndex(domain_index, 0)));
}

pub fn fillPromptBuckets(out: []u8, start: u32, domain_index: u32, spans: []const ImageSpan) void {
    const base = domain_index * 2;
    for (out, 0..) |*b, row| {
        const pos = start + @as(u32, @intCast(row));
        b.* = @intCast(base + modalityAt(pos, spans));
    }
}

// --- schedule ----------------------------------------------------------------

pub const HopKind = enum { pair, scalar };

pub const Hop = struct {
    kind: HopKind = .pair,
    peer_xor: u32 = 0,
    bytes: u64 = 0,
};

pub const Hops = struct {
    n: u8 = 0,
    hops: [2]Hop = .{ .{}, .{} },
    /// Pair r = 0 (world >= 2) or the only rank (world 1) reduces the partials.
    ssq: bool = false,
    /// Rank 0 applies the norm to the accumulators.
    norm: bool = false,
};

/// Packed partial bytes: layers * sample_rows * topk * dn * 4.
pub fn pairBytes(layers: u32, sample_rows: u32, dn: u32) u64 {
    return @as(u64, layers) * sample_rows * topk * dn * 4;
}

/// One f64 sum of squares per (layer, sample, slot).
pub fn scalarBytes(layers: u32, sample_rows: u32) u64 {
    return @as(u64, layers) * sample_rows * topk * 8;
}

/// Device bytes of one rank's REAP buffers at open. Rank 0 also holds the six accumulator grids.
/// `sample` is the capped prefill sample (at least 1). `uniform` false skips the y buffer.
pub fn deviceBytes(s: struct {
    route_slots: u32,
    y_slots: u32,
    dn: u32,
    sample: u32 = 4,
    buckets: u32 = 16,
    rank0: bool = true,
    uniform: bool = true,
}) u64 {
    const layers: u64 = moe_layers;
    const route: u64 = s.route_slots;
    const sample: u64 = s.sample;
    var n: u64 = 0;
    const dec_pick = layers * rows_cap_decode * route * 4;
    const pr_pick = 2 * layers * rows_cap_prompt * route * 4;
    n += dec_pick * 2;
    n += pr_pick * 2;
    n += rows_cap_prompt;
    n += rows_cap_prompt * 4;
    n += rows_cap_prompt;
    n += rows_cap_decode;
    n += rows_cap_decode * 4;
    n += rows_cap_decode;
    const vec = layers * sample * topk * s.dn * 4;
    n += vec * pack_banks;
    n += vec;
    n += scalarBytes(moe_layers, s.sample) * 2;
    n += sample * 4 * 2;
    if (s.uniform) n += layers * rows_cap_y * s.y_slots * s.dn * 4;
    if (s.rank0) n += pullBytes(s.buckets);
    return n;
}

/// Bytes copyD2H moves for one refresh: six grids, both modalities of every domain.
pub fn pullBytes(buckets: u32) u64 {
    const cells: u64 = @as(u64, buckets) * 2 * moe_layers * n_experts;
    return cells * 8 * 6;
}

/// The same hop list on every rank. `dn` is this rank's column count (5120 TP2, 2560 TP4).
/// World 1: no exchange. World 2: one pair exchange. World 4: pair, then the scalar hop.
pub fn exchange(world: u32, rank: u32, layers: u32, sample_rows: u32, dn: u32) Hops {
    var h: Hops = .{};
    if (sample_rows == 0 or layers == 0) return h;
    h.norm = rank == 0;
    h.ssq = if (world <= 1) rank == 0 else rank % 2 == 0;
    if (world <= 1) return h;
    h.hops[0] = .{ .kind = .pair, .peer_xor = 1, .bytes = pairBytes(layers, sample_rows, dn) };
    h.n = 1;
    if (world >= 4) {
        h.hops[1] = .{ .kind = .scalar, .peer_xor = 2, .bytes = scalarBytes(layers, sample_rows) };
        h.n = 2;
    }
    return h;
}

/// Round counter starts at 0, so the first round is sampled. Call once per model.verify, every rank.
pub fn decodeSample(round_i: u64, every: u32) bool {
    const e: u64 = if (every == 0) 8 else every;
    return round_i % e == 0;
}

/// At most `kmax` rows of a chunk of `n`, indices i * n / k. Includes row 0. Every row when k >= n.
pub fn promptSample(n: u32, kmax: u32, out: []u32) u32 {
    if (n == 0 or kmax == 0 or out.len == 0) return 0;
    const m: u32 = @min(@min(kmax, n), @as(u32, @intCast(out.len)));
    var i: u32 = 0;
    while (i < m) : (i += 1) out[i] = i * n / m;
    return m;
}

// --- layout (reap.hip uses these) --------------------------------------------

/// src[((layer * rows_cap + row) * slots + slot) * dn + col]
pub fn packSrc(layer: u32, rows_cap: u32, row: u32, slots: u32, slot: u32, dn: u32, col: u32) usize {
    return (((@as(usize, layer) * rows_cap + row) * slots + slot) * dn) + col;
}

/// dst[((layer * n_sample + sample) * topk + slot) * dn + col]
pub fn packDst(layer: u32, n_sample: u32, sample: u32, slot: u32, dn: u32, col: u32) usize {
    return (((@as(usize, layer) * n_sample + sample) * topk + slot) * dn) + col;
}

pub fn ssIndex(layer: u32, n_sample: u32, sample: u32, slot: u32) usize {
    return ((@as(usize, layer) * n_sample + sample) * topk) + slot;
}

pub fn pickIndex(layer: u32, row_stride: u32, row: u32, slots: u32, k: u32) usize {
    return ((@as(usize, layer) * row_stride + row) * slots) + k;
}

pub fn cellIndex(layers: u32, nexp: u32, bucket: u32, layer: u32, expert: u32) usize {
    return ((@as(usize, bucket) * layers + layer) * nexp) + expert;
}

/// Router outputs for one layer. pick == 0 means "leave the caller's buffer".
pub const PickPair = struct { pick: u64 = 0, wts: u64 = 0 };

/// What a layer left for the norm tier.
pub const Capture = enum(u8) { y_staged, pf_z, skip };

/// One MoE layer, after its experts. which: 0 a decode round, 1 a prompt chunk.
pub const LayerNote = struct {
    li: u32,
    rows: u32,
    slots: u32,
    src: u64 = 0,
    capture: Capture,
    bank: u8,
    which: u8,
};

/// Down epilogue y. reap_y == 0 keeps the base launch: sc.y and discard when 4 * slots <= 32.
/// Nonzero points y at staging and clears discard. Prep's y is never this value.
pub fn downY(sc_y: u64, reap_y: u64, slots: usize) struct { y: u64, discard: i32 } {
    if (reap_y == 0) return .{ .y = sc_y, .discard = @intFromBool(4 * slots <= 32) };
    return .{ .y = reap_y, .discard = 0 };
}

// --- accumulators ------------------------------------------------------------

pub const Acc = struct {
    layers: u32,
    nexp: u32,
    nbucket: u32,
    token: []i64,
    gate: []f64,
    early: []i64,
    raw: []f64,
    ncount: []i64,
    maxn: []f64,
    /// Committed rows and norm-sampled rows, one each per bucket (host-side, not the expert kernel).
    tokens: []i64,
    norm_rows: []i64,

    pub fn init(a: Allocator, layers: u32, nexp: u32, nbucket: u32) !Acc {
        const n = @as(usize, layers) * nexp * nbucket;
        var acc: Acc = .{
            .layers = layers,
            .nexp = nexp,
            .nbucket = nbucket,
            .token = &.{},
            .gate = &.{},
            .early = &.{},
            .raw = &.{},
            .ncount = &.{},
            .maxn = &.{},
            .tokens = &.{},
            .norm_rows = &.{},
        };
        errdefer acc.deinit(a);
        acc.token = try a.alloc(i64, n);
        acc.gate = try a.alloc(f64, n);
        acc.early = try a.alloc(i64, n);
        acc.raw = try a.alloc(f64, n);
        acc.ncount = try a.alloc(i64, n);
        acc.maxn = try a.alloc(f64, n);
        acc.tokens = try a.alloc(i64, nbucket);
        acc.norm_rows = try a.alloc(i64, nbucket);
        acc.zero();
        return acc;
    }

    pub fn deinit(acc: *Acc, a: Allocator) void {
        if (acc.token.len > 0) a.free(acc.token);
        if (acc.gate.len > 0) a.free(acc.gate);
        if (acc.early.len > 0) a.free(acc.early);
        if (acc.raw.len > 0) a.free(acc.raw);
        if (acc.ncount.len > 0) a.free(acc.ncount);
        if (acc.maxn.len > 0) a.free(acc.maxn);
        if (acc.tokens.len > 0) a.free(acc.tokens);
        if (acc.norm_rows.len > 0) a.free(acc.norm_rows);
        acc.token = &.{};
        acc.gate = &.{};
        acc.early = &.{};
        acc.raw = &.{};
        acc.ncount = &.{};
        acc.maxn = &.{};
        acc.tokens = &.{};
        acc.norm_rows = &.{};
    }

    pub fn zero(acc: *Acc) void {
        @memset(acc.token, 0);
        @memset(acc.gate, 0);
        @memset(acc.early, 0);
        @memset(acc.raw, 0);
        @memset(acc.ncount, 0);
        @memset(acc.maxn, 0);
        @memset(acc.tokens, 0);
        @memset(acc.norm_rows, 0);
    }

    pub fn idx(acc: *const Acc, bucket: u32, layer: u32, expert: u32) usize {
        return cellIndex(acc.layers, acc.nexp, bucket, layer, expert);
    }

    pub fn copyFrom(acc: *Acc, src: *const Acc) void {
        @memcpy(acc.token, src.token);
        @memcpy(acc.gate, src.gate);
        @memcpy(acc.early, src.early);
        @memcpy(acc.raw, src.raw);
        @memcpy(acc.ncount, src.ncount);
        @memcpy(acc.maxn, src.maxn);
        @memcpy(acc.tokens, src.tokens);
        @memcpy(acc.norm_rows, src.norm_rows);
    }
};

const Cell = struct {
    raw: f64,
    ncount: i64,
    token: i64,
    gate: f64,
    early: i64,
    maxn: f64,
};

fn cellOf(acc: *const Acc, b0: u32, b1: ?u32, layer: u32, expert: u32) Cell {
    const at0 = acc.idx(b0, layer, expert);
    if (b1 == null) return .{
        .raw = acc.raw[at0],
        .ncount = acc.ncount[at0],
        .token = acc.token[at0],
        .gate = acc.gate[at0],
        .early = acc.early[at0],
        .maxn = acc.maxn[at0],
    };
    const at1 = acc.idx(b1.?, layer, expert);
    return .{
        .raw = acc.raw[at0] + acc.raw[at1],
        .ncount = acc.ncount[at0] + acc.ncount[at1],
        .token = acc.token[at0] + acc.token[at1],
        .gate = acc.gate[at0] + acc.gate[at1],
        .early = acc.early[at0] + acc.early[at1],
        .maxn = @max(acc.maxn[at0], acc.maxn[at1]),
    };
}

fn saliencyOf(c: Cell) f64 {
    const d: f64 = if (c.ncount > 0) @floatFromInt(c.ncount) else 1;
    return c.raw / d;
}

/// Zero when nothing was sampled, even if token_count is not.
fn weightedOf(c: Cell) f64 {
    if (c.ncount == 0) return 0;
    return saliencyOf(c) * @as(f64, @floatFromInt(c.token));
}

fn bucketsOf(modality: Modality, domain_index: u32) struct { b0: u32, b1: ?u32 } {
    const text_b = domain_index * 2;
    return switch (modality) {
        .text => .{ .b0 = text_b, .b1 = null },
        .vision => .{ .b0 = text_b + 1, .b1 = null },
        .all => .{ .b0 = text_b, .b1 = text_b + 1 },
    };
}

fn scalarSum(xs: []const i64, b0: u32, b1: ?u32) i64 {
    if (b1) |b| return xs[b0] + xs[b];
    return xs[b0];
}

const Writer = std.Io.Writer;

/// Python's json float text: repr, with NaN and Infinity spelled as JavaScript does.
/// Copied from server/json_write.zig so this file stays inside the dsv41 module.
fn floatJson(buf: []u8, f: f64) []const u8 {
    if (std.math.isNan(f)) return "NaN";
    if (std.math.isInf(f)) return if (f > 0) "Infinity" else "-Infinity";
    return floatRepr(buf, f);
}

/// ``repr(float)``: the shortest round-trip digits, fixed between 1e-4 and 1e16, else exponent form.
fn floatRepr(buf: []u8, f: f64) []const u8 {
    if (f == 0) return if (std.math.signbit(f)) "-0.0" else "0.0";
    var tmp: [64]u8 = undefined;
    const sci = std.fmt.float.render(&tmp, f, .{ .mode = .scientific }) catch unreachable;
    var digits: [32]u8 = undefined;
    var n: usize = 0;
    var i: usize = 0;
    const negative = sci[0] == '-';
    if (negative) i = 1;
    while (i < sci.len and sci[i] != 'e') : (i += 1) {
        if (sci[i] != '.') {
            digits[n] = sci[i];
            n += 1;
        }
    }
    while (n > 1 and digits[n - 1] == '0') n -= 1;
    const exp = std.fmt.parseInt(i32, sci[i + 1 ..], 10) catch 0;
    const decpt = exp + 1;
    var out: std.ArrayList(u8) = .initBuffer(buf);
    if (negative) out.appendAssumeCapacity('-');
    if (decpt <= -4 or decpt > 16) {
        out.appendAssumeCapacity(digits[0]);
        if (n > 1) {
            out.appendAssumeCapacity('.');
            out.appendSliceAssumeCapacity(digits[1..n]);
        }
        out.appendAssumeCapacity('e');
        out.appendAssumeCapacity(if (exp < 0) '-' else '+');
        const mag: u32 = @intCast(if (exp < 0) -exp else exp);
        if (mag < 10) out.appendAssumeCapacity('0');
        var ebuf: [8]u8 = undefined;
        out.appendSliceAssumeCapacity(std.fmt.bufPrint(&ebuf, "{d}", .{mag}) catch unreachable);
    } else if (decpt <= 0) {
        out.appendSliceAssumeCapacity("0.");
        var z: i32 = decpt;
        while (z < 0) : (z += 1) out.appendAssumeCapacity('0');
        out.appendSliceAssumeCapacity(digits[0..n]);
    } else {
        const d: usize = @intCast(decpt);
        if (d >= n) {
            out.appendSliceAssumeCapacity(digits[0..n]);
            for (0..d - n) |_| out.appendAssumeCapacity('0');
            out.appendSliceAssumeCapacity(".0");
        } else {
            out.appendSliceAssumeCapacity(digits[0..d]);
            out.appendAssumeCapacity('.');
            out.appendSliceAssumeCapacity(digits[d..n]);
        }
    }
    return out.items;
}

fn writeJsonString(w: *Writer, s: []const u8) Writer.Error!void {
    try w.writeByte('"');
    for (s) |c| switch (c) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        else => if (c < 0x20 or c >= 0x7f) try w.print("\\u{x:0>4}", .{c}) else try w.writeByte(c),
    };
    try w.writeByte('"');
}

fn writeFloat(w: *Writer, f: f64) Writer.Error!void {
    var buf: [64]u8 = undefined;
    try w.writeAll(floatJson(&buf, f));
}

fn writeF64Grid(w: *Writer, acc: *const Acc, modality: Modality, domain_index: u32, which: enum { sal, wsum, gate, maxn, raw }) Writer.Error!void {
    const bs = bucketsOf(modality, domain_index);
    try w.writeByte('[');
    var layer: u32 = 0;
    while (layer < acc.layers) : (layer += 1) {
        if (layer > 0) try w.writeByte(',');
        try w.writeByte('[');
        var e: u32 = 0;
        while (e < acc.nexp) : (e += 1) {
            if (e > 0) try w.writeByte(',');
            const c = cellOf(acc, bs.b0, bs.b1, layer, e);
            const v: f64 = switch (which) {
                .sal => saliencyOf(c),
                .wsum => weightedOf(c),
                .gate => c.gate,
                .maxn => c.maxn,
                .raw => c.raw,
            };
            try writeFloat(w, v);
        }
        try w.writeByte(']');
    }
    try w.writeByte(']');
}

fn writeI64Grid(w: *Writer, acc: *const Acc, modality: Modality, domain_index: u32, which: enum { token, early, ncount }) Writer.Error!void {
    const bs = bucketsOf(modality, domain_index);
    try w.writeByte('[');
    var layer: u32 = 0;
    while (layer < acc.layers) : (layer += 1) {
        if (layer > 0) try w.writeByte(',');
        try w.writeByte('[');
        var e: u32 = 0;
        while (e < acc.nexp) : (e += 1) {
            if (e > 0) try w.writeByte(',');
            const c = cellOf(acc, bs.b0, bs.b1, layer, e);
            const v: i64 = switch (which) {
                .token => c.token,
                .early => c.early,
                .ncount => c.ncount,
            };
            try w.print("{d}", .{v});
        }
        try w.writeByte(']');
    }
    try w.writeByte(']');
}

fn modalityName(m: Modality) []const u8 {
    return switch (m) {
        .text => "text",
        .vision => "vision",
        .all => "all",
    };
}

/// Export body. Key order is `report_keys`, then sampled's norm_count before its raw sum.
pub fn writeReport(w: *Writer, acc: *const Acc, domain: []const u8, modality: Modality, domain_index: u32, model_id: []const u8, build_id: []const u8) Writer.Error!void {
    const bs = bucketsOf(modality, domain_index);
    try w.writeAll("{\"model_id\":");
    try writeJsonString(w, model_id);
    try w.print(",\"num_layers\":{d},\"num_experts\":{d}", .{ acc.layers, acc.nexp });
    try w.writeAll(",\"reap_saliency\":");
    try writeF64Grid(w, acc, modality, domain_index, .sal);
    try w.writeAll(",\"reap_weighted_norm_sum\":");
    try writeF64Grid(w, acc, modality, domain_index, .wsum);
    try w.writeAll(",\"gate_mass_sum\":");
    try writeF64Grid(w, acc, modality, domain_index, .gate);
    try w.writeAll(",\"token_count\":");
    try writeI64Grid(w, acc, modality, domain_index, .token);
    try w.writeAll(",\"max_l2_norm\":");
    try writeF64Grid(w, acc, modality, domain_index, .maxn);
    try w.writeAll(",\"early_pos_count\":");
    try writeI64Grid(w, acc, modality, domain_index, .early);
    try w.writeAll(",\"domain\":");
    try writeJsonString(w, domain);
    try w.print(",\"modality\":\"{s}\",\"tokens\":{d},\"norm_rows\":{d}", .{
        modalityName(modality),
        scalarSum(acc.tokens, bs.b0, bs.b1),
        scalarSum(acc.norm_rows, bs.b0, bs.b1),
    });
    try w.writeAll(",\"sampled\":{\"norm_count\":");
    try writeI64Grid(w, acc, modality, domain_index, .ncount);
    try w.writeAll(",\"reap_weighted_norm_sum\":");
    try writeF64Grid(w, acc, modality, domain_index, .raw);
    try w.writeAll("},\"build\":");
    try writeJsonString(w, build_id);
    try w.writeByte('}');
}

pub fn writeList(w: *Writer, domains: *Domains, acc: *const Acc) Writer.Error!void {
    domains.lock();
    defer domains.unlock();
    try w.writeAll("{\"buckets\":[");
    var i: u32 = 0;
    var first = true;
    while (i < domains.n) : (i += 1) {
        for ([_]Modality{ .text, .vision }) |m| {
            if (!first) try w.writeByte(',');
            first = false;
            const b = bucketIndex(i, @intFromEnum(m));
            try w.writeAll("{\"domain\":");
            try writeJsonString(w, domains.name(i));
            try w.print(",\"modality\":\"{s}\",\"tokens\":{d},\"norm_rows\":{d}}}", .{ modalityName(m), acc.tokens[b], acc.norm_rows[b] });
        }
    }
    try w.writeAll("]}");
}

// --- CPU reference -----------------------------------------------------------

fn noteCommitted(acc: *Acc, commit: []const u8, bucket: []const u8, rows: u32) void {
    var row: u32 = 0;
    while (row < rows) : (row += 1) {
        if (commit[row] == 0) continue;
        const b = bucket[row];
        if (b >= acc.nbucket) continue;
        acc.tokens[b] += 1;
    }
}

fn noteSamples(acc: *Acc, bucket: []const u8, rows_idx: ?[]const i32, n_sample: u32) void {
    var s: u32 = 0;
    while (s < n_sample) : (s += 1) {
        const row: u32 = if (rows_idx) |idx| @intCast(idx[s]) else s;
        const b = bucket[row];
        if (b >= acc.nbucket) continue;
        acc.norm_rows[b] += 1;
    }
}

/// The route kernel: one pass per (layer, expert), rows in order, slots 0..topk. Also the host
/// row counter (the kernel does not write `tokens`).
pub fn engineRoute(
    acc: *Acc,
    pick: []const i32,
    wts: []const f32,
    pos: []const i32,
    commit: []const u8,
    bucket: []const u8,
    layers: u32,
    rows: u32,
    row_stride: u32,
    slots: u32,
) void {
    noteCommitted(acc, commit, bucket, rows);
    var layer: u32 = 0;
    while (layer < layers) : (layer += 1) {
        var expert: u32 = 0;
        while (expert < acc.nexp) : (expert += 1) {
            var row: u32 = 0;
            while (row < rows) : (row += 1) {
                if (commit[row] == 0) continue;
                const b = bucket[row];
                if (b >= acc.nbucket) continue;
                const p = pos[row];
                var k: u32 = 0;
                while (k < topk) : (k += 1) {
                    const id = pick[pickIndex(layer, row_stride, row, slots, k)];
                    if (id < 0 or @as(u32, @intCast(id)) >= acc.nexp or @as(u32, @intCast(id)) != expert) continue;
                    const at = acc.idx(b, layer, expert);
                    acc.token[at] += 1;
                    acc.gate[at] += wts[pickIndex(layer, row_stride, row, slots, k)];
                    if (p < early_pos) acc.early[at] += 1;
                }
            }
        }
    }
}

/// The norm kernel. ss1 == null means the hop-1 sum of squares is already the full vector (world 1 and 2).
pub fn engineNorm(
    acc: *Acc,
    pick: []const i32,
    wts: []const f32,
    bucket: []const u8,
    rows_idx: ?[]const i32,
    ss0: []const f64,
    ss1: ?[]const f64,
    layers: u32,
    row_stride: u32,
    slots: u32,
    n_sample: u32,
) void {
    noteSamples(acc, bucket, rows_idx, n_sample);
    var layer: u32 = 0;
    while (layer < layers) : (layer += 1) {
        var expert: u32 = 0;
        while (expert < acc.nexp) : (expert += 1) {
            var sample: u32 = 0;
            while (sample < n_sample) : (sample += 1) {
                const row: u32 = if (rows_idx) |idx| @intCast(idx[sample]) else sample;
                const b = bucket[row];
                if (b >= acc.nbucket) continue;
                var k: u32 = 0;
                while (k < topk) : (k += 1) {
                    const id = pick[pickIndex(layer, row_stride, row, slots, k)];
                    if (id < 0 or @as(u32, @intCast(id)) >= acc.nexp or @as(u32, @intCast(id)) != expert) continue;
                    const vi = ssIndex(layer, n_sample, sample, k);
                    var ss = ss0[vi];
                    if (ss1) |s1| ss += s1[vi];
                    const norm = @sqrt(ss);
                    const at = acc.idx(b, layer, expert);
                    acc.raw[at] += @as(f64, wts[pickIndex(layer, row_stride, row, slots, k)]) * norm;
                    acc.ncount[at] += 1;
                    if (norm > acc.maxn[at]) acc.maxn[at] = norm;
                }
            }
        }
    }
}

/// Numpy `record_batch` over routed (row, slot) pairs. The norm is ||sum of K-partials||, not the
/// sum of per-rank norms. `partials[rank]` is [layers, rows, slots, dn] in packSrc order (rows_cap = rows).
/// World 4: pair p's vector is rank 2p plus rank 2p+1, and ||f||^2 is the sum of the two pairs' squares.
pub fn scriptRecord(
    acc: *Acc,
    pick: []const i32,
    wts: []const f32,
    pos: []const i32,
    commit: []const u8,
    bucket: []const u8,
    partials: []const []const f32,
    world: u32,
    layers: u32,
    rows: u32,
    slots: u32,
    dn: u32,
) void {
    var row: u32 = 0;
    while (row < rows) : (row += 1) {
        if (commit[row] == 0) continue;
        const b = bucket[row];
        if (b >= acc.nbucket) continue;
        acc.tokens[b] += 1;
        acc.norm_rows[b] += 1;
        var layer: u32 = 0;
        while (layer < layers) : (layer += 1) {
            var k: u32 = 0;
            while (k < topk) : (k += 1) {
                const id = pick[pickIndex(layer, rows, row, slots, k)];
                if (id < 0 or @as(u32, @intCast(id)) >= acc.nexp) continue;
                const expert: u32 = @intCast(id);
                const g: f64 = wts[pickIndex(layer, rows, row, slots, k)];
                const norm = scriptNorm(partials, world, layer, rows, row, slots, k, dn);
                const at = acc.idx(b, layer, expert);
                acc.raw[at] += g * norm;
                acc.ncount[at] += 1;
                acc.gate[at] += g;
                acc.token[at] += 1;
                if (norm > acc.maxn[at]) acc.maxn[at] = norm;
                if (pos[row] < early_pos) acc.early[at] += 1;
            }
        }
    }
}

fn scriptNorm(partials: []const []const f32, world: u32, layer: u32, rows: u32, row: u32, slots: u32, slot: u32, dn: u32) f64 {
    const base = packSrc(layer, rows, row, slots, slot, dn, 0);
    if (world <= 1) {
        var s: f64 = 0;
        var c: u32 = 0;
        while (c < dn) : (c += 1) {
            const x: f64 = partials[0][base + c];
            s += x * x;
        }
        return @sqrt(s);
    }
    if (world == 2) {
        var s: f64 = 0;
        var c: u32 = 0;
        while (c < dn) : (c += 1) {
            const x = @as(f64, partials[0][base + c]) + @as(f64, partials[1][base + c]);
            s += x * x;
        }
        return @sqrt(s);
    }
    var s0: f64 = 0;
    var s1: f64 = 0;
    var c: u32 = 0;
    while (c < dn) : (c += 1) {
        const p0 = @as(f64, partials[0][base + c]) + @as(f64, partials[1][base + c]);
        const p1 = @as(f64, partials[2][base + c]) + @as(f64, partials[3][base + c]);
        s0 += p0 * p0;
        s1 += p1 * p1;
    }
    return @sqrt(s0 + s1);
}

/// Sum of squares of (y0 + y1), y1 null meaning y0 alone. The ssq kernel.
pub fn ssqPair(y0: []const f32, y1: ?[]const f32) f64 {
    var s: f64 = 0;
    for (y0, 0..) |a0, i| {
        var x: f64 = a0;
        if (y1) |b| x += b[i];
        s += x * x;
    }
    return s;
}

// --- snapshots ---------------------------------------------------------------

fn fsyncFd(fd: std.posix.fd_t) !void {
    switch (std.posix.errno(std.posix.system.fsync(fd))) {
        .SUCCESS => {},
        else => return error.SyncFailed,
    }
}

fn writeAllFd(fd: std.posix.fd_t, body: []const u8) !void {
    var off: usize = 0;
    while (off < body.len) {
        const n = std.posix.system.write(fd, body[off..].ptr, body.len - off);
        switch (std.posix.errno(n)) {
            .SUCCESS => {
                if (n <= 0) return error.WriteFailed;
                off += @intCast(n);
            },
            .INTR => continue,
            else => return error.WriteFailed,
        }
    }
}

/// tmp in the same directory, fsync the file, rename, fsync the directory.
pub fn writeAtomic(a: Allocator, io: std.Io, dir_path: []const u8, name: []const u8, body: []const u8) !void {
    var tmp_buf: [80]u8 = undefined;
    const tmp_name = try std.fmt.bufPrint(&tmp_buf, "{s}.tmp", .{name});
    const tmp_path = try std.fs.path.join(a, &.{ dir_path, tmp_name });
    defer a.free(tmp_path);
    const tmpz = try a.dupeSentinel(u8, tmp_path, 0);
    defer a.free(tmpz);
    const fd = try std.posix.openatZ(std.posix.AT.FDCWD, tmpz, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
    errdefer _ = std.posix.system.close(fd);
    try writeAllFd(fd, body);
    try fsyncFd(fd);
    _ = std.posix.system.close(fd);
    var d = try std.Io.Dir.cwd().openDir(io, dir_path, .{});
    defer d.close(io);
    try d.rename(tmp_name, d, name, io);
    const dirz = try a.dupeSentinel(u8, dir_path, 0);
    defer a.free(dirz);
    const dfd = try std.posix.openatZ(std.posix.AT.FDCWD, dirz, .{ .ACCMODE = .RDONLY, .DIRECTORY = true }, 0);
    defer _ = std.posix.system.close(dfd);
    try fsyncFd(dfd);
}

fn fileExists(io: std.Io, dir: std.Io.Dir, name: []const u8) bool {
    var f = dir.openFile(io, name, .{}) catch return false;
    f.close(io);
    return true;
}

fn moveAside(io: std.Io, dir: std.Io.Dir, name: []const u8) void {
    var buf: [96]u8 = undefined;
    const bad = std.fmt.bufPrint(&buf, "{s}.bad", .{name}) catch return;
    if (!fileExists(io, dir, bad)) {
        dir.rename(name, dir, bad, io) catch |e| std.log.warn("reap: rename {s} aside failed: {s}", .{ name, @errorName(e) });
        return;
    }
    var n: u32 = 2;
    while (n < 10) : (n += 1) {
        var buf2: [96]u8 = undefined;
        const alt = std.fmt.bufPrint(&buf2, "{s}.bad.{d}", .{ name, n }) catch return;
        if (fileExists(io, dir, alt)) continue;
        dir.rename(name, dir, alt, io) catch |e| std.log.warn("reap: rename {s} aside failed: {s}", .{ name, @errorName(e) });
        return;
    }
    std.log.warn("reap: {s} left in place, .bad names are taken", .{name});
}

const SnapName = struct { domain: []const u8, modality: Modality };

fn splitSnap(name: []const u8) ?SnapName {
    const text_suf = ".text.json";
    const vis_suf = ".vision.json";
    if (std.mem.endsWith(u8, name, text_suf)) {
        const d = name[0 .. name.len - text_suf.len];
        if (!domainOk(d)) return null;
        return .{ .domain = d, .modality = .text };
    }
    if (std.mem.endsWith(u8, name, vis_suf)) {
        const d = name[0 .. name.len - vis_suf.len];
        if (!domainOk(d)) return null;
        return .{ .domain = d, .modality = .vision };
    }
    return null;
}

fn field(v: std.json.Value, key: []const u8) ?std.json.Value {
    if (v != .object) return null;
    return v.object.get(key);
}

fn strField(v: std.json.Value, key: []const u8) ?[]const u8 {
    const f = field(v, key) orelse return null;
    return if (f == .string) f.string else null;
}

fn jsonF64(v: std.json.Value) ?f64 {
    return switch (v) {
        .integer => |n| @floatFromInt(n),
        .float => |f| if (std.math.isFinite(f)) f else null,
        .number_string => |s| std.fmt.parseFloat(f64, s) catch null,
        else => null,
    };
}

fn asI64(v: std.json.Value) ?i64 {
    switch (v) {
        .integer => |n| return n,
        .float => |f| return integralI64(f),
        .number_string => |s| {
            if (std.fmt.parseInt(i64, s, 10)) |n| return n else |_| {}
            return integralI64(std.fmt.parseFloat(f64, s) catch return null);
        },
        else => return null,
    }
}

fn integralI64(f: f64) ?i64 {
    if (!std.math.isFinite(f) or f != @round(f)) return null;
    if (f > @as(f64, @floatFromInt(std.math.maxInt(i64))) or f < @as(f64, @floatFromInt(std.math.minInt(i64)))) return null;
    return @intFromFloat(f);
}

fn gridItems(root: std.json.Value, key: []const u8) ![]std.json.Value {
    const g = field(root, key) orelse return error.BadSnapshot;
    if (g != .array) return error.BadSnapshot;
    return g.array.items;
}

fn checkGrid(root: std.json.Value, key: []const u8, layers: u32, nexp: u32, ints: bool) !void {
    const grid = try gridItems(root, key);
    if (grid.len != layers) return error.BadSnapshot;
    for (grid) |rowv| {
        if (rowv != .array) return error.BadSnapshot;
        const row = rowv.array.items;
        if (row.len != nexp) return error.BadSnapshot;
        for (row) |cell| {
            if (ints) {
                _ = asI64(cell) orelse return error.BadSnapshot;
            } else if (jsonF64(cell) == null) return error.BadSnapshot;
        }
    }
}

fn loadMatrix(acc: *Acc, root: std.json.Value, key: []const u8, bucket: u32, kind: enum { i64_token, i64_early, i64_ncount, f64_gate, f64_raw, f64_max }) !void {
    const grid = try gridItems(root, key);
    if (grid.len != acc.layers) return error.BadSnapshot;
    var layer: u32 = 0;
    while (layer < acc.layers) : (layer += 1) {
        if (grid[layer] != .array) return error.BadSnapshot;
        const row = grid[layer].array.items;
        if (row.len != acc.nexp) return error.BadSnapshot;
        var e: u32 = 0;
        while (e < acc.nexp) : (e += 1) {
            const at = acc.idx(bucket, layer, e);
            switch (kind) {
                .i64_token => acc.token[at] = asI64(row[e]) orelse return error.BadSnapshot,
                .i64_early => acc.early[at] = asI64(row[e]) orelse return error.BadSnapshot,
                .i64_ncount => acc.ncount[at] = asI64(row[e]) orelse return error.BadSnapshot,
                .f64_gate => acc.gate[at] = jsonF64(row[e]) orelse return error.BadSnapshot,
                .f64_raw => acc.raw[at] = jsonF64(row[e]) orelse return error.BadSnapshot,
                .f64_max => acc.maxn[at] = jsonF64(row[e]) orelse return error.BadSnapshot,
            }
        }
    }
}

fn loadSnapshotText(acc: *Acc, domains: *Domains, text: []const u8, model_id: []const u8, domain: []const u8, modality: Modality, a: Allocator) !void {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const root = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), text, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.BadSnapshot,
    };
    const got_id = strField(root, "model_id") orelse return error.BadSnapshot;
    if (!std.mem.eql(u8, got_id, model_id)) return error.BadSnapshot;
    const layers = asI64(field(root, "num_layers") orelse return error.BadSnapshot) orelse return error.BadSnapshot;
    const experts = asI64(field(root, "num_experts") orelse return error.BadSnapshot) orelse return error.BadSnapshot;
    if (layers != acc.layers or experts != acc.nexp) return error.BadSnapshot;
    const got_domain = strField(root, "domain") orelse return error.BadSnapshot;
    if (!std.mem.eql(u8, got_domain, domain)) return error.BadSnapshot;
    const got_mod = strField(root, "modality") orelse return error.BadSnapshot;
    if (!std.mem.eql(u8, got_mod, modalityName(modality))) return error.BadSnapshot;
    const sampled = field(root, "sampled") orelse return error.BadSnapshot;
    if (sampled != .object) return error.BadSnapshot;
    _ = asI64(field(root, "tokens") orelse return error.BadSnapshot) orelse return error.BadSnapshot;
    _ = asI64(field(root, "norm_rows") orelse return error.BadSnapshot) orelse return error.BadSnapshot;
    try checkGrid(root, "token_count", acc.layers, acc.nexp, true);
    try checkGrid(root, "early_pos_count", acc.layers, acc.nexp, true);
    try checkGrid(root, "gate_mass_sum", acc.layers, acc.nexp, false);
    try checkGrid(root, "max_l2_norm", acc.layers, acc.nexp, false);
    try checkGrid(sampled, "norm_count", acc.layers, acc.nexp, true);
    try checkGrid(sampled, "reap_weighted_norm_sum", acc.layers, acc.nexp, false);
    const di = domains.admit(domain) catch return error.BadSnapshot;
    const b = bucketIndex(di, @intFromEnum(modality));
    if (b >= acc.nbucket) return error.BadSnapshot;
    try loadMatrix(acc, root, "token_count", b, .i64_token);
    try loadMatrix(acc, root, "gate_mass_sum", b, .f64_gate);
    try loadMatrix(acc, root, "early_pos_count", b, .i64_early);
    try loadMatrix(acc, root, "max_l2_norm", b, .f64_max);
    try loadMatrix(acc, sampled, "norm_count", b, .i64_ncount);
    try loadMatrix(acc, sampled, "reap_weighted_norm_sum", b, .f64_raw);
    acc.tokens[b] = asI64(field(root, "tokens") orelse return error.BadSnapshot) orelse return error.BadSnapshot;
    acc.norm_rows[b] = asI64(field(root, "norm_rows") orelse return error.BadSnapshot) orelse return error.BadSnapshot;
}

fn readWhole(a: Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8) ![]u8 {
    var f = try dir.openFile(io, name, .{});
    defer f.close(io);
    const st = try f.stat(io);
    if (st.size > 64 * 1024 * 1024) return error.BadSnapshot;
    const buf = try a.alloc(u8, @intCast(st.size));
    errdefer a.free(buf);
    const n = try f.readPositionalAll(io, buf, 0);
    if (n != buf.len) return error.BadSnapshot;
    return buf;
}

/// Load every `<domain>.text.json` / `<domain>.vision.json`, alphabetically. A mismatch, a parse
/// failure, a missing `sampled` object, or a domain past the cap is renamed aside to `.bad` and left
/// out of the accumulators. Returns how many files were loaded.
pub fn reloadDir(a: Allocator, io: std.Io, dir_path: []const u8, acc: *Acc, domains: *Domains, model_id: []const u8) !u32 {
    var d = std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |nm| a.free(nm);
        names.deinit(a);
    }
    var it = d.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .file) continue;
        if (splitSnap(e.name) == null) continue;
        try names.append(a, try a.dupe(u8, e.name));
    }
    std.mem.sort([]u8, names.items, {}, struct {
        fn lt(_: void, x: []u8, y: []u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    var loaded: u32 = 0;
    for (names.items) |nm| {
        const snap = splitSnap(nm).?;
        const text = readWhole(a, io, d, nm) catch |err| {
            std.log.warn("reap: {s} not loaded ({s}), renaming aside", .{ nm, @errorName(err) });
            moveAside(io, d, nm);
            continue;
        };
        defer a.free(text);
        loadSnapshotText(acc, domains, text, model_id, snap.domain, snap.modality, a) catch |err| {
            std.log.warn("reap: {s} not loaded ({s}), renaming aside", .{ nm, @errorName(err) });
            moveAside(io, d, nm);
            continue;
        };
        loaded += 1;
    }
    return loaded;
}

pub fn snapshotAll(a: Allocator, io: std.Io, dir_path: []const u8, acc: *const Acc, domains: *Domains, model_id: []const u8, build_id: []const u8) !void {
    try std.Io.Dir.cwd().createDirPath(io, dir_path);
    var names: [max_domains][max_name]u8 = undefined;
    var lens: [max_domains]u8 = undefined;
    domains.lock();
    const n = domains.n;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        names[i] = domains.names[i];
        lens[i] = domains.lens[i];
    }
    domains.unlock();
    i = 0;
    while (i < n) : (i += 1) {
        const domain = names[i][0..lens[i]];
        for ([_]Modality{ .text, .vision }) |m| {
            var body: std.Io.Writer.Allocating = .init(a);
            defer body.deinit();
            try writeReport(&body.writer, acc, domain, m, i, model_id, build_id);
            var nb: [48]u8 = undefined;
            const fname = try std.fmt.bufPrint(&nb, "{s}.{s}.json", .{ domain, modalityName(m) });
            try writeAtomic(a, io, dir_path, fname, body.written());
        }
    }
}

// --- host mirror (the engine thread publishes; HTTP waits) -------------------

/// How long a stats query waits for the engine thread to refresh the host mirror.
pub const refresh_wait_ns: u64 = 5 * std.time.ns_per_s;

pub const Refresh = struct {
    pending: std.atomic.Value(u32) = .init(0),
    seq: std.atomic.Value(u64) = .init(0),

    /// Note the current sequence, then ask for a refresh. Wait until `seq` moves past the return.
    pub fn ask(r: *Refresh) u64 {
        const seen = r.seq.load(.acquire);
        r.pending.store(1, .release);
        return seen;
    }

    pub fn ready(r: *const Refresh, seen: u64) bool {
        return r.seq.load(.acquire) != seen;
    }

    pub fn wanted(r: *const Refresh) bool {
        return r.pending.load(.acquire) != 0;
    }

    /// The host mirror is current. A request that lands after the clear waits for the next bump.
    pub fn published(r: *Refresh) void {
        r.pending.store(0, .release);
        _ = r.seq.fetchAdd(1, .release);
    }
};

pub fn waitRefresh(r: *Refresh, seen: u64, timeout_ns: u64, io: std.Io) bool {
    const start = monoNs();
    while (!r.ready(seen)) {
        if (monoNs() -% start >= timeout_ns) return false;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return true;
}

pub const MirrorWait = enum { ready, stalled };

/// Ask the engine thread for a fresh host mirror and wait. `.stalled` is the HTTP 503.
pub fn waitForMirror(r: *Refresh, timeout_ns: u64, io: std.Io) MirrorWait {
    const seen = r.ask();
    if (!waitRefresh(r, seen, timeout_ns, io)) return .stalled;
    return .ready;
}

// --- live mirror (the HTTP route; tests install one with pull = null) -------

pub const PullFn = *const fn (ctx: *anyopaque, dst: *Acc) anyerror!void;

pub const Live = struct {
    cfg: Config,
    domains: *Domains,
    acc: *Acc,
    model_id: []const u8,
    build: []const u8,
    pull: ?PullFn = null,
    pull_ctx: ?*anyopaque = null,
};

var installed: ?*Live = null;
var live_mu: std.atomic.Mutex = .unlocked;

fn lockLive() void {
    while (!live_mu.tryLock()) std.Thread.yield() catch {};
}

pub fn install(l: *Live) void {
    lockLive();
    defer live_mu.unlock();
    installed = l;
}

pub fn uninstall(l: *Live) void {
    lockLive();
    defer live_mu.unlock();
    if (installed == l) installed = null;
}

pub fn current() ?*Live {
    lockLive();
    defer live_mu.unlock();
    return installed;
}

/// Hold across an export so close's uninstall waits. close must not take this while holding the
/// device lock (uninstall runs after that lock is dropped).
pub fn lockInstalled() ?*Live {
    lockLive();
    if (installed == null) {
        live_mu.unlock();
        return null;
    }
    return installed;
}

pub fn unlockInstalled() void {
    live_mu.unlock();
}

// --- tests -------------------------------------------------------------------

fn bitEq(a: f64, b: f64) !void {
    try std.testing.expectEqual(@as(u64, @bitCast(a)), @as(u64, @bitCast(b)));
}

fn xorshift(s: *u64) u32 {
    var x = s.*;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    s.* = x;
    return @truncate(x);
}

test "config ignores the other keys unless stats is exactly 1" {
    const a = std.testing.allocator;
    var off = try configFrom(a, .{ .stats = "0", .every = "1", .dir = "/tmp/reap" });
    defer off.deinit(a);
    try std.testing.expect(!off.stats);
    try std.testing.expectEqual(@as(u32, 8), off.every);
    try std.testing.expectEqualStrings("", off.dir);
    var on = try configFrom(a, .{ .stats = "1", .every = "0", .buckets = "99", .prefill_rows = "4", .deploy = "abc", .served_name = "flash" });
    defer on.deinit(a);
    try std.testing.expect(on.stats);
    try std.testing.expectEqual(@as(u32, 8), on.every);
    try std.testing.expectEqual(max_domains, on.buckets);
    try std.testing.expectEqualStrings("abc", on.build);
    try std.testing.expectEqualStrings("flash", on.served_name);
    var buf: [32]u8 = undefined;
    try std.testing.expectEqualStrings("", planSuffix(&buf, off));
    try std.testing.expectEqualStrings(" reap=8/4/64", planSuffix(&buf, on));
}

test "domains: default, charset, cap" {
    var d = Domains.init(2);
    try std.testing.expectEqual(@as(u32, 0), try d.admit(""));
    try std.testing.expectEqual(@as(u32, 0), try d.admit("default"));
    try std.testing.expectEqual(@as(u32, 1), try d.admit("vision.a"));
    try std.testing.expectError(error.Cap, d.admit("other"));
    try std.testing.expectError(error.Invalid, d.admit("Bad"));
    try std.testing.expectError(error.Invalid, d.admit("a/b"));
    var long: [33]u8 = undefined;
    @memset(&long, 'a');
    try std.testing.expectError(error.Invalid, d.admit(&long));
    var msg: [80]u8 = undefined;
    try std.testing.expectEqualStrings("reap_domain cap 16: refusing new domain \"foo\"", formatCap(&msg, 16, "foo"));
}

test "schedule is identical on every rank, and the byte counts" {
    try std.testing.expectEqual(@as(u64, 2_457_600), pairBytes(40, 1, dn_tp4));
    try std.testing.expectEqual(@as(u64, 4_915_200), pairBytes(40, 1, dn_tp2));
    try std.testing.expectEqual(@as(u64, 1920), scalarBytes(40, 1));
    try std.testing.expectEqual(@as(u64, 7680), scalarBytes(40, 4));
    const w1 = exchange(1, 0, 40, 1, dn_tp2);
    try std.testing.expectEqual(@as(u8, 0), w1.n);
    try std.testing.expect(w1.ssq and w1.norm);
    var rank: u32 = 0;
    while (rank < 4) : (rank += 1) {
        const h = exchange(4, rank, 40, 1, dn_tp4);
        try std.testing.expectEqual(@as(u8, 2), h.n);
        try std.testing.expectEqual(HopKind.pair, h.hops[0].kind);
        try std.testing.expectEqual(@as(u32, 1), h.hops[0].peer_xor);
        try std.testing.expectEqual(@as(u64, 2_457_600), h.hops[0].bytes);
        try std.testing.expectEqual(HopKind.scalar, h.hops[1].kind);
        try std.testing.expectEqual(@as(u32, 2), h.hops[1].peer_xor);
        try std.testing.expectEqual(@as(u64, 1920), h.hops[1].bytes);
        try std.testing.expectEqual(rank % 2 == 0, h.ssq);
        try std.testing.expectEqual(rank == 0, h.norm);
        const t2 = exchange(2, rank % 2, 40, 1, dn_tp2);
        try std.testing.expectEqual(@as(u8, 1), t2.n);
        try std.testing.expectEqual(@as(u64, 4_915_200), t2.hops[0].bytes);
        try std.testing.expectEqual(@as(u32, 1), t2.hops[0].peer_xor);
    }
    try std.testing.expect(decodeSample(0, 8));
    try std.testing.expect(!decodeSample(1, 8));
    try std.testing.expect(decodeSample(8, 8));
    var idx: [8]u32 = undefined;
    try std.testing.expectEqual(@as(u32, 4), promptSample(2048, 4, &idx));
    try std.testing.expectEqualSlices(u32, &.{ 0, 512, 1024, 1536 }, idx[0..4]);
    try std.testing.expectEqual(@as(u32, 3), promptSample(3, 4, &idx));
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, idx[0..3]);
    try std.testing.expectEqual(@as(u32, 0), promptSample(0, 4, &idx));
    // One decode y bank of round.max_rows, two pack regions, six rank-0 grids. Served slots are 7 at both widths.
    try std.testing.expectEqual(@as(u64, 23_592_960), pullBytes(16));
    try std.testing.expectEqual(@as(u64, 137_625_600), @as(u64, moe_layers) * rows_cap_y * 7 * dn_tp4 * 4);
    try std.testing.expectEqual(@as(u64, 275_251_200), @as(u64, moe_layers) * rows_cap_y * 7 * dn_tp2 * 4);
    try std.testing.expectEqual(@as(u64, 200_056_224), deviceBytes(.{ .route_slots = 7, .y_slots = 7, .dn = dn_tp4 }));
    try std.testing.expectEqual(@as(u64, 176_463_264), deviceBytes(.{ .route_slots = 7, .y_slots = 7, .dn = dn_tp4, .rank0 = false }));
    try std.testing.expectEqual(@as(u64, 367_173_024), deviceBytes(.{ .route_slots = 7, .y_slots = 7, .dn = dn_tp2 }));
    try std.testing.expectEqual(@as(u64, 343_580_064), deviceBytes(.{ .route_slots = 7, .y_slots = 7, .dn = dn_tp2, .rank0 = false }));
}

test "prompt modality follows the image span; decode buckets are text" {
    const spans = [_]ImageSpan{ .{ .at = 10, .len = 4 }, .{ .at = 100, .len = 2 } };
    try std.testing.expectEqual(@as(u8, 0), modalityAt(9, &spans));
    try std.testing.expectEqual(@as(u8, 1), modalityAt(10, &spans));
    try std.testing.expectEqual(@as(u8, 1), modalityAt(13, &spans));
    try std.testing.expectEqual(@as(u8, 0), modalityAt(14, &spans));
    var map: [6]u8 = undefined;
    fillPromptBuckets(&map, 8, 3, &spans);
    try std.testing.expectEqualSlices(u8, &.{ 6, 6, 7, 7, 7, 7 }, &map);
    fillDecodeBuckets(map[0..4], 3);
    try std.testing.expectEqualSlices(u8, &.{ 6, 6, 6, 6 }, map[0..4]);
}

fn rankNormSum(y0: []const f32, y1: []const f32) f64 {
    var a: f64 = 0;
    var b: f64 = 0;
    for (y0) |v| a += @as(f64, v) * v;
    for (y1) |v| b += @as(f64, v) * v;
    return @sqrt(a) + @sqrt(b);
}

test "full norm is the norm of the summed partials" {
    const y0 = [_]f32{ 3, 0 };
    const y1 = [_]f32{ -3, 0 };
    try bitEq(0, ssqPair(&y0, &y1));
    try std.testing.expectEqual(@as(f64, 6), rankNormSum(&y0, &y1));
    const a = std.testing.allocator;
    var eng = try Acc.init(a, 1, 4, 2);
    defer eng.deinit(a);
    var script = try Acc.init(a, 1, 4, 2);
    defer script.deinit(a);
    var pick = [_]i32{ 1, -1, 384, -1, -1, -1, 384 };
    var wts = [_]f32{ 1.5, 0, 0, 0, 0, 0, 0 };
    var pos = [_]i32{3};
    var commit = [_]u8{1};
    var bucket = [_]u8{0};
    var p0 = [_]f32{ 3, 0 };
    var p1 = [_]f32{ -3, 0 };
    const partials = [_][]const f32{ &p0, &p1 };
    const ss0 = [_]f64{ssqPair(p0[0..2], p1[0..2])};
    // the other five slots are unread; give them zeros so the slice is long enough
    var ss = [_]f64{ ss0[0], 0, 0, 0, 0, 0 };
    engineRoute(&eng, &pick, &wts, &pos, &commit, &bucket, 1, 1, 1, 7);
    engineNorm(&eng, &pick, &wts, &bucket, null, &ss, null, 1, 1, 7, 1);
    scriptRecord(&script, &pick, &wts, &pos, &commit, &bucket, &partials, 2, 1, 1, 7, 2);
    try bitEq(0, eng.raw[eng.idx(0, 0, 1)]);
    try bitEq(0, script.raw[script.idx(0, 0, 1)]);
    try std.testing.expectEqual(@as(i64, 1), eng.ncount[eng.idx(0, 0, 1)]);
    try bitEq(eng.maxn[eng.idx(0, 0, 1)], script.maxn[script.idx(0, 0, 1)]);
    // a rank-norm sum would have stored 1.5 * 6
    try std.testing.expect(eng.raw[eng.idx(0, 0, 1)] != 1.5 * rankNormSum(&y0, &y1));
}

fn fillPartial(buf: []f32, seed: *u64) void {
    for (buf) |*v| v.* = @floatFromInt(@as(i32, @intCast(xorshift(seed) % 21)) - 10);
}

test "cpu reference matches an independent record_batch over random routing" {
    const a = std.testing.allocator;
    const layers: u32 = 2;
    const nexp: u32 = 16;
    const rows: u32 = 8;
    const slots: u32 = 7;
    const dn: u32 = 4;
    const worlds = [_]u32{ 1, 2, 4 };
    for (worlds) |world| {
        var seed: u64 = 0x1234_5678_9abc ^ @as(u64, world);
        var pick: [layers * rows * slots]i32 = undefined;
        var wts: [layers * rows * slots]f32 = undefined;
        var pos: [rows]i32 = undefined;
        var commit: [rows]u8 = undefined;
        var bucket: [rows]u8 = undefined;
        var r: u32 = 0;
        while (r < rows) : (r += 1) {
            pos[r] = if (r < 3) @intCast(r) else 100;
            commit[r] = 1;
            bucket[r] = if (r % 2 == 0) 0 else 1;
            var layer: u32 = 0;
            while (layer < layers) : (layer += 1) {
                var k: u32 = 0;
                while (k < slots) : (k += 1) {
                    const at = pickIndex(layer, rows, r, slots, k);
                    if (k >= topk) {
                        pick[at] = 384;
                        wts[at] = 0;
                    } else if (k == 5) {
                        pick[at] = -1;
                        wts[at] = 4;
                    } else {
                        pick[at] = @intCast((r * 3 + layer + k) % nexp);
                        wts[at] = @floatFromInt(@as(i32, @intCast(xorshift(&seed) % 7)) + 1);
                    }
                }
            }
        }
        const nr: usize = if (world >= 4) 4 else if (world == 2) 2 else 1;
        var storage: [4][layers * rows * slots * dn]f32 = undefined;
        var partials: [4][]const f32 = undefined;
        var ri: usize = 0;
        while (ri < nr) : (ri += 1) {
            fillPartial(&storage[ri], &seed);
            partials[ri] = &storage[ri];
        }
        const nvec = layers * rows * topk;
        var ss0: [nvec]f64 = undefined;
        var ss1: [nvec]f64 = undefined;
        var layer: u32 = 0;
        while (layer < layers) : (layer += 1) {
            var row: u32 = 0;
            while (row < rows) : (row += 1) {
                var k: u32 = 0;
                while (k < topk) : (k += 1) {
                    const base = packSrc(layer, rows, row, slots, k, dn, 0);
                    const vi = ssIndex(layer, rows, row, k);
                    if (world <= 1) {
                        ss0[vi] = ssqPair(storage[0][base..][0..dn], null);
                    } else if (world == 2) {
                        ss0[vi] = ssqPair(storage[0][base..][0..dn], storage[1][base..][0..dn]);
                    } else {
                        ss0[vi] = ssqPair(storage[0][base..][0..dn], storage[1][base..][0..dn]);
                        ss1[vi] = ssqPair(storage[2][base..][0..dn], storage[3][base..][0..dn]);
                    }
                }
            }
        }
        var eng = try Acc.init(a, layers, nexp, 2);
        defer eng.deinit(a);
        var script = try Acc.init(a, layers, nexp, 2);
        defer script.deinit(a);
        engineRoute(&eng, &pick, &wts, &pos, &commit, &bucket, layers, rows, rows, slots);
        const second: ?[]const f64 = if (world >= 4) &ss1 else null;
        engineNorm(&eng, &pick, &wts, &bucket, null, &ss0, second, layers, rows, slots, rows);
        scriptRecord(&script, &pick, &wts, &pos, &commit, &bucket, partials[0..nr], world, layers, rows, slots, dn);
        try std.testing.expectEqualSlices(i64, script.token, eng.token);
        try std.testing.expectEqualSlices(i64, script.early, eng.early);
        try std.testing.expectEqualSlices(i64, script.ncount, eng.ncount);
        try std.testing.expectEqualSlices(i64, script.tokens, eng.tokens);
        try std.testing.expectEqualSlices(i64, script.norm_rows, eng.norm_rows);
        for (script.gate, eng.gate) |g0, g1| try bitEq(g0, g1);
        for (script.raw, eng.raw) |g0, g1| try bitEq(g0, g1);
        for (script.maxn, eng.maxn) |g0, g1| try bitEq(g0, g1);
        try std.testing.expectEqual(@as(i64, 8), eng.tokens[0] + eng.tokens[1]);
        // a dropped verify row never reaches the counters
        var dropped = [_]u8{0};
        var one = [_]u8{0};
        const before = eng.token[eng.idx(0, 0, 0)];
        engineRoute(&eng, &pick, &wts, &pos, &dropped, &one, 1, 1, rows, slots);
        try std.testing.expectEqual(before, eng.token[eng.idx(0, 0, 0)]);
    }
}

test "export key order, rescaling, merge, and reload of the raw sample" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var acc = try Acc.init(a, moe_layers, n_experts, 4);
    defer acc.deinit(a);
    const at = acc.idx(0, 3, 7);
    acc.token[at] = 4;
    acc.gate[at] = 1.25;
    acc.early[at] = 2;
    acc.raw[at] = 2.5;
    acc.ncount[at] = 1;
    acc.maxn[at] = 2.5;
    acc.tokens[0] = 9;
    acc.norm_rows[0] = 3;
    acc.token[acc.idx(1, 3, 7)] = 1;
    acc.maxn[acc.idx(1, 3, 7)] = 9;
    acc.tokens[1] = 2;
    acc.norm_rows[1] = 1;
    var body: std.Io.Writer.Allocating = .init(a);
    defer body.deinit();
    try writeReport(&body.writer, &acc, "default", .text, 0, "flash", zon_version);
    const text = body.written();
    var from: usize = 0;
    for (report_keys) |k| {
        var needle: [40]u8 = undefined;
        const n = try std.fmt.bufPrint(&needle, "\"{s}\"", .{k});
        const found = std.mem.indexOfPos(u8, text, from, n) orelse return error.KeyOrder;
        from = found + n.len;
    }
    const sampled_at = std.mem.indexOf(u8, text, "\"sampled\":").?;
    const nc_at = std.mem.indexOfPos(u8, text, sampled_at, "\"norm_count\"").?;
    _ = std.mem.indexOfPos(u8, text, nc_at, "\"reap_weighted_norm_sum\"").?;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), text, .{});
    const weighted = jsonF64(root.object.get("reap_weighted_norm_sum").?.array.items[3].array.items[7]).?;
    try bitEq(10, weighted); // 2.5 / 1 * 4
    const raw = jsonF64(root.object.get("sampled").?.object.get("reap_weighted_norm_sum").?.array.items[3].array.items[7]).?;
    try bitEq(2.5, raw);
    try std.testing.expectEqual(@as(i64, 4), asI64(root.object.get("token_count").?.array.items[3].array.items[7]).?);
    try std.testing.expectEqual(@as(i64, 9), asI64(root.object.get("tokens").?).?);
    var merged: std.Io.Writer.Allocating = .init(a);
    defer merged.deinit();
    try writeReport(&merged.writer, &acc, "default", .all, 0, "flash", zon_version);
    var arena2: std.heap.ArenaAllocator = .init(a);
    defer arena2.deinit();
    const mall = try std.json.parseFromSliceLeaky(std.json.Value, arena2.allocator(), merged.written(), .{});
    try bitEq(9, jsonF64(mall.object.get("max_l2_norm").?.array.items[3].array.items[7]).?);
    try std.testing.expectEqual(@as(i64, 5), asI64(mall.object.get("token_count").?.array.items[3].array.items[7]).?);
    try std.testing.expectEqual(@as(i64, 11), asI64(mall.object.get("tokens").?).?);
    // ncount 0 → weighted 0 even though token_count is 1 on the vision cell; the merge's ncount is 1
    // from text, so weighted uses the summed raw. A cell with no samples stays 0:
    try bitEq(0, jsonF64(mall.object.get("reap_weighted_norm_sum").?.array.items[0].array.items[0]).?);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    var domains = Domains.init(4);
    try snapshotAll(a, io, dir, &acc, &domains, "flash", zon_version);
    var again = try Acc.init(a, moe_layers, n_experts, 4);
    defer again.deinit(a);
    var domains2 = Domains.init(4);
    const n = try reloadDir(a, io, dir, &again, &domains2, "flash");
    try std.testing.expectEqual(@as(u32, 2), n);
    try bitEq(2.5, again.raw[again.idx(0, 3, 7)]);
    try std.testing.expectEqual(@as(i64, 4), again.token[again.idx(0, 3, 7)]);
    try std.testing.expectEqual(@as(i64, 1), again.ncount[again.idx(0, 3, 7)]);
    try std.testing.expectEqual(@as(i64, 9), again.tokens[0]);
    try std.testing.expectEqual(@as(i64, 3), again.norm_rows[0]);
    // mismatch, garbage, and a domain past the cap are renamed aside
    try tmp.dir.writeFile(io, .{ .sub_path = "zzz.text.json", .data = "{\"model_id\":\"other\",\"num_layers\":40,\"num_experts\":384}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "yyy.text.json", .data = "not json" });
    const swapped = try std.mem.replaceOwned(u8, a, text, "\"domain\":\"default\"", "\"domain\":\"extra\"");
    defer a.free(swapped);
    try tmp.dir.writeFile(io, .{ .sub_path = "extra.text.json", .data = swapped });
    // cap 1 already holds "default", so "extra" is refused and set aside
    var tight = Domains.init(1);
    var sink = try Acc.init(a, moe_layers, n_experts, 2);
    defer sink.deinit(a);
    _ = try reloadDir(a, io, dir, &sink, &tight, "flash");
    try std.testing.expect(fileExists(io, tmp.dir, "zzz.text.json.bad") or fileExists(io, tmp.dir, "zzz.text.json.bad.2"));
    try std.testing.expect(fileExists(io, tmp.dir, "yyy.text.json.bad") or fileExists(io, tmp.dir, "yyy.text.json.bad.2"));
    try std.testing.expect(fileExists(io, tmp.dir, "extra.text.json.bad") or fileExists(io, tmp.dir, "extra.text.json.bad.2"));
    try std.testing.expectEqual(@as(u32, 1), tight.n);
    try std.testing.expectEqual(@as(i64, 4), sink.token[sink.idx(0, 3, 7)]);
}

test "reap host microbench" {
    const Plan = struct {
        /// One round's host plan. A parameter so ReleaseFast cannot fold the body into a constant.
        fn once(k: u32, idx: []u32, map: []u8) u64 {
            var sink: u64 = 0;
            sink +%= promptSample(2048, k, idx);
            const hops = exchange(4, 0, 40, 1, dn_tp4);
            sink +%= hops.hops[0].bytes;
            fillDecodeBuckets(map, 0);
            sink +%= map[0];
            sink +%= idx[0];
            return sink;
        }
    };
    const a = std.testing.allocator;
    const iters: usize = if (@import("builtin").mode == .debug) 1 else 40;
    var idx: [4]u32 = undefined;
    var map: [8]u8 = undefined;
    var fnp: *const fn (u32, []u32, []u8) u64 = Plan.once;
    // Indirect call: the pure plan otherwise collapses to one multiply outside the timer.
    const vp: *volatile *const fn (u32, []u32, []u8) u64 = &fnp;
    var sink: u64 = 0;
    var t0 = monoNs();
    var n: usize = 0;
    while (n < iters * 1000) : (n += 1) sink +%= vp.*(4, &idx, &map);
    std.mem.doNotOptimizeAway(sink);
    const plan_ns = (monoNs() - t0) / (iters * 1000);
    const layers: u32 = 40;
    const rows: u32 = 8;
    const slots: u32 = 7;
    const dn: u32 = 64;
    var pick = try a.alloc(i32, layers * rows * slots);
    defer a.free(pick);
    var wts = try a.alloc(f32, pick.len);
    defer a.free(wts);
    @memset(pick, -1);
    @memset(wts, 0);
    var pos: [rows]i32 = undefined;
    var commit: [rows]u8 = undefined;
    var bucket: [rows]u8 = undefined;
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        pos[r] = @intCast(r);
        commit[r] = 1;
        bucket[r] = 0;
        var layer: u32 = 0;
        while (layer < layers) : (layer += 1) {
            var k: u32 = 0;
            while (k < topk) : (k += 1) {
                const at = pickIndex(layer, rows, r, slots, k);
                pick[at] = @intCast((r + k) % n_experts);
                wts[at] = 0.25;
            }
        }
    }
    var part = try a.alloc(f32, layers * 1 * topk * dn);
    defer a.free(part);
    @memset(part, 0.5);
    var ss0 = try a.alloc(f64, layers * topk);
    defer a.free(ss0);
    var layer: u32 = 0;
    while (layer < layers) : (layer += 1) {
        var k: u32 = 0;
        while (k < topk) : (k += 1) {
            const base = packDst(layer, 1, 0, k, dn, 0);
            ss0[ssIndex(layer, 1, 0, k)] = ssqPair(part[base..][0..dn], null);
        }
    }
    var acc = try Acc.init(a, layers, n_experts, 2);
    defer acc.deinit(a);
    t0 = monoNs();
    n = 0;
    while (n < iters) : (n += 1) {
        engineRoute(&acc, pick, wts, &pos, &commit, &bucket, layers, rows, rows, slots);
        engineNorm(&acc, pick, wts, &bucket, null, ss0, null, layers,  rows, slots, 1);
    }
    const kern_ns = (monoNs() - t0) / iters;
    var full = try Acc.init(a, moe_layers, n_experts, 2);
    defer full.deinit(a);
    var cell: usize = 0;
    while (cell < full.token.len) : (cell += 1) {
        const v: f64 = @as(f64, @floatFromInt((cell % 9) + 1)) * 0.5;
        full.token[cell] = @intCast((cell % 9) + 1);
        full.gate[cell] = v;
        full.early[cell] = @intCast((cell % 5) + 1);
        full.raw[cell] = v;
        full.ncount[cell] = @intCast((cell % 3) + 1);
        full.maxn[cell] = v;
    }
    full.tokens[0] = 3;
    full.norm_rows[0] = 1;
    const ser_buf = try a.alloc(u8, 4 * 1024 * 1024);
    defer a.free(ser_buf);
    var ser_w = std.Io.Writer.fixed(ser_buf);
    const Ser = struct {
        fn once(w: *std.Io.Writer, report: *const Acc) usize {
            w.end = 0;
            writeReport(w, report, "default", .text, 0, "flash", zon_version) catch return 0;
            return w.end;
        }
    };
    var ser_fn: *const fn (*std.Io.Writer, *const Acc) usize = Ser.once;
    const ser_vp: *volatile *const fn (*std.Io.Writer, *const Acc) usize = &ser_fn;
    const ser_iters: usize = if (@import("builtin").mode == .debug) 1 else 8;
    const warmed = ser_vp.*(&ser_w, &full);
    try std.testing.expect(warmed > 100_000);
    std.mem.doNotOptimizeAway(warmed);
    t0 = monoNs();
    var ser_sink: usize = 0;
    n = 0;
    while (n < ser_iters) : (n += 1) ser_sink += ser_vp.*(&ser_w, &full);
    std.mem.doNotOptimizeAway(ser_sink);
    const ser_ns = (monoNs() - t0) / ser_iters;
    std.debug.print("reap microbench plan {d} ns/round, cpu route+norm {d} ns (R=8 layers=40 experts=384 sample dn=64), serialize {d} ns (one bucket 40x384), d2h {d} B (16 domains)\n", .{ plan_ns, kern_ns, ser_ns, pullBytes(16) });
}

test "a query reads the engine mirror and a stalled engine does not become ready" {
    const a = std.testing.allocator;
    var acc = try Acc.init(a, 1, 1, 2);
    defer acc.deinit(a);
    const Eng = struct {
        refresh: *Refresh,
        acc: *Acc,
        src: *std.atomic.Value(i64),
        stop: *std.atomic.Value(u32),
        pulls: *std.atomic.Value(u32),

        fn loop(e: @This()) void {
            while (e.stop.load(.acquire) == 0) {
                if (e.refresh.wanted()) {
                    const v = e.src.load(.acquire);
                    e.acc.token[e.acc.idx(0, 0, 0)] = v;
                    e.acc.tokens[0] = v;
                    _ = e.pulls.fetchAdd(1, .release);
                    e.refresh.published();
                } else std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    var refresh: Refresh = .{};
    var src = std.atomic.Value(i64).init(7);
    var stop = std.atomic.Value(u32).init(0);
    var pulls = std.atomic.Value(u32).init(0);
    var http_gpu = std.atomic.Value(u32).init(0);
    const th = try std.Thread.spawn(.{}, Eng.loop, .{Eng{
        .refresh = &refresh,
        .acc = &acc,
        .src = &src,
        .stop = &stop,
        .pulls = &pulls,
    }});
    const Http = struct {
        fn query(r: *Refresh, mirror: *const Acc, gpu: *std.atomic.Value(u32), alloc: Allocator) !struct { status: u16, body: []u8 } {
            if (waitForMirror(r, 2 * std.time.ns_per_s, std.testing.io) == .stalled) return .{ .status = 503, .body = try alloc.dupe(u8, "") };
            // Serializing the mirror is the whole reply. A device pull would count here.
            _ = gpu.load(.acquire);
            var body: std.Io.Writer.Allocating = .init(alloc);
            errdefer body.deinit();
            try writeReport(&body.writer, mirror, "default", .text, 0, "dsv41-small", zon_version);
            return .{ .status = 200, .body = try body.toOwnedSlice() };
        }
    };
    const first = try Http.query(&refresh, &acc, &http_gpu, a);
    defer a.free(first.body);
    try std.testing.expectEqual(@as(u16, 200), first.status);
    try std.testing.expect(std.mem.indexOf(u8, first.body, "\"tokens\":7") != null);
    try std.testing.expect(std.mem.indexOf(u8, first.body, "\"token_count\":[[7]]") != null);
    try std.testing.expectEqual(@as(u32, 0), http_gpu.load(.acquire));
    try std.testing.expect(pulls.load(.acquire) >= 1);
    src.store(9, .release);
    const second = try Http.query(&refresh, &acc, &http_gpu, a);
    defer a.free(second.body);
    try std.testing.expectEqual(@as(u16, 200), second.status);
    try std.testing.expect(std.mem.indexOf(u8, second.body, "\"tokens\":9") != null);
    try std.testing.expectEqual(@as(u32, 0), http_gpu.load(.acquire));
    stop.store(1, .release);
    th.join();

    var stalled: Refresh = .{};
    const t0 = monoNs();
    try std.testing.expectEqual(MirrorWait.stalled, waitForMirror(&stalled, 40 * std.time.ns_per_ms, std.testing.io));
    const waited = monoNs() -% t0;
    try std.testing.expect(waited >= 40 * std.time.ns_per_ms);
    try std.testing.expect(waited < std.time.ns_per_s);
}

fn fillExportSmall(acc: *Acc) void {
    var b: u32 = 0;
    while (b < acc.nbucket) : (b += 1) {
        acc.tokens[b] = 100 + @as(i64, b) * 7;
        acc.norm_rows[b] = 3 + @as(i64, b);
        var layer: u32 = 0;
        while (layer < acc.layers) : (layer += 1) {
            var e: u32 = 0;
            while (e < acc.nexp) : (e += 1) {
                const at = acc.idx(b, layer, e);
                const ncount: i64 = if (e % 11 == 0) 0 else 2;
                acc.token[at] = @as(i64, (b + 1) * 3 + layer) + @as(i64, e % 4);
                acc.ncount[at] = ncount;
                acc.raw[at] = if (ncount == 0) 1.5 else @floatFromInt((@as(i64, e % 7) + 1) * 2);
                acc.gate[at] = @as(f64, @floatFromInt(@as(i64, (e % 5) + 1))) * 0.25;
                acc.early[at] = if (layer == 0 and e < 4) @as(i64, e + 1) else 0;
                acc.maxn[at] = if (ncount == 0) 0 else if (e % 3 == 0) 2.5 else 0.5;
            }
        }
    }
}

fn checkPrintedIdentity(root: std.json.Value) !void {
    const sal = root.object.get("reap_saliency").?.array;
    const wsum = root.object.get("reap_weighted_norm_sum").?.array;
    const tok = root.object.get("token_count").?.array;
    const nc = root.object.get("sampled").?.object.get("norm_count").?.array;
    var saw_zero = false;
    var saw_sampled = false;
    for (sal.items, wsum.items, tok.items, nc.items) |sl, wl, tl, nl| {
        for (sl.array.items, wl.array.items, tl.array.items, nl.array.items) |s, w, t, n| {
            const sf = jsonF64(s).?;
            const wf = jsonF64(w).?;
            const ti = asI64(t).?;
            const ni = asI64(n).?;
            if (ni == 0) {
                saw_zero = true;
                try bitEq(0, wf);
            } else {
                saw_sampled = true;
                try bitEq(sf * @as(f64, @floatFromInt(ti)), wf);
            }
        }
    }
    try std.testing.expect(saw_zero and saw_sampled);
}

test "small export golden for one domain and for modality all" {
    const a = std.testing.allocator;
    var acc = try Acc.init(a, 3, 32, 4);
    defer acc.deinit(a);
    var domains = Domains.init(4);
    try std.testing.expectEqual(@as(u32, 1), try domains.admit("west"));
    fillExportSmall(&acc);
    const at = acc.idx(0, 0, 1);
    try std.testing.expect(acc.ncount[at] != acc.token[at]);
    try std.testing.expect(acc.maxn[at] > 0);
    try std.testing.expect(acc.early[at] > 0);
    try std.testing.expect(acc.ncount[acc.idx(0, 0, 0)] == 0);
    try std.testing.expect(acc.tokens[2] != acc.tokens[0]);

    var text_w: std.Io.Writer.Allocating = .init(a);
    defer text_w.deinit();
    try writeReport(&text_w.writer, &acc, "default", .text, 0, "dsv41-small", zon_version);
    var all_w: std.Io.Writer.Allocating = .init(a);
    defer all_w.deinit();
    try writeReport(&all_w.writer, &acc, "default", .all, 0, "dsv41-small", zon_version);
    const text = text_w.written();
    const all = all_w.written();

    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const text_root = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), text, .{});
    try std.testing.expectEqual(@as(i64, 3), asI64(text_root.object.get("num_layers").?).?);
    try std.testing.expectEqual(@as(i64, 32), asI64(text_root.object.get("num_experts").?).?);
    try std.testing.expectEqualStrings("default", text_root.object.get("domain").?.string);
    try std.testing.expectEqualStrings("text", text_root.object.get("modality").?.string);
    try std.testing.expectEqualStrings("dsv41-small", text_root.object.get("model_id").?.string);
    try std.testing.expectEqual(@as(i64, 100), asI64(text_root.object.get("tokens").?).?);
    try std.testing.expectEqual(@as(i64, 3), asI64(text_root.object.get("norm_rows").?).?);
    try bitEq(2.5, jsonF64(text_root.object.get("max_l2_norm").?.array.items[0].array.items[3]).?);
    try std.testing.expectEqual(@as(i64, 2), asI64(text_root.object.get("early_pos_count").?.array.items[0].array.items[1]).?);
    try std.testing.expect(asI64(text_root.object.get("token_count").?.array.items[0].array.items[1]).? != asI64(text_root.object.get("sampled").?.object.get("norm_count").?.array.items[0].array.items[1]).?);
    try checkPrintedIdentity(text_root);
    const all_root = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), all, .{});
    try std.testing.expectEqualStrings("all", all_root.object.get("modality").?.string);
    try std.testing.expectEqual(@as(i64, 207), asI64(all_root.object.get("tokens").?).?);
    try std.testing.expectEqual(@as(i64, 7), asI64(all_root.object.get("norm_rows").?).?);
    try checkPrintedIdentity(all_root);
    try std.testing.expectEqualSlices(u8, @embedFile("fixtures/reap_export_small.json"), text);
    try std.testing.expectEqualSlices(u8, @embedFile("fixtures/reap_export_small_all.json"), all);
}

fn monoNs() u64 {
    var ts: std.posix.timespec = undefined;
    const rc = std.posix.system.clock_gettime(std.posix.CLOCK.MONOTONIC, &ts);
    if (std.posix.errno(rc) != .SUCCESS) return 0;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
