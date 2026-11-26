//! Device side of Edge-REAP stats. Rank 0 accumulates. Every rank
//! exchanges the sampled partials. Pointers are fixed at open so a captured graph's
//! arguments do not move. The sample exchange and the stats kernels run after the
//! round, on the engine stream, outside the graph.
const std = @import("std");
const gpu = @import("gpu");
const prompt = @import("prompt.zig");
const round = @import("round.zig");
const coll = @import("coll.zig");
const reap = @import("reap.zig");
// hip_load's comptime block names gpu.kernel_abi, which the CUDA facade does not have.
// A CUDA build never opens REAP (model.open returns MissingKernel) and must not analyze that file.
const hip_load = if (gpu.backend == .hip) @import("hip_load.zig") else void;
const lanes = @import("lanes");

const Allocator = std.mem.Allocator;

pub const Kernels = struct {
    pack: gpu.Function,
    ssq: gpu.Function,
    route: gpu.Function,
    norm: gpu.Function,
};

pub const State = struct {
    gpa: Allocator,
    io: std.Io,
    eng: *const prompt.Engine,
    cfg: reap.Config,
    acc: reap.Acc,
    domains: reap.Domains,
    rank: u32,
    world: u32,
    moe: u32,
    route_slots: u32,
    y_slots: u32,
    dn: u32,
    uniform_y: bool,
    max_sample: u32,
    round_i: u64 = 0,
    domain: u32 = 0,
    dev_mu: std.atomic.Mutex = .unlocked,
    ks: Kernels,
    /// One decode-round bank. Prompt chunks pack from scratch y, so there is no second bank.
    y: gpu.DeviceBuffer,
    pick_dec: gpu.DeviceBuffer,
    wts_dec: gpu.DeviceBuffer,
    pick_pr: gpu.DeviceBuffer,
    wts_pr: gpu.DeviceBuffer,
    pack_buf: gpu.DeviceBuffer,
    recv: gpu.DeviceBuffer,
    ss0: gpu.DeviceBuffer,
    ss1: gpu.DeviceBuffer,
    rows_idx: [2]gpu.DeviceBuffer,
    bucket_pr: gpu.DeviceBuffer,
    pos_pr: gpu.DeviceBuffer,
    commit_pr: gpu.DeviceBuffer,
    bucket_dec: gpu.DeviceBuffer,
    pos_dec: gpu.DeviceBuffer,
    commit_dec: gpu.DeviceBuffer,
    d_token: gpu.DeviceBuffer,
    d_gate: gpu.DeviceBuffer,
    d_early: gpu.DeviceBuffer,
    d_raw: gpu.DeviceBuffer,
    d_ncount: gpu.DeviceBuffer,
    d_max: gpu.DeviceBuffer,
    sample_rows: [2][reap.rows_cap_prompt]u32 = undefined,
    n_sample: [2]u32 = .{ 0, 0 },
    /// 0 unset, 1 y staged, 2 pf packed, 3 skip.
    got: [2][reap.moe_layers]u8 = .{ @splat(0), @splat(0) },
    live: reap.Live = undefined,
    last_snap: i128 = 0,
    hooked: bool = false,
    y_rows: u32 = reap.rows_cap_y,
    refresh: reap.Refresh = .{},
    closing: std.atomic.Value(u32) = .init(0),
    inflight: std.atomic.Value(u32) = .init(0),
    snap_acc: reap.Acc = .{
        .layers = 0,
        .nexp = 0,
        .nbucket = 0,
        .token = &.{},
        .gate = &.{},
        .early = &.{},
        .raw = &.{},
        .ncount = &.{},
        .maxn = &.{},
        .tokens = &.{},
        .norm_rows = &.{},
    },
    snap_domains: reap.Domains = .{ .cap = 0 },
    snap_busy: std.atomic.Value(u32) = .init(0),
    snap_go: std.atomic.Value(u32) = .init(0),
    snap_stop: std.atomic.Value(u32) = .init(0),
    writer: ?std.Thread = null,
};

fn emptyBuf(d: *const gpu.Driver) gpu.DeviceBuffer {
    return .{ .d = d, .ptr = 0, .len = 0 };
}

fn emptyAcc() reap.Acc {
    return .{
        .layers = 0,
        .nexp = 0,
        .nbucket = 0,
        .token = &.{},
        .gate = &.{},
        .early = &.{},
        .raw = &.{},
        .ncount = &.{},
        .maxn = &.{},
        .tokens = &.{},
        .norm_rows = &.{},
    };
}

fn dropBuf(b: *gpu.DeviceBuffer) void {
    if (b.ptr == 0 and b.len == 0) return;
    b.free();
    b.* = .{ .d = undefined, .ptr = 0, .len = 0 };
}

fn lock(st: *State) void {
    while (!st.dev_mu.tryLock()) std.Thread.yield() catch {};
}

fn unlock(st: *State) void {
    st.dev_mu.unlock();
}

fn nowNs(io: std.Io) i128 {
    return std.Io.Timestamp.now(io, .awake).nanoseconds;
}

fn gridOf(n: u64, block: u32) u32 {
    if (n == 0) return 1;
    const g = (n + block - 1) / block;
    return if (g > 65535) 65535 else @intCast(g);
}

fn launch(f: gpu.Function, stream: gpu.Stream, grid: u32, block: u32, args: *gpu.Args) !void {
    try gpu.launch.launch(f, .{ .grid = .{ .x = grid }, .block = .{ .x = block } }, stream, args);
}

/// y stride the down kernel writes. TP2 with a shared expert compacts to slots - 1.
/// A mix of the two across MoE layers turns the norm tier off on every rank.
fn yLayout(e: *const prompt.Engine) struct { route: u32, y: u32, uniform: bool, dn: u32 } {
    const route: u32 = @intCast(e.slots());
    const dn: u32 = if (e.two != null) reap.dn_tp4 else reap.dn_tp2;
    if (e.two != null) return .{ .route = route, .y = route, .uniform = true, .dn = dn };
    var shared: u32 = 0;
    var plain: u32 = 0;
    const n = @min(e.w.layers.len, reap.moe_layers);
    for (e.w.layers[0..n]) |lay| {
        if (lay.shared != null) shared += 1 else plain += 1;
    }
    if (shared > 0 and plain > 0) return .{ .route = route, .y = route, .uniform = false, .dn = dn };
    if (shared > 0) return .{ .route = route, .y = route - 1, .uniform = true, .dn = dn };
    return .{ .route = route, .y = route, .uniform = true, .dn = dn };
}

pub fn open(gpa: Allocator, io: std.Io, d: *const gpu.Driver, e: *const prompt.Engine, ks: Kernels) !*State {
    var cfg = try reap.fromEnv(gpa);
    if (e.comm.world >= 2 and e.comm.vtable.exchange == null) {
        std.log.err("reap: world {d} collective has no exchange; refusing to open", .{e.comm.world});
        cfg.deinit(gpa);
        return error.NoExchange;
    }
    const st = gpa.create(State) catch |err| {
        cfg.deinit(gpa);
        return err;
    };
    const lay = yLayout(e);
    const sample: u32 = @max(@min(cfg.prefill_rows, reap.rows_cap_prompt), 1);
    const nb = cfg.nbucket();
    const cells: usize = @as(usize, nb) * reap.moe_layers * reap.n_experts;
    st.* = .{
        .gpa = gpa,
        .io = io,
        .eng = e,
        .cfg = cfg,
        .acc = emptyAcc(),
        .domains = reap.Domains.init(cfg.buckets),
        .rank = e.comm.rank,
        .world = e.comm.world,
        .moe = @intCast(@min(e.w.layers.len, reap.moe_layers)),
        .route_slots = lay.route,
        .y_slots = lay.y,
        .dn = lay.dn,
        .uniform_y = lay.uniform,
        .max_sample = sample,
        .ks = ks,
        .y = emptyBuf(d),
        .pick_dec = emptyBuf(d),
        .wts_dec = emptyBuf(d),
        .pick_pr = emptyBuf(d),
        .wts_pr = emptyBuf(d),
        .pack_buf = emptyBuf(d),
        .recv = emptyBuf(d),
        .ss0 = emptyBuf(d),
        .ss1 = emptyBuf(d),
        .rows_idx = .{ emptyBuf(d), emptyBuf(d) },
        .bucket_pr = emptyBuf(d),
        .pos_pr = emptyBuf(d),
        .commit_pr = emptyBuf(d),
        .bucket_dec = emptyBuf(d),
        .pos_dec = emptyBuf(d),
        .commit_dec = emptyBuf(d),
        .d_token = emptyBuf(d),
        .d_gate = emptyBuf(d),
        .d_early = emptyBuf(d),
        .d_raw = emptyBuf(d),
        .d_ncount = emptyBuf(d),
        .d_max = emptyBuf(d),
    };
    cfg.dir_owned = false;
    cfg.name_owned = false;
    cfg.build_owned = false;
    errdefer fail(st);
    if (!lay.uniform) std.log.warn("reap: MoE layers disagree on down slots; norm tier off, routing counters stay on", .{});
    const dec_pick = @as(usize, reap.moe_layers) * reap.rows_cap_decode * lay.route * 4;
    const pr_pick = 2 * @as(usize, reap.moe_layers) * reap.rows_cap_prompt * lay.route * 4;
    st.pick_dec = try gpu.DeviceBuffer.alloc(d, dec_pick);
    st.wts_dec = try gpu.DeviceBuffer.alloc(d, dec_pick);
    st.pick_pr = try gpu.DeviceBuffer.alloc(d, pr_pick);
    st.wts_pr = try gpu.DeviceBuffer.alloc(d, pr_pick);
    st.bucket_pr = try gpu.DeviceBuffer.alloc(d, reap.rows_cap_prompt);
    st.pos_pr = try gpu.DeviceBuffer.alloc(d, reap.rows_cap_prompt * 4);
    st.commit_pr = try gpu.DeviceBuffer.alloc(d, reap.rows_cap_prompt);
    st.bucket_dec = try gpu.DeviceBuffer.alloc(d, reap.rows_cap_decode);
    st.pos_dec = try gpu.DeviceBuffer.alloc(d, reap.rows_cap_decode * 4);
    st.commit_dec = try gpu.DeviceBuffer.alloc(d, reap.rows_cap_decode);
    const vec = @as(u64, reap.moe_layers) * sample * reap.topk * lay.dn * 4;
    const ss = @as(u64, reap.moe_layers) * sample * reap.topk * 8;
    st.pack_buf = try gpu.DeviceBuffer.alloc(d, @intCast(vec * reap.pack_banks));
    st.recv = try gpu.DeviceBuffer.alloc(d, @intCast(vec));
    st.ss0 = try gpu.DeviceBuffer.alloc(d, @intCast(ss));
    st.ss1 = try gpu.DeviceBuffer.alloc(d, @intCast(ss));
    st.rows_idx[0] = try gpu.DeviceBuffer.alloc(d, sample * 4);
    st.rows_idx[1] = try gpu.DeviceBuffer.alloc(d, sample * 4);
    if (lay.uniform) {
        const yb = @as(u64, reap.moe_layers) * st.y_rows * lay.y * lay.dn * 4;
        st.y = try gpu.DeviceBuffer.alloc(d, @intCast(yb));
    }
    if (st.rank == 0) {
        st.acc = try reap.Acc.init(gpa, reap.moe_layers, reap.n_experts, nb);
        const nb8 = cells * 8;
        st.d_token = try gpu.DeviceBuffer.alloc(d, nb8);
        st.d_gate = try gpu.DeviceBuffer.alloc(d, nb8);
        st.d_early = try gpu.DeviceBuffer.alloc(d, nb8);
        st.d_raw = try gpu.DeviceBuffer.alloc(d, nb8);
        st.d_ncount = try gpu.DeviceBuffer.alloc(d, nb8);
        st.d_max = try gpu.DeviceBuffer.alloc(d, nb8);
        try gpu.fill8Async(d, e.s.handle, st.d_token.ptr, 0, nb8);
        try gpu.fill8Async(d, e.s.handle, st.d_gate.ptr, 0, nb8);
        try gpu.fill8Async(d, e.s.handle, st.d_early.ptr, 0, nb8);
        try gpu.fill8Async(d, e.s.handle, st.d_raw.ptr, 0, nb8);
        try gpu.fill8Async(d, e.s.handle, st.d_ncount.ptr, 0, nb8);
        try gpu.fill8Async(d, e.s.handle, st.d_max.ptr, 0, nb8);
        try e.s.synchronize();
        if (st.cfg.dir.len > 0) {
            const n = try reap.reloadDir(gpa, io, st.cfg.dir, &st.acc, &st.domains, st.cfg.served_name);
            if (n > 0) std.log.info("reap: reloaded {d} snapshot(s) from {s}", .{ n, st.cfg.dir });
            try gpu.copyH2D(d, st.d_token.ptr, std.mem.sliceAsBytes(st.acc.token));
            try gpu.copyH2D(d, st.d_gate.ptr, std.mem.sliceAsBytes(st.acc.gate));
            try gpu.copyH2D(d, st.d_early.ptr, std.mem.sliceAsBytes(st.acc.early));
            try gpu.copyH2D(d, st.d_raw.ptr, std.mem.sliceAsBytes(st.acc.raw));
            try gpu.copyH2D(d, st.d_ncount.ptr, std.mem.sliceAsBytes(st.acc.ncount));
            try gpu.copyH2D(d, st.d_max.ptr, std.mem.sliceAsBytes(st.acc.maxn));
        }
        st.live = .{
            .cfg = st.cfg,
            .domains = &st.domains,
            .acc = &st.acc,
            .model_id = st.cfg.served_name,
            .build = st.cfg.build,
            .pull = pull,
            .pull_ctx = st,
        };
        reap.install(&st.live);
        lanes.reap_hook.install(st, queryHook, admitHook, serviceHook);
        st.hooked = true;
        if (st.cfg.dir.len > 0) {
            st.snap_acc = try reap.Acc.init(gpa, reap.moe_layers, reap.n_experts, nb);
            st.snap_domains = reap.Domains.init(st.cfg.buckets);
            st.writer = try std.Thread.spawn(.{}, writerMain, .{st});
        }
    }
    var ones: [reap.rows_cap_prompt]u8 = @splat(1);
    try prompt.upload(e, st.commit_pr.ptr, &ones, ones.len);
    st.last_snap = nowNs(io);
    const got = deviceTotal(st);
    const formula = reap.deviceBytes(.{
        .route_slots = st.route_slots,
        .y_slots = st.y_slots,
        .dn = st.dn,
        .sample = sample,
        .buckets = st.cfg.buckets,
        .rank0 = st.rank == 0,
        .uniform = st.uniform_y,
    });
    if (got != formula) std.log.err("reap: device bytes {d} do not match the formula {d}", .{ got, formula });
    std.log.info("reap: rank {d} device bytes {d}", .{ st.rank, got });
    return st;
}

fn fail(st: *State) void {
    st.closing.store(1, .release);
    if (st.hooked) {
        lanes.reap_hook.clear(st);
        reap.uninstall(&st.live);
        st.hooked = false;
    }
    if (st.writer) |th| {
        st.snap_stop.store(1, .release);
        th.join();
        st.writer = null;
    }
    freeDevices(st);
    st.snap_acc.deinit(st.gpa);
    st.acc.deinit(st.gpa);
    st.cfg.deinit(st.gpa);
    st.gpa.destroy(st);
}

fn freeDevices(st: *State) void {
    dropBuf(&st.y);
    dropBuf(&st.pick_dec);
    dropBuf(&st.wts_dec);
    dropBuf(&st.pick_pr);
    dropBuf(&st.wts_pr);
    dropBuf(&st.pack_buf);
    dropBuf(&st.recv);
    dropBuf(&st.ss0);
    dropBuf(&st.ss1);
    dropBuf(&st.rows_idx[0]);
    dropBuf(&st.rows_idx[1]);
    dropBuf(&st.bucket_pr);
    dropBuf(&st.pos_pr);
    dropBuf(&st.commit_pr);
    dropBuf(&st.bucket_dec);
    dropBuf(&st.pos_dec);
    dropBuf(&st.commit_dec);
    dropBuf(&st.d_token);
    dropBuf(&st.d_gate);
    dropBuf(&st.d_early);
    dropBuf(&st.d_raw);
    dropBuf(&st.d_ncount);
    dropBuf(&st.d_max);
}

/// Stop the hook, let an in-flight query leave, join the writer, then the shutdown
/// snapshot (synchronous: close is not on the serving path).
pub fn close(st: *State) void {
    if (st.hooked) {
        lanes.reap_hook.clear(st);
        reap.uninstall(&st.live);
        st.hooked = false;
    }
    st.closing.store(1, .release);
    _ = st.refresh.seq.fetchAdd(1, .release);
    while (st.inflight.load(.acquire) != 0) std.Thread.yield() catch {};
    if (st.writer) |th| {
        st.snap_stop.store(1, .release);
        th.join();
        st.writer = null;
    }
    if (st.rank == 0 and st.cfg.dir.len > 0) {
        lock(st);
        if (pullIntoAcc(st)) {
            reap.snapshotAll(st.gpa, st.io, st.cfg.dir, &st.acc, &st.domains, st.cfg.served_name, st.cfg.build) catch |e| {
                std.log.err("reap: shutdown snapshot failed ({s})", .{@errorName(e)});
            };
        } else |e| std.log.err("reap: shutdown copy failed ({s})", .{@errorName(e)});
        unlock(st);
    }
    freeDevices(st);
    st.snap_acc.deinit(st.gpa);
    st.acc.deinit(st.gpa);
    st.cfg.deinit(st.gpa);
    st.gpa.destroy(st);
}

pub fn setDomain(st: *State, domain: u32) void {
    if (st.rank != 0) return;
    st.domain = if (domain < st.cfg.buckets) domain else 0;
}

pub fn picksOf(ctx: *anyopaque, which: u8, bank: u8, li: u32) reap.PickPair {
    const st: *State = @ptrCast(@alignCast(ctx));
    if (li >= reap.moe_layers) return .{};
    const slots = st.route_slots;
    if (which == 0) {
        const off = @as(u64, li) * reap.rows_cap_decode * slots * 4;
        return .{ .pick = st.pick_dec.ptr + off, .wts = st.wts_dec.ptr + off };
    }
    const b: u64 = if (bank == 0) 0 else 1;
    const off = (b * reap.moe_layers + li) * reap.rows_cap_prompt * slots * 4;
    return .{ .pick = st.pick_pr.ptr + off, .wts = st.wts_pr.ptr + off };
}

pub fn downYOf(ctx: *anyopaque, which: u8, bank: u8, li: u32, slots: u32) u64 {
    _ = bank;
    const st: *State = @ptrCast(@alignCast(ctx));
    // Prompt chunks (which != 0) pack from scratch y. Only a decode round redirects.
    if (which != 0) return 0;
    if (!st.uniform_y or li >= reap.moe_layers or slots != st.y_slots or st.y.ptr == 0) return 0;
    const off = @as(u64, li) * st.y_rows * st.y_slots * st.dn * 4;
    return st.y.ptr + off;
}

pub fn noteLayer(ctx: *anyopaque, note: reap.LayerNote) anyerror!void {
    const st: *State = @ptrCast(@alignCast(ctx));
    if (note.li >= reap.moe_layers or note.bank > 1) return;
    lock(st);
    defer unlock(st);
    const bank = note.bank;
    if (note.capture == .skip or !st.uniform_y) {
        st.got[bank][note.li] = 3;
        return;
    }
    if (note.capture == .y_staged) {
        st.got[bank][note.li] = 1;
        return;
    }
    const ns = st.n_sample[bank];
    if (ns == 0 or note.src == 0) {
        st.got[bank][note.li] = 3;
        return;
    }
    // Tight [layers, n_sample, topk, dn] inside this bank's pack region. max_sample only sizes the region.
    const stride = @as(u64, ns) * reap.topk * st.dn * 4;
    try launchPack(st, note.src, note.rows, note.slots, 1, ns, st.rows_idx[bank].ptr, packBase(st, bank) + @as(u64, note.li) * stride);
    st.got[bank][note.li] = 2;
}

pub fn beginPrompt(st: *State, ch: *const prompt.Chunk) !void {
    const bank: usize = if (ch.reap_bank == 0) 0 else 1;
    lock(st);
    defer unlock(st);
    const n: u32 = @intCast(@min(ch.n, reap.rows_cap_prompt));
    const ns = reap.promptSample(n, st.cfg.prefill_rows, &st.sample_rows[bank]);
    st.n_sample[bank] = ns;
    @memset(&st.got[bank], 0);
    if (ns == 0) return;
    var idx: [reap.rows_cap_prompt]i32 = undefined;
    var i: u32 = 0;
    while (i < ns) : (i += 1) idx[i] = @intCast(st.sample_rows[bank][i]);
    try prompt.upload(st.eng, st.rows_idx[bank].ptr, &idx, ns * 4);
}

pub fn finishPrompt(st: *State, ch: *const prompt.Chunk, layers_ran: u32) !void {
    lock(st);
    finishLocked(st, ch, layers_ran) catch |e| {
        unlock(st);
        return e;
    };
    unlock(st);
    engineService(st);
}

fn finishLocked(st: *State, ch: *const prompt.Chunk, layers_ran: u32) !void {
    const bank: usize = if (ch.reap_bank == 0) 0 else 1;
    const layers = @min(layers_ran, st.moe);
    const n: u32 = @intCast(@min(ch.n, reap.rows_cap_prompt));
    var buckets: [reap.rows_cap_prompt]u8 = undefined;
    var pos: [reap.rows_cap_prompt]i32 = undefined;
    if (n > 0) fillPromptRows(buckets[0..n], pos[0..n], @intCast(ch.start), st.domain, ch.spans);
    if (st.rank == 0 and n > 0 and layers > 0) {
        try prompt.upload(st.eng, st.bucket_pr.ptr, &buckets, n);
        try prompt.upload(st.eng, st.pos_pr.ptr, &pos, n * 4);
        const base = picksOf(st, 1, ch.reap_bank, 0);
        try launchRoute(st, base.pick, base.wts, st.pos_pr.ptr, st.commit_pr.ptr, st.bucket_pr.ptr, layers, n, reap.rows_cap_prompt);
        addCommitted(st, null, buckets[0..n]);
    }
    if (!st.uniform_y or st.n_sample[bank] == 0 or layers == 0) return;
    var kind: u8 = 0;
    var bad = false;
    var li: u32 = 0;
    while (li < layers) : (li += 1) {
        const g = st.got[bank][li];
        if (g == 0 or g == 3) bad = true else if (kind == 0) kind = g else if (g != kind) bad = true;
    }
    if (bad or kind == 0) return;
    const ns = st.n_sample[bank];
    const region = packBase(st, bank);
    // Prompts pack per layer (kind 2) into `region`. Kind 1 is a redirected y, which prompts no longer use.
    if (kind == 1) {
        try launchPack(st, st.y.ptr, st.y_rows, st.y_slots, layers, ns, st.rows_idx[bank].ptr, region);
    }
    try exchangeAndSsq(st, region, layers, ns);
    if (st.rank != 0 or n == 0) return;
    const picks = picksOf(st, 1, ch.reap_bank, 0);
    try launchNorm(st, picks.pick, picks.wts, st.bucket_pr.ptr, st.rows_idx[bank].ptr, layers, reap.rows_cap_prompt, ns);
    addNormRows(st, buckets[0..n], st.sample_rows[bank][0..ns]);
}

pub fn afterDecode(st: *State, buckets: []const u8) !void {
    lock(st);
    defer unlock(st);
    const sample = reap.decodeSample(st.round_i, st.cfg.every) and st.uniform_y and buckets.len > 0 and st.y.ptr != 0;
    st.round_i += 1;
    if (!sample) return;
    const layers = st.moe;
    try launchPack(st, st.y.ptr, st.y_rows, st.y_slots, layers, 1, 0, st.pack_buf.ptr);
    try exchangeAndSsq(st, st.pack_buf.ptr, layers, 1);
    if (st.rank != 0) return;
    try prompt.upload(st.eng, st.bucket_dec.ptr, @ptrCast(buckets.ptr), 1);
    const base = picksOf(st, 0, 0, 0);
    try launchNorm(st, base.pick, base.wts, st.bucket_dec.ptr, 0, layers, reap.rows_cap_decode, 1);
    addNormRows(st, buckets, null);
}

pub fn noteDecodeRoute(st: *State, commit: []const u8, buckets: []const u8, pos: []const i32) !void {
    if (st.rank != 0) return;
    lock(st);
    noteDecodeLocked(st, commit, buckets, pos) catch |e| {
        unlock(st);
        return e;
    };
    unlock(st);
    engineService(st);
}

fn noteDecodeLocked(st: *State, commit: []const u8, buckets: []const u8, pos: []const i32) !void {
    const n: u32 = @intCast(@min(commit.len, reap.rows_cap_decode));
    if (n == 0) return;
    try prompt.upload(st.eng, st.commit_dec.ptr, @ptrCast(commit.ptr), n);
    try prompt.upload(st.eng, st.bucket_dec.ptr, @ptrCast(buckets.ptr), n);
    try prompt.upload(st.eng, st.pos_dec.ptr, @ptrCast(pos.ptr), n * 4);
    const base = picksOf(st, 0, 0, 0);
    try launchRoute(st, base.pick, base.wts, st.pos_dec.ptr, st.commit_dec.ptr, st.bucket_dec.ptr, st.moe, n, reap.rows_cap_decode);
    addCommitted(st, commit[0..n], buckets[0..n]);
}

fn launchPack(st: *State, src: u64, rows_cap: u32, slots: u32, layers: u32, n_sample: u32, rows_idx: u64, dst: u64) !void {
    var a: gpu.Args = .{};
    a.add(src);
    a.add(dst);
    a.add(rows_idx);
    a.add(layers);
    a.add(rows_cap);
    a.add(slots);
    a.add(reap.topk);
    a.add(st.dn);
    a.add(n_sample);
    const n = @as(u64, layers) * n_sample * reap.topk * st.dn;
    try launch(st.ks.pack, st.eng.s, gridOf(n, 256), 256, &a);
}

fn launchSsq(st: *State, y0: u64, y1: u64, nvec: u32) !void {
    var a: gpu.Args = .{};
    a.add(y0);
    a.add(y1);
    a.add(st.ss0.ptr);
    a.add(nvec);
    a.add(st.dn);
    try launch(st.ks.ssq, st.eng.s, gridOf(nvec, 256), 256, &a);
}

fn launchRoute(st: *State, pick: u64, wts: u64, pos: u64, commit: u64, bucket: u64, layers: u32, rows: u32, row_stride: u32) !void {
    var a: gpu.Args = .{};
    a.add(pick);
    a.add(wts);
    a.add(pos);
    a.add(commit);
    a.add(bucket);
    a.add(st.d_token.ptr);
    a.add(st.d_gate.ptr);
    a.add(st.d_early.ptr);
    a.add(layers);
    a.add(rows);
    a.add(row_stride);
    a.add(st.route_slots);
    a.add(reap.topk);
    a.add(reap.n_experts);
    a.add(st.cfg.nbucket());
    const n = @as(u64, layers) * reap.n_experts;
    try launch(st.ks.route, st.eng.s, gridOf(n, 128), 128, &a);
}

fn launchNorm(st: *State, pick: u64, wts: u64, bucket: u64, rows_idx: u64, layers: u32, row_stride: u32, n_sample: u32) !void {
    var a: gpu.Args = .{};
    a.add(pick);
    a.add(wts);
    a.add(bucket);
    a.add(rows_idx);
    a.add(st.ss0.ptr);
    a.add(if (st.world >= 4) st.ss1.ptr else @as(u64, 0));
    a.add(st.d_raw.ptr);
    a.add(st.d_ncount.ptr);
    a.add(st.d_max.ptr);
    a.add(layers);
    a.add(row_stride);
    a.add(st.route_slots);
    a.add(reap.topk);
    a.add(reap.n_experts);
    a.add(st.cfg.nbucket());
    a.add(n_sample);
    const n = @as(u64, layers) * reap.n_experts;
    try launch(st.ks.norm, st.eng.s, gridOf(n, 128), 128, &a);
}

fn exchangeAndSsq(st: *State, src: u64, layers: u32, n_sample: u32) !void {
    const hops = reap.exchange(st.world, st.rank, layers, n_sample, st.dn);
    if (hops.n >= 1) {
        const h = hops.hops[0];
        const nbytes: usize = @intCast(h.bytes);
        const peer = st.rank ^ h.peer_xor;
        try st.eng.comm.exchange(coll.Site.at(.reap_pair, 40), src, nbytes, st.recv.ptr, nbytes, .u8, peer, st.eng.s);
    }
    const do_ssq = if (st.world <= 1) st.rank == 0 else st.rank % 2 == 0;
    if (do_ssq) {
        const y1: u64 = if (st.world <= 1) 0 else st.recv.ptr;
        try launchSsq(st, src, y1, layers * n_sample * reap.topk);
    }
    if (hops.n >= 2) {
        const h = hops.hops[1];
        const nbytes: usize = @intCast(h.bytes);
        const peer = st.rank ^ h.peer_xor;
        try st.eng.comm.exchange(coll.Site.at(.reap_scalar, 41), st.ss0.ptr, nbytes, st.ss1.ptr, nbytes, .u8, peer, st.eng.s);
    }
}

fn fillPromptRows(buckets: []u8, pos: []i32, start: u32, domain: u32, spans: []const prompt.Span) void {
    for (buckets, 0..) |*b, row| {
        const p: u32 = start + @as(u32, @intCast(row));
        if (pos.len == buckets.len) pos[row] = @intCast(p);
        var vis: u8 = 0;
        for (spans) |sp| {
            const at: u32 = @intCast(sp.at);
            const len: u32 = @intCast(sp.len);
            if (p >= at and p - at < len) vis = 1;
        }
        b.* = @intCast(domain * 2 + vis);
    }
}

fn addCommitted(st: *State, commit: ?[]const u8, buckets: []const u8) void {
    if (st.rank != 0 or st.acc.tokens.len == 0) return;
    for (buckets, 0..) |b, i| {
        if (commit) |c| if (c[i] == 0) continue;
        if (b < st.acc.nbucket) st.acc.tokens[b] += 1;
    }
}

fn addNormRows(st: *State, buckets: []const u8, rows_idx: ?[]const u32) void {
    if (st.rank != 0 or st.acc.norm_rows.len == 0 or buckets.len == 0) return;
    if (rows_idx) |idx| {
        for (idx) |row| {
            if (row >= buckets.len) continue;
            const b = buckets[row];
            if (b < st.acc.nbucket) st.acc.norm_rows[b] += 1;
        }
        return;
    }
    const b = buckets[0];
    if (b < st.acc.nbucket) st.acc.norm_rows[b] += 1;
}

fn pullIntoAcc(st: *State) !void {
    if (st.rank != 0 or st.d_token.ptr == 0) return;
    try st.eng.s.synchronize();
    try gpu.copyD2H(st.eng.d, std.mem.sliceAsBytes(st.acc.token), st.d_token.ptr);
    try gpu.copyD2H(st.eng.d, std.mem.sliceAsBytes(st.acc.gate), st.d_gate.ptr);
    try gpu.copyD2H(st.eng.d, std.mem.sliceAsBytes(st.acc.early), st.d_early.ptr);
    try gpu.copyD2H(st.eng.d, std.mem.sliceAsBytes(st.acc.raw), st.d_raw.ptr);
    try gpu.copyD2H(st.eng.d, std.mem.sliceAsBytes(st.acc.ncount), st.d_ncount.ptr);
    try gpu.copyD2H(st.eng.d, std.mem.sliceAsBytes(st.acc.maxn), st.d_max.ptr);
}

fn pull(ctx: *anyopaque, dst: *reap.Acc) anyerror!void {
    const st: *State = @ptrCast(@alignCast(ctx));
    lock(st);
    defer unlock(st);
    try pullIntoAcc(st);
    if (dst != &st.acc) dst.copyFrom(&st.acc);
}

fn queryHook(ctx: *anyopaque, q: *lanes.reap_hook.Query) void {
    const st: *State = @ptrCast(@alignCast(ctx));
    _ = st.inflight.fetchAdd(1, .acq_rel);
    defer _ = st.inflight.fetchSub(1, .acq_rel);
    answer(st, q) catch |e| {
        q.status = 500;
        q.out = std.fmt.allocPrint(q.allocator, "{{\"error\":{{\"message\":\"reap export failed: {s}\"}}}}", .{@errorName(e)}) catch null;
    };
}

fn refreshing(q: *lanes.reap_hook.Query) !void {
    q.status = 503;
    q.out = try q.allocator.dupe(u8, "{\"error\":{\"message\":\"reap stats are refreshing\"}}");
}

/// Wait for the engine thread to publish the host mirror, then serialize that mirror.
/// This runs on the HTTP thread and does not call the device.
fn answer(st: *State, q: *lanes.reap_hook.Query) !void {
    var modality: reap.Modality = .all;
    var idx: u32 = 0;
    if (q.domain.len != 0) {
        modality = reap.modalityOf(q.modality) orelse {
            q.status = 400;
            return;
        };
        idx = st.domains.lookup(q.domain) orelse {
            q.status = 404;
            return;
        };
    }
    if (st.closing.load(.acquire) != 0) return refreshing(q);
    if (reap.waitForMirror(&st.refresh, reap.refresh_wait_ns, st.io) == .stalled or st.closing.load(.acquire) != 0) return refreshing(q);
    lock(st);
    defer unlock(st);
    if (st.closing.load(.acquire) != 0) return refreshing(q);
    var body: std.Io.Writer.Allocating = .init(q.allocator);
    errdefer body.deinit();
    if (q.domain.len == 0) {
        try reap.writeList(&body.writer, &st.domains, &st.acc);
    } else {
        const name = st.domains.name(idx);
        var name_buf: [reap.max_name]u8 = undefined;
        const name_len = @min(name.len, name_buf.len);
        @memcpy(name_buf[0..name_len], name[0..name_len]);
        try reap.writeReport(&body.writer, &st.acc, name_buf[0..name_len], modality, idx, st.cfg.served_name, st.cfg.build);
    }
    q.status = 200;
    q.out = try body.toOwnedSlice();
}

fn admitHook(ctx: *anyopaque, domain: []const u8, index: *u32, err: []u8) usize {
    const st: *State = @ptrCast(@alignCast(ctx));
    const i = st.domains.admit(domain) catch |e| switch (e) {
        error.Invalid => {
            const m = reap.invalid_domain;
            const n = @min(m.len, err.len);
            @memcpy(err[0..n], m[0..n]);
            return n;
        },
        error.Cap => return reap.formatCap(err, st.cfg.buckets, if (domain.len == 0) "default" else domain).len,
    };
    index.* = i;
    return 0;
}

fn engineService(st: *State) void {
    if (st.rank != 0 or st.closing.load(.acquire) != 0) return;
    maybeSnapshot(st);
    serviceQuery(st);
}

fn serviceHook(ctx: *anyopaque) void {
    engineService(@ptrCast(@alignCast(ctx)));
}

fn serviceQuery(st: *State) void {
    if (!st.refresh.wanted()) return;
    lock(st);
    defer unlock(st);
    if (st.closing.load(.acquire) != 0 or !st.refresh.wanted()) return;
    pullIntoAcc(st) catch |e| {
        std.log.err("reap: export copy failed ({s})", .{@errorName(e)});
        return;
    };
    st.refresh.published();
}

/// D2H into the live mirror, then a copy the writer thread serializes. The engine does not write the files.
fn maybeSnapshot(st: *State) void {
    if (st.cfg.dir.len == 0 or st.cfg.snapshot_min == 0 or st.writer == null) return;
    const step = @as(i128, st.cfg.snapshot_min) * 60 * std.time.ns_per_s;
    if (nowNs(st.io) - st.last_snap < step) return;
    if (st.snap_busy.load(.acquire) != 0) return;
    lock(st);
    defer unlock(st);
    if (st.closing.load(.acquire) != 0) return;
    if (nowNs(st.io) - st.last_snap < step) return;
    if (st.snap_busy.cmpxchgStrong(0, 1, .acq_rel, .acquire) != null) return;
    pullIntoAcc(st) catch |e| {
        st.snap_busy.store(0, .release);
        std.log.err("reap: snapshot copy failed ({s})", .{@errorName(e)});
        return;
    };
    st.snap_acc.copyFrom(&st.acc);
    st.domains.cloneInto(&st.snap_domains);
    st.refresh.published();
    st.last_snap = nowNs(st.io);
    st.snap_go.store(1, .release);
}

fn writerMain(st: *State) void {
    while (true) {
        const go = st.snap_go.load(.acquire) != 0;
        if (go) {
            reap.snapshotAll(st.gpa, st.io, st.cfg.dir, &st.snap_acc, &st.snap_domains, st.cfg.served_name, st.cfg.build) catch |e| {
                std.log.err("reap: snapshot write failed ({s})", .{@errorName(e)});
            };
            st.snap_go.store(0, .release);
            st.snap_busy.store(0, .release);
        }
        if (st.snap_stop.load(.acquire) != 0) break;
        if (!go) std.Io.sleep(st.io, .fromMilliseconds(20), .awake) catch {};
    }
}

fn packBase(st: *const State, bank: usize) u64 {
    const span = @as(u64, reap.moe_layers) * st.max_sample * reap.topk * st.dn * 4;
    return st.pack_buf.ptr + span * @as(u64, @intCast(bank));
}

fn deviceTotal(st: *const State) u64 {
    var n: u64 = 0;
    const bufs = [_]*const gpu.DeviceBuffer{
        &st.y,          &st.pick_dec,   &st.wts_dec,    &st.pick_pr, &st.wts_pr,
        &st.pack_buf,   &st.recv,       &st.ss0,        &st.ss1,     &st.rows_idx[0],
        &st.rows_idx[1], &st.bucket_pr, &st.pos_pr,     &st.commit_pr, &st.bucket_dec,
        &st.pos_dec,    &st.commit_dec, &st.d_token,    &st.d_gate,  &st.d_early,
        &st.d_raw,      &st.d_ncount,   &st.d_max,
    };
    for (bufs) |b| n += b.len;
    return n;
}

// The pack struct's field order is the kernel's. CUDA never loads the HIP param tables.
comptime {
    if (gpu.backend == .hip) _ = hip_load.params.ReapPack;
    _ = coll.Phase.reap_pair;
    if (round.max_rows != reap.rows_cap_y) @compileError("reap y rows must match round.max_rows");
}
