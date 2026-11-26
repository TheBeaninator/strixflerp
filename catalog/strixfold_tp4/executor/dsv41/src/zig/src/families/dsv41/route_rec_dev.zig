//! Rank-0 device side of per-request routing records (pod card 28153e18f3; docs/experiments/route-records.md).
//! Read-only: the record kernel reads the router's pick/wts after they are written and adds into a per-stream grid.
//! Every call runs on the engine thread, so there is no lock. Request end copies the stream's grid to a host buffer
//! and hands it to route_rec.Writer's thread; no JSON, top-K or file I/O here.

const std = @import("std");
const gpu = @import("gpu");
const prompt = @import("prompt.zig");
const reap = @import("reap.zig");
const rr = @import("route_rec.zig");

const Allocator = std.mem.Allocator;

pub const PicksFn = *const fn (*anyopaque, u8, u8, u32) reap.PickPair;

/// Another owner's pick/wts staging (REAP's, when REAP_STATS is also on). Null: this state owns one.
pub const Shared = struct { f: PicksFn, ctx: *anyopaque };

pub const State = struct {
    gpa: Allocator,
    io: std.Io,
    eng: *const prompt.Engine,
    cfg: rr.Config,
    kernel: gpu.Function,
    route_slots: u32,
    moe: u32,
    world: u32,
    picks: PicksFn,
    picks_ctx: *anyopaque,
    own: bool,
    pick_dec: gpu.DeviceBuffer,
    wts_dec: gpu.DeviceBuffer,
    pick_pr: gpu.DeviceBuffer,
    wts_pr: gpu.DeviceBuffer,
    d_mass: gpu.DeviceBuffer,
    d_count: gpu.DeviceBuffer,
    commit_dec: gpu.DeviceBuffer,
    slot_dec: gpu.DeviceBuffer,
    commit_pr: gpu.DeviceBuffer,
    slot_pr: gpu.DeviceBuffer,
    live: [rr.max_streams]bool = @splat(false),
    meta: [rr.max_streams]rr.Meta = undefined,
    prompt_slot: u8 = rr.no_slot,
    lane: rr.Lane,
    writer: rr.Writer = undefined,
    writer_on: bool = false,
};

fn emptyBuf(d: *const gpu.Driver) gpu.DeviceBuffer {
    return .{ .d = d, .ptr = 0, .len = 0 };
}

fn dropBuf(b: *gpu.DeviceBuffer) void {
    if (b.ptr == 0 and b.len == 0) return;
    b.free();
    b.* = .{ .d = b.d, .ptr = 0, .len = 0 };
}

fn nowMs(io: std.Io) i64 {
    return @intCast(@divTrunc(std.Io.Clock.real.now(io).toNanoseconds(), std.time.ns_per_ms));
}

fn gridOf(n: u64, block: u32) u32 {
    if (n == 0) return 1;
    const g = (n + block - 1) / block;
    return if (g > 65535) 65535 else @intCast(g);
}

/// The pick/wts staging layout, identical to reap_dev.picksOf (the host test compares them offset for offset):
/// decode [40, rows_cap_decode, slots], prompt [2 banks, 40, rows_cap_prompt, slots], i32 picks and f32 weights.
pub fn pickOffset(which: u8, bank: u8, li: u32, slots: u32) u64 {
    if (which == 0) return @as(u64, li) * reap.rows_cap_decode * slots * 4;
    const b: u64 = if (bank == 0) 0 else 1;
    return (b * reap.moe_layers + li) * reap.rows_cap_prompt * slots * 4;
}

/// prompt.Engine.reap_picks when records own the staging (REAP off).
pub fn picksOf(ctx: *anyopaque, which: u8, bank: u8, li: u32) reap.PickPair {
    const st: *State = @ptrCast(@alignCast(ctx));
    if (li >= reap.moe_layers) return .{};
    const off = pickOffset(which, bank, li, st.route_slots);
    if (which == 0) return .{ .pick = st.pick_dec.ptr + off, .wts = st.wts_dec.ptr + off };
    return .{ .pick = st.pick_pr.ptr + off, .wts = st.wts_pr.ptr + off };
}

/// Device bytes this state allocates (own: it holds the staging too).
pub fn deviceBytes(own: bool, route_slots: u32) u64 {
    var n: u64 = rr.grid_mass_bytes + rr.grid_count_bytes;
    n += 2 * @as(u64, reap.rows_cap_decode) + 2 * @as(u64, reap.rows_cap_prompt);
    if (own) {
        n += 2 * @as(u64, reap.moe_layers) * reap.rows_cap_decode * route_slots * 4;
        n += 2 * 2 * @as(u64, reap.moe_layers) * reap.rows_cap_prompt * route_slots * 4;
    }
    return n;
}

/// Rank 0 only. Refuses an empty ROUTE_RECORDS_DIR.
pub fn open(gpa: Allocator, io: std.Io, d: *const gpu.Driver, e: *const prompt.Engine, kernel: gpu.Function, shared: ?Shared) !*State {
    var cfg = try rr.fromEnv(gpa);
    errdefer cfg.deinit(gpa);
    if (cfg.dir.len == 0) {
        std.log.err("route records: ROUTE_RECORDS=1 needs ROUTE_RECORDS_DIR (TF_DS_ROUTE_RECORDS_DIR); refusing to open", .{});
        return error.RouteRecordsDir;
    }
    var lane = try rr.Lane.init(gpa, rr.pool_default, rr.queue_default);
    errdefer lane.deinit();
    const st = try gpa.create(State);
    errdefer gpa.destroy(st);
    st.* = .{
        .gpa = gpa,
        .io = io,
        .eng = e,
        .cfg = cfg,
        .kernel = kernel,
        .route_slots = @intCast(e.slots()),
        .moe = @intCast(@min(e.w.layers.len, reap.moe_layers)),
        .world = e.comm.world,
        .picks = if (shared) |s| s.f else picksOf,
        .picks_ctx = if (shared) |s| s.ctx else undefined,
        .own = shared == null,
        .pick_dec = emptyBuf(d),
        .wts_dec = emptyBuf(d),
        .pick_pr = emptyBuf(d),
        .wts_pr = emptyBuf(d),
        .d_mass = emptyBuf(d),
        .d_count = emptyBuf(d),
        .commit_dec = emptyBuf(d),
        .slot_dec = emptyBuf(d),
        .commit_pr = emptyBuf(d),
        .slot_pr = emptyBuf(d),
        .lane = lane,
    };
    if (shared == null) st.picks_ctx = st;
    errdefer freeDevices(st);
    if (st.own) {
        const dec = @as(usize, reap.moe_layers) * reap.rows_cap_decode * st.route_slots * 4;
        const pr = 2 * @as(usize, reap.moe_layers) * reap.rows_cap_prompt * st.route_slots * 4;
        st.pick_dec = try gpu.DeviceBuffer.alloc(d, dec);
        st.wts_dec = try gpu.DeviceBuffer.alloc(d, dec);
        st.pick_pr = try gpu.DeviceBuffer.alloc(d, pr);
        st.wts_pr = try gpu.DeviceBuffer.alloc(d, pr);
    }
    st.d_mass = try gpu.DeviceBuffer.alloc(d, rr.grid_mass_bytes);
    st.d_count = try gpu.DeviceBuffer.alloc(d, rr.grid_count_bytes);
    st.commit_dec = try gpu.DeviceBuffer.alloc(d, reap.rows_cap_decode);
    st.slot_dec = try gpu.DeviceBuffer.alloc(d, reap.rows_cap_decode);
    st.commit_pr = try gpu.DeviceBuffer.alloc(d, reap.rows_cap_prompt);
    st.slot_pr = try gpu.DeviceBuffer.alloc(d, reap.rows_cap_prompt);
    try gpu.fill8Async(d, e.s.handle, st.d_mass.ptr, 0, rr.grid_mass_bytes);
    try gpu.fill8Async(d, e.s.handle, st.d_count.ptr, 0, rr.grid_count_bytes);
    var ones: [reap.rows_cap_prompt]u8 = @splat(1);
    try prompt.upload(e, st.commit_pr.ptr, &ones, ones.len);
    try e.s.synchronize();
    st.writer = try rr.Writer.init(gpa, io, &st.lane, cfg.dir, cfg.maxBytes(), cfg.topk);
    st.writer_on = true;
    errdefer {
        st.writer.close();
        st.writer_on = false;
    }
    try st.writer.spawn();
    std.log.info("route records: rank 0 device bytes {d} (staging {s}), dir {s}, topk {d}, max {d} MB", .{ deviceBytes(st.own, st.route_slots), if (st.own) "own" else "shared with REAP", cfg.dir, cfg.topk, cfg.max_mb });
    return st;
}

fn freeDevices(st: *State) void {
    dropBuf(&st.pick_dec);
    dropBuf(&st.wts_dec);
    dropBuf(&st.pick_pr);
    dropBuf(&st.wts_pr);
    dropBuf(&st.d_mass);
    dropBuf(&st.d_count);
    dropBuf(&st.commit_dec);
    dropBuf(&st.slot_dec);
    dropBuf(&st.commit_pr);
    dropBuf(&st.slot_pr);
}

/// Live records are written with finish "closed"; the writer drains its queue, then joins.
pub fn close(st: *State) void {
    var s: usize = 0;
    while (s < rr.max_streams) : (s += 1) if (st.live[s]) end(st, @intCast(s), "closed", 0);
    if (st.writer_on) {
        st.writer.close();
        st.writer_on = false;
    }
    const drops = st.lane.drops();
    if (drops > 0) std.log.warn("route records: {d} record(s) dropped (pool or queue full)", .{drops});
    freeDevices(st);
    st.lane.deinit();
    st.cfg.deinit(st.gpa);
    st.gpa.destroy(st);
}

/// A stream starts its prompt: its record (if the server gave it an id) becomes live, and the prompt chunks
/// that follow add into its slot.
pub fn begin(st: *State, slot: usize, id: []const u8, tag: []const u8, request: []const u8) void {
    st.prompt_slot = rr.no_slot;
    if (slot >= rr.max_streams) return;
    if (st.live[slot]) {
        // the same record filling again keeps adding; another record still live here ends first (its grid is zeroed)
        if (std.mem.eql(u8, st.meta[slot].id[0..st.meta[slot].id_len], id[0..@min(id.len, st.meta[slot].id.len)])) {
            st.prompt_slot = @intCast(slot);
            return;
        }
        end(st, slot, "superseded", 0);
    }
    if (id.len == 0) return;
    var m: rr.Meta = .{};
    m.setId(id);
    m.setRequest(request);
    m.setTag(tag);
    m.setModel(st.cfg.served_name);
    m.setBuild(st.cfg.build);
    m.world = st.world;
    m.t_start = nowMs(st.io);
    m.topk = st.cfg.topk;
    st.meta[slot] = m;
    st.live[slot] = true;
    st.prompt_slot = @intCast(slot);
}

fn launchRec(st: *State, pick: u64, wts: u64, commit: u64, slotmap: u64, layers: u32, rows: u32, row_stride: u32) !void {
    var a: gpu.Args = .{};
    a.add(pick);
    a.add(wts);
    a.add(commit);
    a.add(slotmap);
    a.add(st.d_mass.ptr);
    a.add(st.d_count.ptr);
    a.add(layers);
    a.add(rows);
    a.add(row_stride);
    a.add(st.route_slots);
    a.add(reap.topk);
    a.add(reap.n_experts);
    a.add(rr.max_streams);
    a.add(rr.grid_layers);
    const n = @as(u64, layers) * reap.n_experts;
    try gpu.launch.launch(st.kernel, .{ .grid = .{ .x = gridOf(n, 128) }, .block = .{ .x = 128 } }, st.eng.s, &a);
}

/// After a prompt chunk's layers (layers_ran: 20 for an encoder-only chunk, else every MoE layer).
pub fn finishPrompt(st: *State, ch: *const prompt.Chunk, layers_ran: u32) !void {
    const slot = st.prompt_slot;
    if (slot == rr.no_slot or !st.live[slot]) return;
    const layers = @min(layers_ran, st.moe);
    const n: u32 = @intCast(@min(ch.n, reap.rows_cap_prompt));
    if (n == 0 or layers == 0) return;
    var map: [reap.rows_cap_prompt]u8 = undefined;
    @memset(map[0..n], slot);
    try prompt.upload(st.eng, st.slot_pr.ptr, &map, n);
    const base = st.picks(st.picks_ctx, 1, ch.reap_bank, 0);
    if (base.pick == 0) return;
    try launchRec(st, base.pick, base.wts, st.commit_pr.ptr, st.slot_pr.ptr, layers, n, reap.rows_cap_prompt);
    // rows count once, at the launch that runs every MoE layer (an encoder-only cut adds mass, not rows)
    if (layers >= st.moe) {
        var vision: u32 = 0;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const p = ch.start + i;
            for (ch.spans) |sp| if (p >= sp.at and p < sp.at + sp.len) {
                vision += 1;
                break;
            };
        }
        st.meta[slot].rows.prompt += n - vision;
        st.meta[slot].rows.vision += vision;
    }
}

/// After a decode round's draws. commit[r]: row r is kept. slot[r]: row r's stream slot.
pub fn noteDecode(st: *State, commit: []const u8, slot: []const u8) !void {
    const n: u32 = @intCast(@min(@min(commit.len, slot.len), reap.rows_cap_decode));
    if (n == 0) return;
    var map: [reap.rows_cap_decode]u8 = undefined;
    var any = false;
    for (0..n) |r| {
        const s = slot[r];
        if (s < rr.max_streams and st.live[s]) {
            map[r] = s;
            if (commit[r] != 0) {
                st.meta[s].rows.decode += 1;
                any = true;
            }
        } else map[r] = rr.no_slot;
    }
    if (!any) return;
    try prompt.upload(st.eng, st.commit_dec.ptr, @ptrCast(commit.ptr), n);
    try prompt.upload(st.eng, st.slot_dec.ptr, &map, n);
    const base = st.picks(st.picks_ctx, 0, 0, 0);
    if (base.pick == 0) return;
    try launchRec(st, base.pick, base.wts, st.commit_dec.ptr, st.slot_dec.ptr, st.moe, n, reap.rows_cap_decode);
}

/// The stream ended (any reason). Copy its grid to a pool buffer (one stream sync, 184,320 B), zero the device
/// slice before the slot can be reused, and queue the record for the writer. Pool or queue full: drop and count.
pub fn end(st: *State, slot: usize, finish: []const u8, cached: u32) void {
    if (slot >= rr.max_streams or !st.live[slot]) return;
    st.live[slot] = false;
    if (st.prompt_slot == slot) st.prompt_slot = rr.no_slot;
    var m = st.meta[slot];
    m.t_end = nowMs(st.io);
    m.setFinish(finish);
    m.rows.prompt_cached = cached;
    const mass_off = slot * rr.slot_mass_bytes;
    const count_off = slot * rr.slot_count_bytes;
    const d = st.eng.d;
    const bi = st.lane.take();
    if (bi) |b| {
        const ok = blk: {
            st.eng.s.synchronize() catch break :blk false;
            gpu.copyD2H(d, std.mem.sliceAsBytes(st.lane.massOf(b)), st.d_mass.ptr + mass_off) catch break :blk false;
            gpu.copyD2H(d, std.mem.sliceAsBytes(st.lane.countOf(b)), st.d_count.ptr + count_off) catch break :blk false;
            break :blk true;
        };
        if (!ok or !st.lane.give(b, m)) {
            st.lane.release(b);
            noteDrop(st);
        }
    } else noteDrop(st);
    gpu.fill8Async(d, st.eng.s.handle, st.d_mass.ptr + mass_off, 0, rr.slot_mass_bytes) catch |err| std.log.err("route records: zeroing slot {d} failed ({s})", .{ slot, @errorName(err) });
    gpu.fill8Async(d, st.eng.s.handle, st.d_count.ptr + count_off, 0, rr.slot_count_bytes) catch |err| std.log.err("route records: zeroing slot {d} failed ({s})", .{ slot, @errorName(err) });
}

fn noteDrop(st: *State) void {
    st.lane.addDrop();
    const n = st.lane.drops();
    if (n == 1 or n % 64 == 0) std.log.warn("route records: dropped a record (pool or queue full; {d} so far)", .{n});
}

test "records-only staging has REAP's pick/wts layout, offset for offset" {
    // reap_dev.picksOf: decode li * rows_cap_decode * slots * 4; prompt (bank * 40 + li) * rows_cap_prompt * slots * 4
    const slots: u32 = 7;
    var li: u32 = 0;
    while (li < reap.moe_layers) : (li += 1) {
        try std.testing.expectEqual(@as(u64, li) * reap.rows_cap_decode * slots * 4, pickOffset(0, 0, li, slots));
        try std.testing.expectEqual(@as(u64, li) * reap.rows_cap_prompt * slots * 4, pickOffset(1, 0, li, slots));
        try std.testing.expectEqual((@as(u64, reap.moe_layers) + li) * reap.rows_cap_prompt * slots * 4, pickOffset(1, 1, li, slots));
    }
    try std.testing.expectEqual(@as(u32, 64), reap.rows_cap_decode);
    try std.testing.expectEqual(@as(u32, 2048), reap.rows_cap_prompt);
    try std.testing.expectEqual(rr.rows_cap_decode, reap.rows_cap_decode);
    try std.testing.expectEqual(rr.rows_cap_prompt, reap.rows_cap_prompt);
    try std.testing.expectEqual(rr.moe_layers, reap.moe_layers);
    try std.testing.expectEqual(rr.n_experts, reap.n_experts);
    try std.testing.expectEqual(rr.topk, reap.topk);
}

test "device bytes: shared staging adds only the grids and maps" {
    try std.testing.expectEqual(@as(u64, 2_949_120 + 4_224), deviceBytes(false, 7));
    try std.testing.expectEqual(@as(u64, 2_949_120 + 4_224 + 143_360 + 9_175_040), deviceBytes(true, 7));
}
