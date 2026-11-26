//! Per-request routing records, host side. GPU-free: the device kernel is
//! route_rec.hip, launched from route_rec_dev.zig. One writer per (layer, expert), so the same
//! traffic yields the same bits. The writer thread owns the files. The engine thread only copies.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const moe_layers: u32 = 40;
pub const n_experts: u32 = 384;
pub const topk: u32 = 6;
pub const route_slots: u32 = 7;
pub const rows_cap_decode: u32 = 64;
pub const rows_cap_prompt: u32 = 2048;
pub const grid_layers: u32 = 40;
/// draft.max_streams. A slot byte is 0..15; 0xFF is "no record".
pub const max_streams: u32 = 16;
pub const default_topk: u32 = 16;
pub const default_max_mb: u32 = 2048;
pub const pool_default: u32 = 32;
pub const queue_default: u32 = 256;
/// build.zig.zon `.version`, used when TF_DS_DEPLOY is unset. The same string Edge-REAP uses.
pub const zon_version = "1.0.2";
pub const kernel_symbol = "tf_ds_route_rec";
pub const no_slot: u8 = 0xff;

pub const cells: usize = grid_layers * n_experts;
pub const slot_mass_bytes: usize = cells * 8;
pub const slot_count_bytes: usize = cells * 4;
pub const grid_mass_bytes: usize = max_streams * slot_mass_bytes;
pub const grid_count_bytes: usize = max_streams * slot_count_bytes;

/// One extra launch on rank 0 when records are on. Every other rank, and records off: none.
pub fn extraLaunches(on: bool, rank: u32) u32 {
    return if (on and rank == 0) 1 else 0;
}

pub const Config = struct {
    on: bool = false,
    topk: u32 = default_topk,
    max_mb: u32 = default_max_mb,
    dir: []const u8 = "",
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

    pub fn maxBytes(c: Config) u64 {
        return @as(u64, c.max_mb) * 1024 * 1024;
    }

    /// Rotate past the larger of 1 MiB and a sixteenth of the cap.
    pub fn rotateAt(c: Config) u64 {
        return @max(@as(u64, 1 << 20), c.maxBytes() / 16);
    }
};

pub const EnvIn = struct {
    on: ?[]const u8 = null,
    topk: ?[]const u8 = null,
    max_mb: ?[]const u8 = null,
    dir: ?[]const u8 = null,
    served_name: ?[]const u8 = null,
    deploy: ?[]const u8 = null,
};

fn parseU32(s: ?[]const u8) ?u32 {
    const t = s orelse return null;
    if (t.len == 0) return null;
    return std.fmt.parseInt(u32, t, 10) catch null;
}

/// On only for the exact text "1". Other keys apply only then. topk 0 falls back to 16 and
/// above 384 clamps to 384. max_mb 0 falls back to 2048. An empty dir is kept empty: rank 0
/// refuses it at open.
pub fn configFrom(a: Allocator, in: EnvIn) !Config {
    const on = if (in.on) |s| std.mem.eql(u8, s, "1") else false;
    if (!on) return .{};
    var c: Config = .{ .on = true };
    errdefer c.deinit(a);
    c.topk = parseU32(in.topk) orelse default_topk;
    if (c.topk == 0) c.topk = default_topk;
    if (c.topk > n_experts) c.topk = n_experts;
    c.max_mb = parseU32(in.max_mb) orelse default_max_mb;
    if (c.max_mb == 0) c.max_mb = default_max_mb;
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
        .on = envSlice("TF_DS_ROUTE_RECORDS"),
        .topk = envSlice("TF_DS_ROUTE_RECORDS_TOPK"),
        .max_mb = envSlice("TF_DS_ROUTE_RECORDS_MAX_MB"),
        .dir = envSlice("TF_DS_ROUTE_RECORDS_DIR"),
        .served_name = envSlice("TF_DS_SERVED_NAME"),
        .deploy = envSlice("TF_DS_DEPLOY"),
    });
}

pub fn recordsOn() bool {
    const v = std.c.getenv("TF_DS_ROUTE_RECORDS") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// Leading space included, so planText can append it. Empty when records are off (the stock plan).
pub fn planSuffix(buf: []u8, c: Config) []const u8 {
    if (!c.on) return "";
    return std.fmt.bufPrint(buf, " rrec={d}", .{c.topk}) catch "";
}

// --- index formulas (route_rec.hip uses these) --------------------------------

/// pick/wts element: ((layer * row_stride + row) * slots) + k
pub fn pickIndex(layer: u32, row_stride: u32, row: u32, slots: u32, k: u32) usize {
    return ((@as(usize, layer) * row_stride + row) * slots) + k;
}

/// Accumulator cell: (slot * grid_layers + layer) * nexp + expert
pub fn accIndex(slot: u32, layers: u32, layer: u32, expert: u32, nexp: u32) usize {
    return ((@as(usize, slot) * layers + layer) * nexp) + expert;
}

pub const Launch = extern struct {
    pick: u64 = 0,
    wts: u64 = 0,
    commit: u64 = 0,
    slotmap: u64 = 0,
    mass: u64 = 0,
    count: u64 = 0,
    layers: u32 = 0,
    rows: u32 = 0,
    row_stride: u32 = 0,
    slots: u32 = 0,
    topk: u32 = 0,
    nexp: u32 = 0,
    nslots: u32 = 0,
    grid_layers: u32 = 0,
};

pub const View = struct {
    pick: []const i32,
    wts: []const f32,
    commit: []const u8,
    slotmap: []const u8,
    layers: u32,
    rows: u32,
    row_stride: u32,
    slots: u32,
    top_k: u32,
    nexp: u32,
    nslots: u32,
    grid_layers: u32,
};

/// The kernel's loop: one (layer, expert) at a time, rows in order, then k. Adds are f64.
pub fn accumulate(mass: []f64, count: []u32, v: View) void {
    var layer: u32 = 0;
    while (layer < v.layers) : (layer += 1) {
        var expert: u32 = 0;
        while (expert < v.nexp) : (expert += 1) {
            var row: u32 = 0;
            while (row < v.rows) : (row += 1) {
                if (row >= v.commit.len or v.commit[row] == 0) continue;
                const slot = v.slotmap[row];
                if (slot == no_slot or slot >= v.nslots) continue;
                const base = pickIndex(layer, v.row_stride, row, v.slots, 0);
                var k: u32 = 0;
                while (k < v.top_k) : (k += 1) {
                    const id = v.pick[base + k];
                    if (id < 0) continue;
                    const eid: u32 = @intCast(id);
                    if (eid >= v.nexp or eid != expert) continue;
                    const at = accIndex(slot, v.grid_layers, layer, expert, v.nexp);
                    mass[at] += v.wts[base + k];
                    count[at] += 1;
                }
            }
        }
    }
}

/// Same adds per cell, in the same row order and k order, walked by row instead of by cell.
pub fn accumulateIndep(mass: []f64, count: []u32, v: View) void {
    var row: u32 = 0;
    while (row < v.rows) : (row += 1) {
        if (row >= v.commit.len or v.commit[row] == 0) continue;
        const slot = v.slotmap[row];
        if (slot == no_slot or slot >= v.nslots) continue;
        var layer: u32 = 0;
        while (layer < v.layers) : (layer += 1) {
            const base = pickIndex(layer, v.row_stride, row, v.slots, 0);
            var k: u32 = 0;
            while (k < v.top_k) : (k += 1) {
                const id = v.pick[base + k];
                if (id < 0) continue;
                const eid: u32 = @intCast(id);
                if (eid >= v.nexp) continue;
                const at = accIndex(slot, v.grid_layers, layer, eid, v.nexp);
                mass[at] += v.wts[base + k];
                count[at] += 1;
            }
        }
    }
}

/// Per-row slot byte. A slot that is not live, or out of range, is 0xFF.
pub fn fillSlotMap(out: []u8, slots: []const i64, live: []const bool) void {
    for (slots, 0..) |s, i| {
        if (s < 0 or s >= @as(i64, @intCast(live.len)) or !live[@intCast(s)]) out[i] = no_slot else out[i] = @intCast(s);
    }
}

pub const Span = struct { at: usize, len: usize };

pub fn inSpan(pos: usize, spans: []const Span) bool {
    for (spans) |sp| if (pos >= sp.at and pos < sp.at + sp.len) return true;
    return false;
}

/// Routed prompt rows, split by whether the position sits in an image span. Not overlapping.
pub fn splitPrompt(start: usize, n: usize, spans: []const Span) struct { prompt: u32, vision: u32 } {
    var vision: u32 = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) if (inSpan(start + i, spans)) {
        vision += 1;
    };
    return .{ .prompt = @intCast(n - vision), .vision = vision };
}

pub const Rows = struct {
    prompt: u32 = 0,
    prompt_cached: u32 = 0,
    decode: u32 = 0,
    vision: u32 = 0,
};

pub const Top = struct { expert: u32, mass: f64, count: u32 };

fn beats(am: f64, ai: u32, bm: f64, bi: u32) bool {
    if (am > bm) return true;
    if (am < bm) return false;
    return ai < bi;
}

/// Exactly `k` experts, highest gate mass first. A tie keeps the lower expert id. Zeros stay,
/// so an all-zero layer yields experts 0..k-1.
pub fn selectTop(mass: []const f64, count: []const u32, k: u32, out: []Top) u32 {
    const kk: u32 = @min(k, @as(u32, @intCast(@min(mass.len, out.len))));
    if (kk == 0) return 0;
    var n: u32 = 0;
    var e: u32 = 0;
    while (e < mass.len) : (e += 1) {
        const m = mass[e];
        const c = if (e < count.len) count[e] else 0;
        if (n < kk) {
            var i: u32 = n;
            while (i > 0 and beats(m, e, out[i - 1].mass, out[i - 1].expert)) : (i -= 1) out[i] = out[i - 1];
            out[i] = .{ .expert = e, .mass = m, .count = c };
            n += 1;
        } else if (beats(m, e, out[kk - 1].mass, out[kk - 1].expert)) {
            var i: u32 = kk - 1;
            while (i > 0 and beats(m, e, out[i - 1].mass, out[i - 1].expert)) : (i -= 1) out[i] = out[i - 1];
            out[i] = .{ .expert = e, .mass = m, .count = c };
        }
    }
    return n;
}

/// Totals over every expert of one layer, summed in expert-id order.
pub fn layerTotal(mass: []const f64, count: []const u32) struct { mass: f64, count: u64 } {
    var m: f64 = 0;
    var c: u64 = 0;
    for (mass, 0..) |g, i| {
        m += g;
        c += if (i < count.len) count[i] else 0;
    }
    return .{ .mass = m, .count = c };
}

const IoWriter = std.Io.Writer;

fn floatJson(buf: []u8, f: f64) []const u8 {
    if (std.math.isNan(f)) return "NaN";
    if (std.math.isInf(f)) return if (f > 0) "Infinity" else "-Infinity";
    return floatRepr(buf, f);
}

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

fn writeJsonString(w: *IoWriter, s: []const u8) IoWriter.Error!void {
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

fn writeFloat(w: *IoWriter, f: f64) IoWriter.Error!void {
    var buf: [64]u8 = undefined;
    try w.writeAll(floatJson(&buf, f));
}

pub const Meta = struct {
    id: [64]u8 = undefined,
    id_len: u8 = 0,
    request_id: [96]u8 = undefined,
    req_len: u8 = 0,
    tag: [160]u8 = undefined,
    tag_len: u8 = 0,
    has_tag: bool = false,
    model: [64]u8 = undefined,
    model_len: u8 = 0,
    build: [64]u8 = undefined,
    build_len: u8 = 0,
    world: u32 = 0,
    t_start: i64 = 0,
    t_end: i64 = 0,
    finish: [16]u8 = undefined,
    finish_len: u8 = 0,
    rows: Rows = .{},
    topk: u32 = default_topk,

    pub fn setId(m: *Meta, s: []const u8) void {
        m.id_len = copyCap(&m.id, s);
    }
    pub fn setRequest(m: *Meta, s: []const u8) void {
        m.req_len = copyCap(&m.request_id, s);
    }
    pub fn setTag(m: *Meta, s: []const u8) void {
        if (s.len == 0) {
            m.has_tag = false;
            m.tag_len = 0;
            return;
        }
        m.tag_len = copyCap(&m.tag, s);
        m.has_tag = m.tag_len > 0;
    }
    pub fn setModel(m: *Meta, s: []const u8) void {
        m.model_len = copyCap(&m.model, s);
    }
    pub fn setBuild(m: *Meta, s: []const u8) void {
        m.build_len = copyCap(&m.build, s);
    }
    pub fn setFinish(m: *Meta, s: []const u8) void {
        m.finish_len = copyCap(&m.finish, s);
    }
};

fn copyCap(dest: []u8, src: []const u8) u8 {
    const n: u8 = @intCast(@min(src.len, dest.len));
    @memcpy(dest[0..n], src[0..n]);
    return n;
}

/// One JSON object, keys in the schema order. No trailing newline (the writer adds it).
pub fn writeRecord(w: *IoWriter, meta: *const Meta, mass: []const f64, count: []const u32) IoWriter.Error!void {
    const k = @min(meta.topk, n_experts);
    try w.writeAll("{\"v\":1,\"id\":");
    try writeJsonString(w, meta.id[0..meta.id_len]);
    try w.writeAll(",\"request_id\":");
    try writeJsonString(w, meta.request_id[0..meta.req_len]);
    try w.writeAll(",\"route_tag\":");
    if (!meta.has_tag) try w.writeAll("null") else try writeJsonString(w, meta.tag[0..meta.tag_len]);
    try w.writeAll(",\"model\":");
    try writeJsonString(w, meta.model[0..meta.model_len]);
    try w.writeAll(",\"build\":");
    try writeJsonString(w, meta.build[0..meta.build_len]);
    try w.print(",\"world\":{d},\"t_start\":{d},\"t_end\":{d},\"finish\":", .{ meta.world, meta.t_start, meta.t_end });
    try writeJsonString(w, meta.finish[0..meta.finish_len]);
    try w.print(",\"rows\":{{\"prompt\":{d},\"prompt_cached\":{d},\"decode\":{d},\"vision\":{d}}},\"topk\":{d},\"layers\":[", .{ meta.rows.prompt, meta.rows.prompt_cached, meta.rows.decode, meta.rows.vision, k });
    var top_buf: [n_experts]Top = undefined;
    var layer: u32 = 0;
    while (layer < grid_layers) : (layer += 1) {
        const at = @as(usize, layer) * n_experts;
        const lm = if (at + n_experts <= mass.len) mass[at..][0..n_experts] else mass[at..];
        const lc = if (at + n_experts <= count.len) count[at..][0..n_experts] else count[at..];
        const tot = layerTotal(lm, lc);
        const ntop = selectTop(lm, lc, k, &top_buf);
        if (layer > 0) try w.writeByte(',');
        try w.print("{{\"l\":{d},\"mass\":", .{layer});
        try writeFloat(w, tot.mass);
        try w.print(",\"count\":{d},\"top\":[", .{tot.count});
        var t: u32 = 0;
        while (t < ntop) : (t += 1) {
            if (t > 0) try w.writeByte(',');
            try w.print("[{d},", .{top_buf[t].expert});
            try writeFloat(w, top_buf[t].mass);
            try w.print(",{d}]", .{top_buf[t].count});
        }
        try w.writeAll("]}");
    }
    try w.writeAll("]}");
}

// --- files ---------------------------------------------------------------------

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

pub const Civil = struct { y: i32, m: u32, d: u32, hh: u32, mm: u32, ss: u32 };

/// Days since 1970-01-01, Howard Hinnant's civil_from_days. Seconds are UTC.
pub fn civilFromUnix(sec_in: i64) Civil {
    const sec: i64 = if (sec_in < 0) 0 else sec_in;
    const days = @divFloor(sec, 86400);
    const sod: u32 = @intCast(@mod(sec, 86400));
    const z = days + 719468;
    const era = if (z >= 0) @divFloor(z, 146097) else @divFloor(z - 146096, 146097);
    const doe: u32 = @intCast(z - era * 146097);
    const yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    var y: i32 = @as(i32, @intCast(yoe)) + @as(i32, @intCast(era)) * 400;
    const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const mp = (5 * doy + 2) / 153;
    const d = doy - (153 * mp + 2) / 5 + 1;
    const m: u32 = if (mp < 10) mp + 3 else mp - 9;
    if (m <= 2) y += 1;
    return .{ .y = y, .m = m, .d = d, .hh = sod / 3600, .mm = (sod % 3600) / 60, .ss = sod % 60 };
}

pub fn fileName(buf: []u8, unix_sec: i64, seq: u32) []const u8 {
    const c = civilFromUnix(unix_sec);
    const year: u32 = @intCast(c.y);
    return std.fmt.bufPrint(buf, "route-{d:0>4}{d:0>2}{d:0>2}T{d:0>2}{d:0>2}{d:0>2}Z-{d:0>6}.jsonl", .{ year, c.m, c.d, c.hh, c.mm, c.ss, seq }) catch buf[0..0];
}

/// One level: a RAM home emptied by a reboot still serves (the parent, the deploy's home, exists). An existing
/// directory is fine; any other failure shows at the first file open.
fn ensureDir(gpa: Allocator, dir: []const u8) !void {
    const z = try gpa.dupeSentinel(u8, dir, 0);
    defer gpa.free(z);
    _ = std.c.mkdir(z, 0o755);
}

fn isRouteLog(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "route-") and std.mem.endsWith(u8, name, ".jsonl");
}

pub const Files = struct {
    io: std.Io,
    gpa: Allocator,
    dir: []const u8,
    max_bytes: u64,
    rotate_at: u64,
    seq: u32 = 0,
    fd: std.posix.fd_t = -1,
    name: [96]u8 = undefined,
    name_len: usize = 0,
    size: u64 = 0,
    /// Clock override for tests. Null: the real clock.
    now_sec: ?i64 = null,

    pub fn open(io: std.Io, gpa: Allocator, dir: []const u8, max_bytes: u64) !Files {
        var f: Files = .{
            .io = io,
            .gpa = gpa,
            .dir = dir,
            .max_bytes = max_bytes,
            .rotate_at = @max(@as(u64, 1 << 20), max_bytes / 16),
        };
        try ensureDir(gpa, dir);
        try f.openNew();
        try f.prune();
        return f;
    }

    pub fn close(f: *Files) void {
        if (f.fd >= 0) {
            fsyncFd(f.fd) catch {};
            _ = std.posix.system.close(f.fd);
            f.fd = -1;
        }
    }

    fn nowSec(f: *Files) i64 {
        return f.now_sec orelse std.Io.Clock.real.now(f.io).toSeconds();
    }

    fn openNew(f: *Files) !void {
        var nbuf: [96]u8 = undefined;
        const name = fileName(&nbuf, f.nowSec(), f.seq);
        f.seq += 1;
        const path = try std.fs.path.join(f.gpa, &.{ f.dir, name });
        defer f.gpa.free(path);
        const z = try f.gpa.dupeSentinel(u8, path, 0);
        defer f.gpa.free(z);
        const fd = try std.posix.openatZ(std.posix.AT.FDCWD, z, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
        f.fd = fd;
        @memcpy(f.name[0..name.len], name);
        f.name_len = name.len;
        f.size = 0;
    }

    fn curName(f: *const Files) []const u8 {
        return f.name[0..f.name_len];
    }

    fn rotate(f: *Files) !void {
        if (f.fd >= 0) {
            try fsyncFd(f.fd);
            _ = std.posix.system.close(f.fd);
            f.fd = -1;
        }
        try f.openNew();
        try f.prune();
    }

    /// One write of `line` (the caller includes the newline). Rotates first when the current
    /// file is already past the threshold.
    pub fn writeLine(f: *Files, line: []const u8) !void {
        if (f.fd < 0) try f.openNew();
        if (f.size >= f.rotate_at) try f.rotate();
        try writeAllFd(f.fd, line);
        f.size += line.len;
    }

    fn fileSize(f: *Files, dir: std.Io.Dir, name: []const u8) u64 {
        var file = dir.openFile(f.io, name, .{}) catch return 0;
        defer file.close(f.io);
        const st = file.stat(f.io) catch return 0;
        return @intCast(st.size);
    }

    /// Delete the oldest route-*.jsonl until the total is within the cap. Never the current
    /// file, never any other name.
    pub fn prune(f: *Files) !void {
        var d = std.Io.Dir.cwd().openDir(f.io, f.dir, .{ .iterate = true }) catch return;
        defer d.close(f.io);
        var names: std.ArrayList([]u8) = .empty;
        defer {
            for (names.items) |nm| f.gpa.free(nm);
            names.deinit(f.gpa);
        }
        var it = d.iterate();
        while (try it.next(f.io)) |e| {
            if (e.kind != .file) continue;
            if (!isRouteLog(e.name)) continue;
            try names.append(f.gpa, try f.gpa.dupe(u8, e.name));
        }
        std.mem.sort([]u8, names.items, {}, struct {
            fn lt(_: void, x: []u8, y: []u8) bool {
                return std.mem.lessThan(u8, x, y);
            }
        }.lt);
        var total: u64 = 0;
        const sizes = try f.gpa.alloc(u64, names.items.len);
        defer f.gpa.free(sizes);
        for (names.items, sizes) |nm, *sz| {
            sz.* = f.fileSize(d, nm);
            total += sz.*;
        }
        var i: usize = 0;
        while (total > f.max_bytes and i < names.items.len) : (i += 1) {
            if (std.mem.eql(u8, names.items[i], f.curName())) continue;
            d.deleteFile(f.io, names.items[i]) catch continue;
            total -= sizes[i];
        }
    }
};

// --- queue ---------------------------------------------------------------------

const SlotBuf = struct {
    mass: []f64,
    count: []u32,
};

const Job = struct {
    meta: Meta,
    bi: u32,
};

pub const Lane = struct {
    gpa: Allocator,
    mu: std.atomic.Mutex = .unlocked,
    bufs: []SlotBuf,
    free_stack: []u32,
    nfree: u32,
    q: []Job,
    qh: u32 = 0,
    qn: u32 = 0,
    dropped: u64 = 0,

    pub fn init(gpa: Allocator, nbuf: u32, nq: u32) !Lane {
        const nb: u32 = if (nbuf == 0) 1 else nbuf;
        const nqueue: u32 = if (nq == 0) 1 else nq;
        var lane: Lane = .{ .gpa = gpa, .bufs = &.{}, .free_stack = &.{}, .nfree = 0, .q = &.{} };
        errdefer lane.deinit();
        lane.bufs = try gpa.alloc(SlotBuf, nb);
        for (lane.bufs) |*b| b.* = .{ .mass = &.{}, .count = &.{} };
        lane.free_stack = try gpa.alloc(u32, nb);
        lane.q = try gpa.alloc(Job, nqueue);
        lane.nfree = nb;
        for (lane.bufs, 0..) |*b, i| {
            b.mass = try gpa.alloc(f64, cells);
            b.count = try gpa.alloc(u32, cells);
            lane.free_stack[i] = @intCast(nb - 1 - @as(u32, @intCast(i)));
        }
        return lane;
    }

    pub fn deinit(l: *Lane) void {
        for (l.bufs) |b| {
            if (b.mass.len > 0) l.gpa.free(b.mass);
            if (b.count.len > 0) l.gpa.free(b.count);
        }
        if (l.bufs.len > 0) l.gpa.free(l.bufs);
        if (l.free_stack.len > 0) l.gpa.free(l.free_stack);
        if (l.q.len > 0) l.gpa.free(l.q);
        l.bufs = &.{};
        l.free_stack = &.{};
        l.q = &.{};
    }

    fn lock(l: *Lane) void {
        while (!l.mu.tryLock()) std.Thread.yield() catch {};
    }

    pub fn take(l: *Lane) ?u32 {
        l.lock();
        defer l.mu.unlock();
        if (l.nfree == 0) return null;
        l.nfree -= 1;
        return l.free_stack[l.nfree];
    }

    pub fn release(l: *Lane, bi: u32) void {
        l.lock();
        defer l.mu.unlock();
        if (l.nfree >= l.free_stack.len) return;
        l.free_stack[l.nfree] = bi;
        l.nfree += 1;
    }

    pub fn massOf(l: *Lane, bi: u32) []f64 {
        return l.bufs[bi].mass;
    }

    pub fn countOf(l: *Lane, bi: u32) []u32 {
        return l.bufs[bi].count;
    }

    /// False when the queue is full. The buffer stays with the caller.
    pub fn give(l: *Lane, bi: u32, meta: Meta) bool {
        l.lock();
        defer l.mu.unlock();
        if (l.qn >= l.q.len) return false;
        const at = (l.qh + l.qn) % @as(u32, @intCast(l.q.len));
        l.q[at] = .{ .meta = meta, .bi = bi };
        l.qn += 1;
        return true;
    }

    pub fn pop(l: *Lane) ?Job {
        l.lock();
        defer l.mu.unlock();
        if (l.qn == 0) return null;
        const job = l.q[l.qh];
        l.qh = (l.qh + 1) % @as(u32, @intCast(l.q.len));
        l.qn -= 1;
        return job;
    }

    pub fn addDrop(l: *Lane) void {
        l.lock();
        defer l.mu.unlock();
        l.dropped += 1;
    }

    pub fn drops(l: *Lane) u64 {
        l.lock();
        defer l.mu.unlock();
        return l.dropped;
    }
};

pub const Writer = struct {
    lane: *Lane,
    files: Files,
    io: std.Io,
    topk: u32,
    stop: std.atomic.Value(u32) = .init(0),
    thread: ?std.Thread = null,

    pub fn init(gpa: Allocator, io: std.Io, lane: *Lane, dir: []const u8, max_bytes: u64, k: u32) !Writer {
        return .{
            .lane = lane,
            .files = try Files.open(io, gpa, dir, max_bytes),
            .io = io,
            .topk = if (k == 0) default_topk else k,
        };
    }

    /// The Writer must already sit at the address the thread will use.
    pub fn spawn(w: *Writer) !void {
        w.thread = try std.Thread.spawn(.{}, writerMain, .{w});
    }

    /// Stop the thread and let it drain the queue, then fsync the current file.
    pub fn close(w: *Writer) void {
        w.stop.store(1, .release);
        if (w.thread) |th| {
            th.join();
            w.thread = null;
        }
        w.files.close();
    }
};

fn writerMain(w: *Writer) void {
    const buf = w.lane.gpa.alloc(u8, 2 * 1024 * 1024) catch return;
    defer w.lane.gpa.free(buf);
    while (true) {
        if (w.lane.pop()) |job| {
            var meta = job.meta;
            if (meta.topk == 0) meta.topk = w.topk;
            var wr = std.Io.Writer.fixed(buf);
            writeRecord(&wr, &meta, w.lane.massOf(job.bi), w.lane.countOf(job.bi)) catch {
                w.lane.release(job.bi);
                continue;
            };
            if (wr.end + 1 <= buf.len) {
                buf[wr.end] = '\n';
                w.files.writeLine(buf[0 .. wr.end + 1]) catch {};
            }
            w.lane.release(job.bi);
            continue;
        }
        if (w.stop.load(.acquire) != 0) break;
        std.Io.sleep(w.io, .fromMilliseconds(2), .awake) catch {};
    }
}

// --- tests ---------------------------------------------------------------------

fn bitEq(a: f64, b: f64) !void {
    try std.testing.expectEqual(@as(u64, @bitCast(a)), @as(u64, @bitCast(b)));
}

const Rng = struct {
    s: u64,
    fn next(r: *Rng) u64 {
        r.s = r.s *% 6364136223846793005 +% 1;
        return r.s;
    }
    fn u32n(r: *Rng, n: u32) u32 {
        return @intCast(r.next() % n);
    }
};

test "config defaults stay off and the plan suffix is empty" {
    const a = std.testing.allocator;
    var c = try configFrom(a, .{});
    defer c.deinit(a);
    try std.testing.expect(!c.on);
    try std.testing.expectEqual(default_topk, c.topk);
    try std.testing.expectEqual(default_max_mb, c.max_mb);
    var buf: [16]u8 = undefined;
    try std.testing.expectEqualStrings("", planSuffix(&buf, c));
    try std.testing.expectEqual(@as(u32, 0), extraLaunches(false, 0));
    try std.testing.expectEqual(@as(u32, 0), extraLaunches(true, 1));
    try std.testing.expectEqual(@as(u32, 1), extraLaunches(true, 0));
    try std.testing.expect(!recordsOn());
}

test "topk and max_mb clamp, and an empty dir is kept empty" {
    const a = std.testing.allocator;
    var c = try configFrom(a, .{ .on = "1", .topk = "0", .max_mb = "0", .dir = "" });
    defer c.deinit(a);
    try std.testing.expect(c.on);
    try std.testing.expectEqual(default_topk, c.topk);
    try std.testing.expectEqual(default_max_mb, c.max_mb);
    try std.testing.expectEqual(@as(usize, 0), c.dir.len);
    c.deinit(a);
    c = try configFrom(a, .{ .on = "1", .topk = "1", .max_mb = "8" });
    try std.testing.expectEqual(@as(u32, 1), c.topk);
    try std.testing.expectEqual(@as(u32, 8), c.max_mb);
    c.deinit(a);
    c = try configFrom(a, .{ .on = "1", .topk = "999" });
    try std.testing.expectEqual(n_experts, c.topk);
    var buf: [16]u8 = undefined;
    try std.testing.expectEqualStrings(" rrec=384", planSuffix(&buf, c));
    c.deinit(a);
    var off = try configFrom(a, .{ .on = "0" });
    defer off.deinit(a);
    try std.testing.expect(!off.on);
}

test "unix 0 is 19700101T000000Z and the file name sorts with the sequence" {
    const c = civilFromUnix(0);
    try std.testing.expectEqual(@as(i32, 1970), c.y);
    try std.testing.expectEqual(@as(u32, 1), c.m);
    try std.testing.expectEqual(@as(u32, 1), c.d);
    try std.testing.expectEqual(@as(u32, 0), c.hh);
    var buf: [96]u8 = undefined;
    try std.testing.expectEqualStrings("route-19700101T000000Z-000000.jsonl", fileName(&buf, 0, 0));
    try std.testing.expectEqualStrings("route-19700101T000001Z-000003.jsonl", fileName(&buf, 1, 3));
}

test "the kernel reference matches an independent f64 walk, including a reused slot" {
    const a = std.testing.allocator;
    const nslots: u32 = 4;
    const layers: u32 = grid_layers;
    const rows: u32 = 6;
    const stride: u32 = rows_cap_decode;
    const n = @as(usize, nslots) * cells;
    var mass_k = try a.alloc(f64, n);
    defer a.free(mass_k);
    var count_k = try a.alloc(u32, n);
    defer a.free(count_k);
    const mass_i = try a.alloc(f64, n);
    defer a.free(mass_i);
    const count_i = try a.alloc(u32, n);
    defer a.free(count_i);
    @memset(mass_k, 0);
    @memset(count_k, 0);
    @memset(mass_i, 0);
    @memset(count_i, 0);
    const pick_n = @as(usize, layers) * stride * route_slots;
    var pick = try a.alloc(i32, pick_n);
    defer a.free(pick);
    var wts = try a.alloc(f32, pick_n);
    defer a.free(wts);
    @memset(pick, -1);
    @memset(wts, 0);
    var rng = Rng{ .s = 0x9e3779b97f4a7c15 };
    var commit: [rows]u8 = undefined;
    var slotmap: [rows]u8 = undefined;
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        commit[r] = if (rng.u32n(3) == 0) 0 else 1;
        slotmap[r] = switch (rng.u32n(5)) {
            0 => no_slot,
            1 => 0,
            2 => 1,
            3 => 2,
            else => 7, // past nslots
        };
        var layer: u32 = 0;
        while (layer < layers) : (layer += 1) {
            var k: u32 = 0;
            while (k < route_slots) : (k += 1) {
                const at = pickIndex(layer, stride, r, route_slots, k);
                const roll = rng.u32n(8);
                pick[at] = switch (roll) {
                    0 => -1,
                    1 => n_experts, // shared expert, not recorded
                    else => @intCast(rng.u32n(n_experts)),
                };
                const bits: u32 = rng.u32n(1000);
                wts[at] = @as(f32, @floatFromInt(bits)) / 1000.0;
            }
        }
    }
    const view = View{
        .pick = pick,
        .wts = wts,
        .commit = &commit,
        .slotmap = &slotmap,
        .layers = layers,
        .rows = rows,
        .row_stride = stride,
        .slots = route_slots,
        .top_k = topk,
        .nexp = n_experts,
        .nslots = nslots,
        .grid_layers = grid_layers,
    };
    accumulate(mass_k, count_k, view);
    accumulateIndep(mass_i, count_i, view);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mass_k), std.mem.sliceAsBytes(mass_i));
    try std.testing.expectEqualSlices(u32, count_k, count_i);
    // a second round, different mask, still bit-equal, and slot 0 zeroed then reused matches a fresh grid
    r = 0;
    while (r < rows) : (r += 1) {
        commit[r] = 1;
        slotmap[r] = if (r % 2 == 0) 0 else 1;
    }
    accumulate(mass_k, count_k, view);
    accumulateIndep(mass_i, count_i, view);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mass_k), std.mem.sliceAsBytes(mass_i));
    const slot0 = cells;
    @memset(mass_k[0..slot0], 0);
    @memset(count_k[0..slot0], 0);
    const fresh_m = try a.alloc(f64, n);
    defer a.free(fresh_m);
    const fresh_c = try a.alloc(u32, n);
    defer a.free(fresh_c);
    @memcpy(fresh_m, mass_k);
    @memcpy(fresh_c, count_k);
    var map2: [rows]u8 = undefined;
    @memset(&map2, 0);
    var commit2: [rows]u8 = @splat(1);
    commit2[1] = 0;
    const again = View{
        .pick = pick,
        .wts = wts,
        .commit = &commit2,
        .slotmap = &map2,
        .layers = layers,
        .rows = rows,
        .row_stride = stride,
        .slots = route_slots,
        .top_k = topk,
        .nexp = n_experts,
        .nslots = nslots,
        .grid_layers = grid_layers,
    };
    accumulate(mass_k, count_k, again);
    accumulate(fresh_m, fresh_c, again);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mass_k), std.mem.sliceAsBytes(fresh_m));
    try std.testing.expectEqualSlices(u32, count_k, fresh_c);
}

test "a prompt chunk and an encoder cut leave unread layers at zero" {
    const a = std.testing.allocator;
    const rows: u32 = 4;
    const stride: u32 = rows_cap_prompt;
    const ran: u32 = 20;
    const pick_n = @as(usize, ran) * stride * route_slots;
    var pick = try a.alloc(i32, pick_n);
    defer a.free(pick);
    var wts = try a.alloc(f32, pick_n);
    defer a.free(wts);
    @memset(pick, 0);
    @memset(wts, 0.5);
    var layer: u32 = 0;
    while (layer < ran) : (layer += 1) {
        var row: u32 = 0;
        while (row < rows) : (row += 1) {
            const at = pickIndex(layer, stride, row, route_slots, 0);
            pick[at] = 3;
            wts[at] = 1.25;
        }
    }
    const mass = try a.alloc(f64, cells);
    defer a.free(mass);
    const count = try a.alloc(u32, cells);
    defer a.free(count);
    @memset(mass, 0);
    @memset(count, 0);
    var commit: [rows]u8 = @splat(1);
    var slotmap: [rows]u8 = @splat(0);
    slotmap[2] = no_slot;
    accumulate(mass, count, .{
        .pick = pick,
        .wts = wts,
        .commit = &commit,
        .slotmap = &slotmap,
        .layers = ran,
        .rows = rows,
        .row_stride = stride,
        .slots = route_slots,
        .top_k = topk,
        .nexp = n_experts,
        .nslots = 1,
        .grid_layers = grid_layers,
    });
    // three committed rows (one slot is 0xFF) hit expert 3 on layers 0..19
    try std.testing.expectEqual(@as(u32, 3), count[accIndex(0, grid_layers, 0, 3, n_experts)]);
    try bitEq(3.75, mass[accIndex(0, grid_layers, 19, 3, n_experts)]);
    try std.testing.expectEqual(@as(u32, 0), count[accIndex(0, grid_layers, 20, 3, n_experts)]);
    try std.testing.expectEqual(@as(u32, 0), count[accIndex(0, grid_layers, 39, 0, n_experts)]);
    const spans = [_]Span{.{ .at = 10, .len = 2 }};
    const split = splitPrompt(8, 6, &spans);
    try std.testing.expectEqual(@as(u32, 2), split.vision);
    try std.testing.expectEqual(@as(u32, 4), split.prompt);
}

test "top-K breaks ties by expert id and the totals cover every expert" {
    var mass: [n_experts]f64 = @splat(0);
    var count: [n_experts]u32 = @splat(0);
    mass[5] = 2;
    mass[9] = 2;
    mass[1] = 1;
    count[5] = 4;
    count[9] = 4;
    count[1] = 1;
    var out: [n_experts]Top = undefined;
    try std.testing.expectEqual(@as(u32, 1), selectTop(&mass, &count, 1, &out));
    try std.testing.expectEqual(@as(u32, 5), out[0].expert);
    try std.testing.expectEqual(@as(u32, 16), selectTop(&mass, &count, 16, &out));
    try std.testing.expectEqual(@as(u32, 5), out[0].expert);
    try std.testing.expectEqual(@as(u32, 9), out[1].expert);
    try std.testing.expectEqual(@as(u32, 1), out[2].expert);
    try std.testing.expectEqual(@as(u32, 0), out[3].expert); // zeros, lower id first
    try std.testing.expectEqual(n_experts, selectTop(&mass, &count, 384, &out));
    try std.testing.expectEqual(@as(u32, 383), out[383].expert);
    const tot = layerTotal(&mass, &count);
    try bitEq(5, tot.mass);
    try std.testing.expectEqual(@as(u64, 9), tot.count);
    @memset(&mass, 0);
    _ = selectTop(&mass, &count, 3, &out);
    try std.testing.expectEqual(@as(u32, 0), out[0].expert);
    try std.testing.expectEqual(@as(u32, 1), out[1].expert);
    try std.testing.expectEqual(@as(u32, 2), out[2].expert);
}

test "a record line keeps key order and parses back" {
    const a = std.testing.allocator;
    var meta: Meta = .{ .world = 4, .t_start = 10, .t_end = 20, .topk = 1, .rows = .{ .prompt = 3, .prompt_cached = 8, .decode = 2, .vision = 1 } };
    meta.setId("rr-aa-1");
    meta.setRequest("chatcmpl-abc");
    meta.setModel("flash");
    meta.setBuild(zon_version);
    meta.setFinish("stop");
    var mass = try a.alloc(f64, cells);
    defer a.free(mass);
    var count = try a.alloc(u32, cells);
    defer a.free(count);
    @memset(mass, 0);
    @memset(count, 0);
    mass[accIndex(0, grid_layers, 0, 7, n_experts)] = 1.5;
    count[accIndex(0, grid_layers, 0, 7, n_experts)] = 2;
    const buf = try a.alloc(u8, 2 * 1024 * 1024);
    defer a.free(buf);
    var w = std.Io.Writer.fixed(buf);
    try writeRecord(&w, &meta, mass, count);
    const line = buf[0..w.end];
    const keys = [_][]const u8{ "\"v\"", "\"id\"", "\"request_id\"", "\"route_tag\"", "\"model\"", "\"build\"", "\"world\"", "\"t_start\"", "\"t_end\"", "\"finish\"", "\"rows\"", "\"prompt\"", "\"prompt_cached\"", "\"decode\"", "\"vision\"", "\"topk\"", "\"layers\"" };
    var from: usize = 0;
    for (keys) |k| {
        const found = std.mem.indexOfPos(u8, line, from, k) orelse return error.KeyOrder;
        from = found + k.len;
    }
    try std.testing.expect(std.mem.indexOf(u8, line, "\"route_tag\":null") != null);
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), line, .{});
    try std.testing.expectEqual(@as(i64, 1), root.object.get("v").?.integer);
    try std.testing.expectEqualStrings("rr-aa-1", root.object.get("id").?.string);
    try std.testing.expectEqualStrings("chatcmpl-abc", root.object.get("request_id").?.string);
    try std.testing.expect(root.object.get("route_tag").? == .null);
    try std.testing.expectEqual(@as(i64, 4), root.object.get("world").?.integer);
    const layers = root.object.get("layers").?.array.items;
    try std.testing.expectEqual(@as(usize, 40), layers.len);
    const top = layers[0].object.get("top").?.array.items;
    try std.testing.expectEqual(@as(usize, 1), top.len);
    try std.testing.expectEqual(@as(i64, 7), top[0].array.items[0].integer);
    try bitEq(1.5, top[0].array.items[1].float);
    try std.testing.expectEqual(@as(i64, 2), top[0].array.items[2].integer);
    try bitEq(1.5, layers[0].object.get("mass").?.float);
    try std.testing.expectEqual(@as(i64, 2), layers[0].object.get("count").?.integer);
    meta.setTag("task=abc123/arm=tp4-high/draw=0/turn=3");
    meta.topk = 384;
    w.end = 0;
    try writeRecord(&w, &meta, mass, count);
    const wide = buf[0..w.end];
    try std.testing.expect(std.mem.indexOf(u8, wide, "\"route_tag\":\"task=abc123/arm=tp4-high/draw=0/turn=3\"") != null);
    var arena2: std.heap.ArenaAllocator = .init(a);
    defer arena2.deinit();
    const again = try std.json.parseFromSliceLeaky(std.json.Value, arena2.allocator(), wide, .{});
    try std.testing.expectEqual(@as(usize, 384), again.object.get("layers").?.array.items[0].object.get("top").?.array.items.len);
}

fn tmpPath(a: Allocator, sub: []const u8) ![]u8 {
    return std.fs.path.join(a, &.{ ".zig-cache", "tmp", sub });
}

test "a missing records directory is created (one level)" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const base = try tmpPath(a, &tmp.sub_path);
    defer a.free(base);
    const dir = try std.fs.path.join(a, &.{ base, "route-records" });
    defer a.free(dir);
    var files = try Files.open(io, a, dir, 1 << 20);
    try files.writeLine("{}\n");
    files.close();
    var sub = try tmp.dir.openDir(io, "route-records", .{ .iterate = true });
    defer sub.close(io);
    var it = sub.iterate();
    var n: usize = 0;
    while (try it.next(io)) |e| {
        if (isRouteLog(e.name)) n += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), n);
}

test "record files rotate, drop the oldest, and leave other names alone" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const dir = try tmpPath(a, &tmp.sub_path);
    defer a.free(dir);
    // 70 bytes each: after the older one is deleted the total is still over the 64-byte cap,
    // so both seeded logs go and the empty current file stays.
    var seed: [70]u8 = undefined;
    @memset(&seed, 'a');
    seed[69] = '\n';
    try tmp.dir.writeFile(io, .{ .sub_path = "route-20200101T000000Z-000000.jsonl", .data = &seed });
    try tmp.dir.writeFile(io, .{ .sub_path = "route-20200101T000001Z-000001.jsonl", .data = &seed });
    try tmp.dir.writeFile(io, .{ .sub_path = "notes.txt", .data = "keep" });
    // cap 64 bytes: rotate_at stays 1 MiB, so one short line does not rotate, and prune must
    // not delete the current file even though the seeded logs plus it exceed the cap... the
    // seeded logs are deleted, the current file stays.
    var files = try Files.open(io, a, dir, 64);
    files.now_sec = 1_700_000_000;
    defer files.close();
    try files.writeLine("now\n");
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var saw_notes = false;
    var saw_old = false;
    var route_n: u32 = 0;
    var it = d.iterate();
    while (try it.next(io)) |e| {
        if (std.mem.eql(u8, e.name, "notes.txt")) saw_notes = true;
        if (std.mem.startsWith(u8, e.name, "route-20200101")) saw_old = true;
        if (isRouteLog(e.name)) route_n += 1;
    }
    try std.testing.expect(saw_notes);
    try std.testing.expect(!saw_old);
    try std.testing.expect(route_n >= 1);
    // a current file larger than the cap is kept
    var tight = try Files.open(io, a, dir, 8);
    tight.now_sec = 1_700_000_100;
    defer tight.close();
    try tight.writeLine("longer than eight\n");
    try tight.prune();
    var f = try d.openFile(io, tight.curName(), .{});
    f.close(io);
}

test "rotation deletes the oldest route log and never the current file" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmpPath(a, &tmp.sub_path);
    defer a.free(dir);
    var files = try Files.open(io, a, dir, 1 << 20);
    files.now_sec = 1_600_000_000;
    defer files.close();
    // rotate_at is 1 MiB. One line past that, then another line forces a rotation.
    const big = try a.alloc(u8, (1 << 20) + 2);
    defer a.free(big);
    @memset(big, 'x');
    big[big.len - 1] = '\n';
    try files.writeLine(big);
    const first = try a.dupe(u8, files.curName());
    defer a.free(first);
    files.now_sec = 1_600_000_060;
    try files.writeLine("next\n");
    try std.testing.expect(!std.mem.eql(u8, first, files.curName()));
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer d.close(io);
    const gone = d.openFile(io, first, .{});
    try std.testing.expectError(error.FileNotFound, gone);
    var cur = try d.openFile(io, files.curName(), .{});
    cur.close(io);
}

test "a full queue and an empty pool are counted, and shutdown drains the queue" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var lane = try Lane.init(a, 2, 1);
    defer lane.deinit();
    const b0 = lane.take() orelse return error.NoBuffer;
    var meta: Meta = .{ .topk = 1, .world = 2 };
    meta.setId("rr-1-0");
    meta.setRequest("chatcmpl-z");
    meta.setFinish("stop");
    try std.testing.expect(lane.give(b0, meta));
    const b1 = lane.take() orelse return error.NoBuffer;
    try std.testing.expect(!lane.give(b1, meta));
    lane.addDrop();
    lane.release(b1);
    // release put one buffer back; taking it empties the pool again
    const returned = lane.take() orelse return error.NoBuffer;
    try std.testing.expect(lane.take() == null);
    lane.release(returned);
    lane.addDrop();
    try std.testing.expectEqual(@as(u64, 2), lane.drops());
    const job = lane.pop() orelse return error.Empty;
    lane.release(job.bi);
    // drain: three records through a live writer
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmpPath(a, &tmp.sub_path);
    defer a.free(dir);
    var pool = try Lane.init(a, 4, 8);
    defer pool.deinit();
    var w = try Writer.init(a, io, &pool, dir, 8 << 20, 1);
    try w.spawn();
    defer w.close();
    var i: u32 = 0;
    while (i < 3) : (i += 1) {
        const bi = pool.take() orelse return error.NoBuffer;
        @memset(pool.massOf(bi), 0);
        @memset(pool.countOf(bi), 0);
        var m: Meta = .{ .topk = 1, .t_start = i, .t_end = i + 1 };
        var idb: [32]u8 = undefined;
        const id = try std.fmt.bufPrint(&idb, "rr-d-{d}", .{i});
        m.setId(id);
        m.setRequest(id);
        m.setFinish("stop");
        try std.testing.expect(pool.give(bi, m));
    }
    w.close();
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    var it = d.iterate();
    while (try it.next(io)) |e| {
        if (!isRouteLog(e.name)) continue;
        var f = try d.openFile(io, e.name, .{});
        defer f.close(io);
        const st = try f.stat(io);
        const buf = try a.alloc(u8, @intCast(st.size));
        defer a.free(buf);
        const n = try f.readPositionalAll(io, buf, 0);
        try text.appendSlice(a, buf[0..n]);
    }
    var lines: u32 = 0;
    var split = std.mem.splitScalar(u8, text.items, '\n');
    while (split.next()) |line| if (line.len > 0) {
        lines += 1;
        try std.testing.expect(std.mem.startsWith(u8, line, "{\"v\":1,"));
    };
    try std.testing.expectEqual(@as(u32, 3), lines);
}

fn monoNs() u64 {
    var ts: std.posix.timespec = undefined;
    const rc = std.posix.system.clock_gettime(std.posix.CLOCK.MONOTONIC, &ts);
    if (std.posix.errno(rc) != .SUCCESS) return 0;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

test "route record host microbench" {
    const Plan = struct {
        fn once(slots: []const i64, live: []const bool, out: []u8) u64 {
            fillSlotMap(out, slots, live);
            return out[0];
        }
    };
    const a = std.testing.allocator;
    const iters: usize = if (@import("builtin").mode == .debug) 1 else 200;
    var slots_row: [8]i64 = .{ 0, 1, 2, 3, 0, 1, 4, 15 };
    var live: [16]bool = @splat(true);
    live[4] = false;
    var map: [8]u8 = undefined;
    var fnp: *const fn ([]const i64, []const bool, []u8) u64 = Plan.once;
    const vp: *volatile *const fn ([]const i64, []const bool, []u8) u64 = &fnp;
    var sink: u64 = 0;
    var t0 = monoNs();
    var n: usize = 0;
    while (n < iters * 1000) : (n += 1) sink +%= vp.*(&slots_row, &live, &map);
    std.mem.doNotOptimizeAway(sink);
    const plan_ns = (monoNs() - t0) / (iters * 1000);

    const layers: u32 = 40;
    const rows: u32 = 8;
    const stride: u32 = 64;
    var pick = try a.alloc(i32, layers * stride * route_slots);
    defer a.free(pick);
    var wts = try a.alloc(f32, pick.len);
    defer a.free(wts);
    @memset(pick, -1);
    @memset(wts, 0);
    var commit: [rows]u8 = @splat(1);
    var slotmap: [rows]u8 = .{ 0, 1, 2, 3, 0, 1, 2, 3 };
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        var layer: u32 = 0;
        while (layer < layers) : (layer += 1) {
            var k: u32 = 0;
            while (k < topk) : (k += 1) {
                const at = pickIndex(layer, stride, r, route_slots, k);
                pick[at] = @intCast((r * 3 + k * 5 + layer) % n_experts);
                wts[at] = 0.25;
            }
        }
    }
    const mass = try a.alloc(f64, 4 * cells);
    defer a.free(mass);
    const count = try a.alloc(u32, 4 * cells);
    defer a.free(count);
    const view = View{
        .pick = pick,
        .wts = wts,
        .commit = &commit,
        .slotmap = &slotmap,
        .layers = layers,
        .rows = rows,
        .row_stride = stride,
        .slots = route_slots,
        .top_k = topk,
        .nexp = n_experts,
        .nslots = 4,
        .grid_layers = grid_layers,
    };
    const kern_iters: usize = if (@import("builtin").mode == .debug) 1 else 4;
    t0 = monoNs();
    n = 0;
    while (n < kern_iters) : (n += 1) {
        @memset(mass, 0);
        @memset(count, 0);
        accumulate(mass, count, view);
    }
    std.mem.doNotOptimizeAway(mass[0]);
    const kern_ns = (monoNs() - t0) / kern_iters;

    var one_m = try a.alloc(f64, cells);
    defer a.free(one_m);
    var one_c = try a.alloc(u32, cells);
    defer a.free(one_c);
    var e: usize = 0;
    while (e < cells) : (e += 1) {
        one_m[e] = @as(f64, @floatFromInt(e % 9)) * 0.25;
        one_c[e] = @intCast(e % 3);
    }
    var meta: Meta = .{ .topk = 16, .world = 4, .t_start = 1, .t_end = 2, .rows = .{ .prompt = 4, .decode = 8 } };
    meta.setId("rr-bench-1");
    meta.setRequest("chatcmpl-bench");
    meta.setTag("task=bench/arm=tp4");
    meta.setModel("flash");
    meta.setBuild(zon_version);
    meta.setFinish("stop");
    const ser_buf = try a.alloc(u8, 2 * 1024 * 1024);
    defer a.free(ser_buf);
    const Ser = struct {
        fn once(buf: []u8, m: *const Meta, mass_s: []const f64, count_s: []const u32) usize {
            var wr = std.Io.Writer.fixed(buf);
            writeRecord(&wr, m, mass_s, count_s) catch return 0;
            return wr.end;
        }
    };
    var ser_fn: *const fn ([]u8, *const Meta, []const f64, []const u32) usize = Ser.once;
    const ser_vp: *volatile *const fn ([]u8, *const Meta, []const f64, []const u32) usize = &ser_fn;
    const ser_iters: usize = if (@import("builtin").mode == .debug) 1 else 8;
    const warmed = ser_vp.*(ser_buf, &meta, one_m, one_c);
    try std.testing.expect(warmed > 1000);
    std.mem.doNotOptimizeAway(warmed);
    t0 = monoNs();
    var ser_sink: usize = 0;
    n = 0;
    while (n < ser_iters) : (n += 1) ser_sink += ser_vp.*(ser_buf, &meta, one_m, one_c);
    std.mem.doNotOptimizeAway(ser_sink);
    const ser_ns = (monoNs() - t0) / ser_iters;
    std.debug.print("route record microbench slotmap {d} ns/round, cpu kernel {d} ns (R=8 layers=40 experts=384), topk+json {d} ns (K=16)\n", .{ plan_ns, kern_ns, ser_ns });
}
