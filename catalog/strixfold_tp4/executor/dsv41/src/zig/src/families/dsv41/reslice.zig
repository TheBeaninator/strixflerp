//! A TP4 (2D split) rank file's entries cut from the TP2 rank file of the same TP2 rank (node g reads rank g % 2 of
//! 2): every entry of plan.wants at node g is a TP2 entry of the same tensor and part, either whole (plain tensors,
//! DSpark, suh, wo_a slices, anything the pair does not split) or a whole-128-block column range of it (wq_b, the head,
//! the experts' gate / up, and the output columns of wo_b, down and Engram wkv), the input rows always the TP2 ones.
//! cut() names the byte runs; copy() takes them, same() compares them tile by tile.
const std = @import("std");
const rank_cache = @import("rank_cache.zig");
const DType = rank_cache.DType;
const Entry = rank_cache.Entry;

/// A loader key: "name|dtype" (plain) or "prefix|cols|rows|part" (an EXL3 part or a dense slice).
pub const Key = struct {
    name: []const u8,
    exl3: bool = false,
    cols: ?[2]usize = null,
    rows: ?[2]usize = null,
    part: []const u8 = "",

    pub fn parse(key: []const u8) !Key {
        var f: [4][]const u8 = undefined;
        var n: usize = 0;
        var it = std.mem.splitScalar(u8, key, '|');
        while (it.next()) |x| : (n += 1) {
            if (n == 4) return error.BadKey;
            f[n] = x;
        }
        if (n == 2) return .{ .name = f[0] };
        if (n != 4) return error.BadKey;
        return .{ .name = f[0], .exl3 = true, .cols = try range(f[1]), .rows = try range(f[2]), .part = f[3] };
    }

    /// "None" or Python's "(a, b)".
    fn range(text: []const u8) !?[2]usize {
        if (std.mem.eql(u8, text, "None")) return null;
        if (text.len < 5 or text[0] != '(' or text[text.len - 1] != ')') return error.BadKey;
        var it = std.mem.splitSequence(u8, text[1 .. text.len - 1], ", ");
        const a = try std.fmt.parseInt(usize, it.next() orelse return error.BadKey, 10);
        const b = try std.fmt.parseInt(usize, it.next() orelse return error.BadKey, 10);
        if (it.next() != null or b < a) return error.BadKey;
        return .{ a, b };
    }
};

fn eqRange(x: ?[2]usize, y: ?[2]usize) bool {
    if (x == null or y == null) return x == null and y == null;
    return x.?[0] == y.?[0] and x.?[1] == y.?[1];
}

/// Where a TP4 entry lies in its TP2 entry: `count` runs of `len` bytes, run i at `first + i * stride` of the TP2
/// entry's bytes; dtype and shape are the TP4 entry's.
pub const Cut = struct {
    from: []const u8,
    src: Entry,
    dtype: DType,
    rank: u8,
    shape: [rank_cache.max_rank]usize,
    count: usize = 1,
    stride: usize = 0,
    first: usize = 0,
    len: usize,
    /// the trellis tile (bytes) when the runs are trellis rows, else 0
    tile: usize = 0,

    pub fn bytes(c: Cut) u64 {
        return c.count * c.len;
    }
    pub fn whole(c: Cut) bool {
        return c.count == 1 and c.first == 0 and c.len == c.src.bytes;
    }
    pub fn dim(c: Cut, i: usize) usize {
        return if (i < c.rank) c.shape[i] else 1;
    }
};

/// The TP2 file's EXL3 / dense keys by "name|part" (one each); plain keys are looked up as they are.
pub const Parts = std.StringHashMapUnmanaged([]const u8);

pub fn parts(a: std.mem.Allocator, tp2: *const rank_cache.Index) !Parts {
    var m: Parts = .empty;
    for (tp2.entries.keys()) |k| {
        const key = try Key.parse(k);
        if (!key.exl3) continue;
        const gop = try m.getOrPut(a, try std.fmt.allocPrint(a, "{s}|{s}", .{ key.name, key.part }));
        if (gop.found_existing) return error.DuplicatePart;
        gop.value_ptr.* = k;
    }
    return m;
}

fn wholeOf(from: []const u8, e: Entry) Cut {
    return .{ .from = from, .src = e, .dtype = e.dtype, .rank = e.rank, .shape = e.shape, .len = e.bytes };
}

/// The TP4 entry `key4` as a cut of the TP2 file's entries.
pub fn cut(tp2: *const rank_cache.Index, by_part: *const Parts, key4: []const u8) !Cut {
    const k = try Key.parse(key4);
    if (!k.exl3) return wholeOf(key4, tp2.get(key4) orelse return error.NotInTp2);
    var buf: [320]u8 = undefined;
    const from = by_part.get(try std.fmt.bufPrint(&buf, "{s}|{s}", .{ k.name, k.part })) orelse return error.NotInTp2;
    const e = tp2.get(from).?;
    const s = try Key.parse(from);
    // the 2D split keeps the TP2 rank's input rows and narrows only output columns
    if (!eqRange(k.rows, s.rows)) return error.NotASlice;
    if (eqRange(k.cols, s.cols) or std.mem.eql(u8, k.part, "suh")) return wholeOf(from, e);
    const c4 = k.cols orelse return error.NotASlice;
    const es = e.dtype.size();
    const tr = std.mem.eql(u8, k.part, "tr");
    const svh = std.mem.eql(u8, k.part, "svh");
    const dense = std.mem.eql(u8, k.part, "dense");
    if (!(tr or svh or dense)) return error.UnknownPart;
    if ((tr and e.rank != 3) or (svh and e.rank != 1) or (dense and e.rank != 2)) return error.BadShape;
    // the TP2 entry's output columns: its own slice, else all of the tensor's
    const n_all: usize = if (tr) e.dim(1) * 16 else e.dim(0);
    const base: [2]usize = s.cols orelse .{ 0, n_all };
    if (base[1] - base[0] != n_all) return error.BadShape;
    if (c4[0] < base[0] or c4[1] > base[1] or c4[1] <= c4[0]) return error.NotASlice;
    const lo = c4[0] - base[0];
    const w = c4[1] - c4[0];
    var out: Cut = .{ .from = from, .src = e, .dtype = e.dtype, .rank = e.rank, .shape = e.shape, .len = 0 };
    if (tr) {
        // [K / 16, N / 16, words]: every K tile row's run of the column tiles
        if (lo % 16 != 0 or w % 16 != 0) return error.NotASlice;
        const tile = e.dim(2) * es;
        out.shape[1] = w / 16;
        out.count = e.dim(0);
        out.stride = e.dim(1) * tile;
        out.first = lo / 16 * tile;
        out.len = w / 16 * tile;
        out.tile = tile;
    } else if (svh) {
        out.shape[0] = w;
        out.first = lo * es;
        out.len = w * es;
    } else {
        // dense [N, K]: output columns are its rows
        const row = e.dim(1) * es;
        out.shape[0] = w;
        out.first = lo * row;
        out.len = w * row;
    }
    return out;
}

/// The cut's bytes from its TP2 entry's bytes `src` into `out` (c.bytes() long).
pub fn copy(c: Cut, src: []const u8, out: []u8) !void {
    if (src.len != c.src.bytes or out.len != c.bytes()) return error.BadSize;
    for (0..c.count) |i| @memcpy(out[i * c.len ..][0..c.len], src[c.first + i * c.stride ..][0..c.len]);
}

/// Whether `got` (a TP4 entry's bytes) is the cut of `src`: a trellis compared tile by tile (K tile row i, column
/// tile j of the TP4 entry against tile j + first / tile of the TP2 row), anything else run by run.
pub fn same(c: Cut, src: []const u8, got: []const u8) bool {
    if (src.len != c.src.bytes or got.len != c.bytes()) return false;
    if (c.tile != 0) {
        const n4 = c.len / c.tile;
        const n2 = c.stride / c.tile;
        const j0 = c.first / c.tile;
        for (0..c.count) |i| for (0..n4) |j| {
            const a = got[(i * n4 + j) * c.tile ..][0..c.tile];
            const b = src[(i * n2 + j0 + j) * c.tile ..][0..c.tile];
            if (!std.mem.eql(u8, a, b)) return false;
        };
        return true;
    }
    for (0..c.count) |i| if (!std.mem.eql(u8, got[i * c.len ..][0..c.len], src[c.first + i * c.stride ..][0..c.len])) return false;
    return true;
}

test "TP4 cuts of a TP2 file: whole entries, trellis column tiles, svh and dense row ranges" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // TP2 rank 1: a plain tensor, wq_b's columns (512, 1024) as a [2, 32, 2] trellis + svh, wo_b's rows with all 256
    // output columns as a [2, 16, 2] trellis, and a dense [256, 8] wo_b slice
    const json =
        \\[["norm.weight|None", "float16", [4], 0, 8],
        \\ ["l.wq_b|(512, 1024)|None|tr", "int16", [2, 32, 2], 8, 256],
        \\ ["l.wq_b|(512, 1024)|None|svh", "float16", [512], 264, 1024],
        \\ ["l.wo_b|None|(64, 96)|tr", "int16", [2, 16, 2], 1288, 128],
        \\ ["l.wo_b|None|(64, 96)|suh", "float16", [32], 1416, 64],
        \\ ["m.w2|None|(8, 16)|dense", "bfloat16", [256, 8], 1480, 4096]]
    ;
    var ix = try rank_cache.parseIndex(gpa, json, 5576);
    defer ix.deinit();
    const by = try parts(a, &ix);
    const src = try a.alloc(u8, 5576);
    for (src, 0..) |*b, i| b.* = @truncate(i *% 131 +% i / 251);

    const plain = try cut(&ix, &by, "norm.weight|None");
    try std.testing.expect(plain.whole());
    const suh = try cut(&ix, &by, "l.wo_b|(0, 128)|(64, 96)|suh");
    try std.testing.expect(suh.whole());

    // pair 1's half of wq_b's 512 columns: (768, 1024) = column tiles 16..31 of each of the 2 K tile rows
    const tr = try cut(&ix, &by, "l.wq_b|(768, 1024)|None|tr");
    try std.testing.expectEqual(@as(usize, 16), tr.dim(1));
    try std.testing.expectEqual(@as(u64, 128), tr.bytes());
    const e = ix.get(tr.from).?;
    const out = try a.alloc(u8, tr.bytes());
    try copy(tr, src[e.offset..][0..e.bytes], out);
    try std.testing.expect(same(tr, src[e.offset..][0..e.bytes], out));
    // row 1, tile 0 of the cut is TP2 row 1, tile 16
    try std.testing.expectEqualSlices(u8, src[e.offset + (32 + 16) * 4 ..][0..4], out[16 * 4 ..][0..4]);
    out[77] ^= 1;
    try std.testing.expect(!same(tr, src[e.offset..][0..e.bytes], out));

    const svh = try cut(&ix, &by, "l.wq_b|(512, 768)|None|svh");
    try std.testing.expectEqual(@as(usize, 256), svh.dim(0));
    try std.testing.expectEqual(@as(usize, 0), svh.first);
    // wo_b stores all output columns: pair 0's (0, 128) are tiles 0..7 of 16
    const wo = try cut(&ix, &by, "l.wo_b|(0, 128)|(64, 96)|tr");
    try std.testing.expectEqual(@as(usize, 8), wo.dim(1));
    try std.testing.expectEqual(@as(usize, 2), wo.count);
    const dn = try cut(&ix, &by, "m.w2|(128, 256)|(8, 16)|dense");
    try std.testing.expectEqual(@as(usize, 128 * 16), dn.first);
    try std.testing.expectEqual(@as(usize, 128), dn.dim(0));

    try std.testing.expectError(error.NotASlice, cut(&ix, &by, "l.wo_b|(0, 128)|(32, 64)|tr"));
    try std.testing.expectError(error.NotASlice, cut(&ix, &by, "l.wq_b|(256, 512)|None|tr"));
    try std.testing.expectError(error.NotASlice, cut(&ix, &by, "l.wq_b|(520, 1024)|None|tr"));
    try std.testing.expectError(error.NotInTp2, cut(&ix, &by, "l.wkv|None|None|tr"));
}
