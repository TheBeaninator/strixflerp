//! The AMD kit loader (ADAPTER.md 7.4): reads a kit's manifest, checks it against this binary's ABI version and the
//! device's exact arch, checks every code object's SHA-256, ELF machine and own metadata against the manifest, loads it,
//! resolves every kernel, and refuses to start when a kernel the configuration needs is missing. Kernels are then
//! bound by id and specialization (typed: `Kernel(id, Params)`), never by name parsing. In strict mode (TF_GPU_STRICT,
//! on unless set to 0) every launch is checked against its kernel's metadata and counted, and a launch of a kernel
//! that is not in the manifest fails. TF_GPU_COUNTS=1 counts the launches without strict mode's checks.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const checkImage = @import("module.zig").checkImage;
const Stream = @import("stream.zig").Stream;
const launch_mod = @import("launch.zig");
const manifest = @import("../manifest.zig");
const hsaco = @import("../hsaco.zig");
const kernel_mod = @import("../kernel.zig");
const abi_h = @import("dsv41_abi");

/// The ABI this binary was built for: abi.h's TF_DS_HIP_ABI_VERSION (one definition, translated for Zig).
pub const abi_version: u32 = abi_h.TF_DS_HIP_ABI_VERSION;

pub const SpecItem = manifest.SpecItem;

/// `vision` false: the objects that hold only the vision tower's kernels (vision.*) are not read, loaded or required.
pub const Features = struct { drafts: bool = false, engram: bool = false, profile: bool = false, graphs: bool = false, vision: bool = true };

pub const Entry = struct {
    k: manifest.Kernel,
    f: Function,
    /// The explicit (caller-passed) arguments from the code object's metadata.
    explicit: []const hsaco.Arg,
    max_threads: u32,
};

/// The process's kit while its launches are checked or counted (one kit a process).
var active: ?*Kit = null;

pub const Kit = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    d: *const Driver,
    m: manifest.Manifest,
    /// SHA-256 of manifest.json's bytes: what a peer handshake and a benchmark manifest record.
    manifest_sha256: [64]u8,
    modules: []Module,
    entries: []Entry,
    by_handle: std.AutoHashMapUnmanaged(usize, u32) = .empty,
    counts: []std.atomic.Value(u64),
    /// Counted launches of kernels the manifest does not list (TF_GPU_COUNTS=1; strict mode refuses them).
    unlisted: std.atomic.Value(u64) = .init(0),
    strict: bool,

    /// Opens the kit in `dir` for a device of exactly `arch`, resolving every kernel `features` needs.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, dir: []const u8, arch: []const u8, features: Features) !*Kit {
        const k = try gpa.create(Kit);
        errdefer gpa.destroy(k);
        k.* = .{ .gpa = gpa, .arena = .init(gpa), .d = d, .m = undefined, .manifest_sha256 = undefined, .modules = &.{}, .entries = &.{}, .counts = &.{}, .strict = strictFromEnv() };
        errdefer k.arena.deinit();
        errdefer k.by_handle.deinit(gpa);
        const a = k.arena.allocator();
        const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "manifest.json" }), a, .limited(64 << 20));
        k.manifest_sha256 = manifest.sha256Hex(text);
        k.m = manifest.Manifest.parse(a, text) catch |e| {
            std.log.err("kit {s}: manifest.json is not a {s} manifest ({s})", .{ dir, manifest.schema, @errorName(e) });
            return error.KitManifest;
        };
        if (k.m.abi_version != abi_version) {
            std.log.err("kit {s} has ABI version {d}; this binary is built for {d} (abi.h TF_DS_HIP_ABI_VERSION)", .{ dir, k.m.abi_version, abi_version });
            return error.AbiMismatch;
        }
        if (!std.mem.eql(u8, k.m.arch, arch) or k.m.wave != 32) {
            std.log.err("kit {s} is built for {s} (wave {d}); this device is {s}", .{ dir, k.m.arch, k.m.wave, arch });
            return error.ArchMismatch;
        }
        const keep = try a.alloc(bool, k.m.objects.len);
        for (k.m.objects, keep) |o, *x| x.* = features.vision or !visionOnly(k.m, o.file);
        k.modules = try a.alloc(Module, std.mem.count(bool, keep, &.{true}));
        const slot = try a.alloc(usize, k.m.objects.len); // an object's module
        var loaded: usize = 0;
        errdefer for (k.modules[0..loaded]) |*mod| mod.unload();
        const metas = try a.alloc(hsaco.CodeObject, k.m.objects.len);
        for (k.m.objects, 0..) |o, i| {
            if (!keep[i]) continue;
            const bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ dir, o.file }), gpa, .limited(1 << 30), .@"16", null);
            defer gpa.free(bytes);
            const sum = manifest.sha256Hex(bytes);
            if (!std.mem.eql(u8, &sum, o.sha256) or bytes.len != o.bytes) {
                std.log.err("kit {s}: {s} does not match its manifest SHA-256/size", .{ dir, o.file });
                return error.HashMismatch;
            }
            try checkImage(bytes, arch);
            metas[i] = try ownCodeObject(a, try hsaco.parse(a, bytes));
            k.modules[loaded] = try Module.load(d, bytes);
            slot[i] = loaded;
            loaded += 1;
        }
        var entries: std.ArrayList(Entry) = .empty;
        for (k.m.kernels) |mk| {
            const oi = for (k.m.objects, 0..) |o, j| {
                if (std.mem.eql(u8, o.file, mk.object)) break j;
            } else {
                std.log.err("kit {s}: kernel {s} names object {s}, which the manifest does not list", .{ dir, mk.id, mk.object });
                return error.KitManifest;
            };
            if (!keep[oi]) continue;
            const meta = metas[oi].find(mk.symbol) orelse {
                std.log.err("kit {s}: {s} ({s}) is not in {s}", .{ dir, mk.symbol, mk.id, mk.object });
                return error.MissingKernel;
            };
            try sameAsMetadata(dir, mk, meta);
            if (meta.wave != 32 or meta.dynamic_stack or (k.strict and meta.scratch > 0 and !mk.allow_scratch)) {
                std.log.err("kit {s}: {s} wave {d}, dynamic stack {}, scratch {d} bytes (spills) not allowed by its registry entry", .{ dir, mk.id, meta.wave, meta.dynamic_stack, meta.scratch });
                return error.KernelRefused;
            }
            var name_z: [256]u8 = undefined;
            const f = try k.modules[slot[oi]].function(try std.fmt.bufPrintSentinel(&name_z, "{s}", .{mk.symbol}, 0));
            try k.by_handle.put(gpa, @intFromPtr(f.handle), @intCast(entries.items.len));
            try entries.append(a, .{ .k = mk, .f = f, .explicit = meta.explicit(), .max_threads = meta.max_flat_workgroup_size });
        }
        k.entries = entries.items;
        k.counts = try a.alloc(std.atomic.Value(u64), k.entries.len);
        for (k.counts) |*c| c.* = .init(0);
        try k.requireAll(dir, features);
        if (k.strict or countsFromEnv()) {
            active = k;
            launch_mod.before = if (k.strict) strictCheck else countOnly;
        }
        return k;
    }

    pub fn deinit(k: *Kit) void {
        if (active == k) {
            active = null;
            launch_mod.before = null;
        }
        for (k.modules) |*mod| mod.unload();
        k.by_handle.deinit(k.gpa);
        k.arena.deinit();
        k.gpa.destroy(k);
    }

    fn requireAll(k: *Kit, dir: []const u8, f: Features) !void {
        var missing: usize = 0;
        const lists = [_]struct { bool, []const []const u8 }{ .{ true, k.m.required.always }, .{ f.drafts, k.m.required.drafts }, .{ f.engram, k.m.required.engram }, .{ f.profile, k.m.required.profile }, .{ f.graphs, k.m.required.graphs } };
        for (lists) |l| if (l[0]) for (l[1]) |id| {
            if (k.anyOf(id) or (!f.vision and isVision(id))) continue;
            std.log.err("kit {s}: required kernel {s} is missing", .{ dir, id });
            missing += 1;
        };
        if (missing > 0) return error.MissingKernel;
    }

    fn anyOf(k: *const Kit, id: []const u8) bool {
        for (k.entries) |e| if (std.mem.eql(u8, e.k.id, id)) return true;
        return false;
    }

    /// The entry of `id` with exactly `spec`, or null.
    pub fn find(k: *const Kit, id: []const u8, spec: []const SpecItem) ?u32 {
        for (k.entries, 0..) |e, i| {
            if (std.mem.eql(u8, e.k.id, id) and manifest.sameSpec(e.k.spec, spec)) return @intCast(i);
        }
        return null;
    }

    /// The function of `id` with exactly `spec`, or MissingKernel naming it.
    pub fn function(k: *const Kit, id: []const u8, spec: []const SpecItem) !Function {
        const i = k.find(id, spec) orelse {
            std.log.err("the kit has no {s} ({d} spec fields)", .{ id, spec.len });
            return error.MissingKernel;
        };
        return k.entries[i].f;
    }

    /// Launches of entry `i` since open (strict mode or TF_GPU_COUNTS=1; zero otherwise).
    pub fn launches(k: *const Kit, i: u32) u64 {
        return k.counts[i].load(.monotonic);
    }

    /// The counters copied into `at` (entries.len + 1 values: each entry's, then the unlisted launches').
    pub fn mark(k: *const Kit, at: []u64) void {
        for (k.counts, at[0..k.counts.len]) |*c, *x| x.* = c.load(.monotonic);
        at[k.counts.len] = k.unlisted.load(.monotonic);
    }

    /// The launches since the mark `since` as a JSON object: the mode, the total, the launches of kernels outside the
    /// manifest and each launched kernel id's count (its specializations summed). A graph replay launches nothing
    /// here. Without room for the ids and `reserve` bytes after the object, it holds the totals alone.
    pub fn launchesJson(k: *const Kit, since: []const u64, w: *std.Io.Writer, reserve: usize) !void {
        const at = w.end;
        if (k.launchesInto(since, w, true)) |_| {
            if (w.buffer.len - w.end >= reserve) return;
        } else |_| {}
        w.end = at;
        try k.launchesInto(since, w, false);
    }

    fn launchesInto(k: *const Kit, since: []const u64, w: *std.Io.Writer, ids: bool) !void {
        const n = k.entries.len;
        const unlisted = k.unlisted.load(.monotonic) - since[n];
        var total = unlisted;
        for (k.counts, since[0..n]) |*c, s| total += c.load(.monotonic) - s;
        try w.print("{{\"mode\": \"{s}\", \"total\": {d}, \"unlisted\": {d}", .{ if (k.strict) "strict" else "count", total, unlisted });
        if (ids) {
            try w.writeAll(", \"by_kernel\": {");
            var first = true;
            for (k.entries, 0..) |e, i| {
                var sum: u64 = 0;
                for (k.entries, 0..) |o, j| if (std.mem.eql(u8, o.k.id, e.k.id)) {
                    if (j < i) break; // an id is reported at its first entry
                    sum += k.counts[j].load(.monotonic) - since[j];
                };
                if (sum == 0) continue;
                try w.print("{s}\"{s}\": {d}", .{ if (first) "" else ", ", e.k.id, sum });
                first = false;
            }
            try w.writeAll("}");
        }
        try w.writeAll("}");
    }
};

/// The kit whose launches are being counted (strict mode or TF_GPU_COUNTS=1), or null.
pub fn counted() ?*const Kit {
    return active;
}

fn isVision(id: []const u8) bool {
    return std.mem.startsWith(u8, id, "vision.");
}

/// Whether every kernel of `file` is the vision tower's (a text configuration then skips the object).
fn visionOnly(m: manifest.Manifest, file: []const u8) bool {
    var any = false;
    for (m.kernels) |mk| if (std.mem.eql(u8, mk.object, file)) {
        if (!isVision(mk.id)) return false;
        any = true;
    };
    return any;
}

test "a text configuration skips only the objects that hold nothing but vision kernels" {
    const kern = struct {
        fn of(id: []const u8, object: []const u8) manifest.Kernel {
            return .{ .id = id, .symbol = id, .object = object, .args = &.{}, .kernarg_size = 0, .wave = 32, .max_flat_workgroup_size = 256, .lds_static = 0, .scratch = 0, .vgpr = 0, .sgpr = 0 };
        }
    };
    const m: manifest.Manifest = .{ .schema = manifest.schema, .kit = "t", .abi_version = 1, .arch = "gfx1151", .wave = 32, .objects = &.{}, .kernels = &.{ kern.of("vision.linear", "v.hsaco"), kern.of("vision.span", "v.hsaco"), kern.of("vision.add", "mixed.hsaco"), kern.of("glue.x", "mixed.hsaco"), kern.of("glue.y", "g.hsaco") } };
    try std.testing.expect(visionOnly(m, "v.hsaco"));
    try std.testing.expect(!visionOnly(m, "mixed.hsaco") and !visionOnly(m, "g.hsaco") and !visionOnly(m, "none.hsaco"));
    try std.testing.expect(isVision("vision.rope") and !isVision("glue.vision"));
}

fn strictFromEnv() bool {
    const v = std.c.getenv("TF_GPU_STRICT") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

fn countsFromEnv() bool {
    const v = std.c.getenv("TF_GPU_COUNTS") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// The manifest's record of a kernel must be what its object says (a kit edited or rebuilt by halves is refused).
fn sameAsMetadata(dir: []const u8, mk: manifest.Kernel, meta: hsaco.Kernel) !void {
    var same = mk.args.len == meta.args.len and mk.kernarg_size == meta.kernarg_size and mk.wave == meta.wave and mk.max_flat_workgroup_size == meta.max_flat_workgroup_size;
    if (same) for (mk.args, meta.args) |x, y| {
        same = same and x.offset == y.offset and x.size == y.size and std.mem.eql(u8, x.kind, y.kind);
    };
    if (!same) {
        std.log.err("kit {s}: manifest entry {s} disagrees with {s}'s metadata in {s}", .{ dir, mk.id, mk.symbol, mk.object });
        return error.MetadataMismatch;
    }
}

/// The metadata with its strings copied into the arena (the object's bytes are freed after loading).
fn ownCodeObject(a: std.mem.Allocator, c: hsaco.CodeObject) !hsaco.CodeObject {
    const ks = try a.alloc(hsaco.Kernel, c.kernels.len);
    for (c.kernels, ks) |src, *dst| {
        const args = try a.alloc(hsaco.Arg, src.args.len);
        for (src.args, args) |x, *y| y.* = .{ .offset = x.offset, .size = x.size, .kind = try a.dupe(u8, x.kind) };
        dst.* = src;
        dst.name = try a.dupe(u8, src.name);
        dst.symbol = try a.dupe(u8, src.symbol);
        dst.args = args;
    }
    return .{ .e_flags = c.e_flags, .target = try a.dupe(u8, c.target), .kernels = ks };
}

/// Strict mode's check of every launch (launch.zig `before`): the kernel is in the manifest, the packed arguments sit
/// where its metadata says, the block fits its bounds and LDS its declaration.
fn strictCheck(f: Function, cfg: launch_mod.Config, args: *const launch_mod.Args) @import("driver.zig").Error!void {
    const k = active orelse return;
    const i = k.by_handle.get(@intFromPtr(f.handle)) orelse {
        std.log.err("strict: a launch of a kernel that is not in the kit manifest", .{});
        return error.Invalid;
    };
    const e = &k.entries[i];
    var ok = args.count == e.explicit.len;
    if (ok) for (e.explicit, 0..) |x, j| {
        ok = ok and args.offsets[j] == x.offset and args.sizes[j] == x.size;
    };
    const threads = @as(u64, cfg.block.x) * cfg.block.y * cfg.block.z;
    if (e.k.block) |b| ok = ok and cfg.block.x == b[0] and cfg.block.y == b[1] and cfg.block.z == b[2];
    ok = ok and threads <= e.max_threads and cfg.shared <= e.k.lds_dynamic_max and @as(u64, e.k.lds_static) + cfg.shared <= 65536;
    if (!ok) {
        std.log.err("strict: launch of {s} does not match its metadata ({d} args packed, {d} expected; block {d}x{d}x{d} of at most {d}; {d} dynamic LDS of {d} declared)", .{ e.k.id, args.count, e.explicit.len, cfg.block.x, cfg.block.y, cfg.block.z, e.max_threads, cfg.shared, e.k.lds_dynamic_max });
        return error.Invalid;
    }
    _ = k.counts[i].fetchAdd(1, .monotonic);
}

/// TF_GPU_COUNTS=1 without strict mode (launch.zig `before`): the launch is counted, nothing is checked or refused.
fn countOnly(f: Function, _: launch_mod.Config, _: *const launch_mod.Args) @import("driver.zig").Error!void {
    const k = active orelse return;
    const c = if (k.by_handle.get(@intFromPtr(f.handle))) |i| &k.counts[i] else &k.unlisted;
    _ = c.fetchAdd(1, .monotonic);
}

test "counted launches: each kernel id's count is the launches the hook saw; a mark makes them a request's" {
    const gpa = std.testing.allocator;
    const ids = [_][]const u8{ "glue.a", "tri.b", "tri.b", "glue.c" };
    var entries: [ids.len]Entry = undefined;
    var counts: [ids.len]std.atomic.Value(u64) = @splat(.init(0));
    var kit: Kit = .{ .gpa = gpa, .arena = .init(gpa), .d = undefined, .m = undefined, .manifest_sha256 = undefined, .modules = &.{}, .entries = &entries, .counts = &counts, .strict = false };
    defer kit.by_handle.deinit(gpa);
    for (ids, &entries, 0..) |id, *e, i| {
        const mk: manifest.Kernel = .{ .id = id, .symbol = id, .object = "t.hsaco", .args = &.{}, .kernarg_size = 0, .wave = 32, .max_flat_workgroup_size = 256, .lds_static = 0, .scratch = 0, .vgpr = 0, .sgpr = 0 };
        e.* = .{ .k = mk, .f = .{ .d = undefined, .handle = @ptrFromInt(0x1000 + 16 * i) }, .explicit = &.{}, .max_threads = 256 };
        try kit.by_handle.put(gpa, @intFromPtr(e.f.handle), @intCast(i));
    }
    active = &kit;
    defer active = null;
    const cfg: launch_mod.Config = .{ .grid = .{ .x = 4, .y = 1, .z = 1 }, .block = .{ .x = 64, .y = 1, .z = 1 } };
    const args: launch_mod.Args = .{};
    // count-only: 3, 2, 5 and 0 launches of the four entries, and one of a kernel the manifest does not list
    for ([_]usize{ 3, 2, 5, 0 }, entries) |n, e| for (0..n) |_| try countOnly(e.f, cfg, &args);
    try countOnly(.{ .d = undefined, .handle = @ptrFromInt(0x9000) }, cfg, &args);
    try std.testing.expectEqual(@as(u64, 5), kit.launches(2));
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try kit.launchesJson(&@as([ids.len + 1]u64, @splat(0)), &w, 0);
    try std.testing.expectEqualStrings("{\"mode\": \"count\", \"total\": 11, \"unlisted\": 1, \"by_kernel\": {\"glue.a\": 3, \"tri.b\": 7}}", w.buffered());
    // a request: the counters at its start, then strict mode's check counting 2 launches that match the metadata
    var at: [ids.len + 1]u64 = undefined;
    kit.mark(&at);
    kit.strict = true;
    for (0..2) |_| try strictCheck(entries[3].f, cfg, &args);
    w = .fixed(&buf);
    try kit.launchesJson(&at, &w, 0);
    const full = "{\"mode\": \"strict\", \"total\": 2, \"unlisted\": 0, \"by_kernel\": {\"glue.c\": 2}}";
    try std.testing.expectEqualStrings(full, w.buffered());
    // no room for the ids, or for the bytes the caller writes after the object: the totals alone
    const short = "{\"mode\": \"strict\", \"total\": 2, \"unlisted\": 0}";
    w = .fixed(buf[0 .. full.len - 1]);
    try kit.launchesJson(&at, &w, 0);
    try std.testing.expectEqualStrings(short, w.buffered());
    w = .fixed(buf[0..full.len]);
    try kit.launchesJson(&at, &w, 1);
    try std.testing.expectEqualStrings(short, w.buffered());
}

/// A kernel bound by id and specialization, with its parameters declared as `P` (kernel.zig).
pub fn Kernel(comptime id: []const u8, comptime P: type) type {
    return struct {
        const Self = @This();
        pub const kernel_id = id;
        pub const Params = P;
        pub const params = kernel_mod.table(P);

        f: Function,
        index: u32,

        /// The kit's kernel `id` with exactly `spec`, refused unless its metadata matches `P`.
        pub fn bind(kit: *const Kit, spec: []const SpecItem) !Self {
            const i = kit.find(id, spec) orelse {
                std.log.err("kit has no {s} with this specialization ({d} spec fields)", .{ id, spec.len });
                return error.MissingKernel;
            };
            var buf: [256]u8 = undefined;
            if (kernel_mod.mismatch(&params, kit.entries[i].explicit, &buf)) |why| {
                std.log.err("{s}: the Zig declaration disagrees with the compiled kernel: {s}", .{ id, why });
                return error.AbiMismatch;
            }
            return .{ .f = kit.entries[i].f, .index = i };
        }

        /// The kit's entry `i` (one specialization of `id`, as a loader finds them by walking `kit.entries`), refused
        /// unless it is `id` and its metadata matches `P`.
        pub fn bindIndex(kit: *const Kit, i: usize) !Self {
            const e = &kit.entries[i];
            if (!std.mem.eql(u8, e.k.id, id)) return error.MissingKernel;
            var buf: [256]u8 = undefined;
            if (kernel_mod.mismatch(&params, e.explicit, &buf)) |why| {
                std.log.err("{s} ({s}): the Zig declaration disagrees with the compiled kernel: {s}", .{ id, e.k.symbol, why });
                return error.AbiMismatch;
            }
            return .{ .f = e.f, .index = @intCast(i) };
        }

        pub fn launch(self: Self, stream: Stream, cfg: launch_mod.Config, p: P) !void {
            var a: launch_mod.Args = .{};
            kernel_mod.pack(P, p, &a);
            try launch_mod.launch(self.f, cfg, stream, &a);
        }
    };
}
