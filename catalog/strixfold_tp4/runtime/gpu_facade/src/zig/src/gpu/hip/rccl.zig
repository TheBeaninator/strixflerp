//! RCCL opened at run time for the HIP ranks (TP4-RCCL-COLL): the NCCL calls the family's Comm makes (comm.zig), from
//! librccl built for gfx1151 against the same HIP runtime the engine opens (TheRock 10.2). RCCL keeps NCCL's entry-point
//! names, its unique id (128 bytes) and its DataType and RedOp numbering, so this table is cuda/nccl.zig's over the HIP
//! ABI. A second HIP runtime in the process breaks it (Triton fails with HIP 709 "context is destroyed"; never
//! LD_PRELOAD one): openPath refuses a process holding two libamdhip64 files.
//!
//! TF_RCCL_LIB names the library (default librccl.so.1 on the loader path, e.g.
//! /data/models/tools/rccl-gfx1151-tr102/lib, whose RUNPATH is TheRock 10.2's lib).

const std = @import("std");
const abi = @import("abi.zig");

pub const Result = c_int;
pub const Comm = ?*opaque {};
pub const UniqueId = extern struct { internal: [128]u8 };

pub const DataType = enum(c_int) { i8 = 0, u8 = 1, i32 = 2, u32 = 3, i64 = 4, u64 = 5, f16 = 6, f32 = 7, f64 = 8, bf16 = 9 };
pub const RedOp = enum(c_int) { sum = 0, prod = 1, max = 2, min = 3, avg = 4 };

pub const Error = error{ LibraryUnavailable, MissingSymbol, NcclFailed, TwoHipRuntimes };

const R = Result;
const D = abi.DevicePtr;

pub const Api = struct {
    ncclGetVersion: *const fn (*c_int) callconv(.c) R,
    ncclGetUniqueId: *const fn (*UniqueId) callconv(.c) R,
    ncclCommInitRank: *const fn (*Comm, c_int, UniqueId, c_int) callconv(.c) R,
    ncclCommInitAll: *const fn ([*]Comm, c_int, ?[*]const c_int) callconv(.c) R,
    ncclCommDestroy: *const fn (Comm) callconv(.c) R,
    ncclGetErrorString: *const fn (R) callconv(.c) ?[*:0]const u8,
    ncclAllReduce: *const fn (D, D, usize, DataType, RedOp, Comm, abi.Stream) callconv(.c) R,
    ncclAllGather: *const fn (D, D, usize, DataType, Comm, abi.Stream) callconv(.c) R,
    ncclBroadcast: *const fn (D, D, usize, DataType, c_int, Comm, abi.Stream) callconv(.c) R,
    ncclSend: *const fn (D, usize, DataType, c_int, Comm, abi.Stream) callconv(.c) R,
    ncclRecv: *const fn (D, usize, DataType, c_int, Comm, abi.Stream) callconv(.c) R,
    ncclGroupStart: *const fn () callconv(.c) R,
    ncclGroupEnd: *const fn () callconv(.c) R,
};

pub const default_path = "librccl.so.1";

pub const Library = struct {
    /// null: a table the caller made (fake; tests)
    lib: ?std.DynLib,
    api: Api,

    /// TF_RCCL_LIB, else librccl.so.1 on the loader path.
    pub fn open() Error!Library {
        const env = std.c.getenv("TF_RCCL_LIB");
        return openPath(if (env) |p| std.mem.span(p) else default_path);
    }

    /// Resolves every field of `Api` (a missing symbol refuses the library), then holds the process to one HIP runtime.
    pub fn openPath(path: []const u8) Error!Library {
        var lib = std.DynLib.open(path) catch {
            std.log.err("{s} did not open (TF_RCCL_LIB or the loader path: the gfx1151 RCCL built against TheRock 10.2)", .{path});
            return error.LibraryUnavailable;
        };
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse {
                std.log.err("{s} has no {s}", .{ path, name });
                return error.MissingSymbol;
            };
        }
        var rt: Runtimes = .{};
        rt.scan();
        if (rt.count > 1) {
            std.log.err("{s}: {d} HIP runtimes in the process ({s}, {s}); librccl must use the engine's libamdhip64", .{ path, rt.count, rt.name(0), rt.name(1) });
            return error.TwoHipRuntimes;
        }
        return .{ .lib = lib, .api = api };
    }

    /// A table made by the caller (a host fake of the calls, for tests); close does not unload anything.
    pub fn fake(api: Api) Library {
        return .{ .lib = null, .api = api };
    }

    pub fn close(self: *Library) void {
        if (self.lib) |*l| l.close();
    }

    pub fn check(self: *const Library, r: Result, what: []const u8) Error!void {
        if (r == 0) return;
        const s = self.api.ncclGetErrorString(r);
        const text = if (s) |p| std.mem.span(p) else "?";
        // a fake's failures are a test's planned faults: a warning (the test runner fails a test on any error log)
        if (self.lib == null) std.log.warn("{s}: {s} ({d})", .{ what, text, r }) else std.log.err("{s}: {s} ({d})", .{ what, text, r });
        return error.NcclFailed;
    }

    /// RCCL's version as 10000 * major + 100 * minor + patch.
    pub fn version(self: *const Library) Error!c_int {
        var v: c_int = 0;
        try self.check(self.api.ncclGetVersion(&v), "ncclGetVersion");
        return v;
    }
};

/// The distinct libamdhip64 files mapped into this process (by real path), the first few by name.
pub const Runtimes = struct {
    count: usize = 0,
    names: [2][512]u8 = undefined,
    lens: [2]usize = .{ 0, 0 },

    pub fn name(r: *const Runtimes, i: usize) []const u8 {
        return if (i < r.count and i < 2) r.names[i][0..r.lens[i]] else "-";
    }

    pub fn scan(r: *Runtimes) void {
        r.* = .{};
        _ = std.c.dl_iterate_phdr(visit, r);
    }

    fn visit(info: *std.c.dl_phdr_info, _: usize, data: ?*anyopaque) callconv(.c) c_int {
        const r: *Runtimes = @ptrCast(@alignCast(data.?));
        const raw = info.name orelse return 0;
        const path = std.mem.span(raw);
        if (!std.mem.startsWith(u8, std.fs.path.basename(path), "libamdhip64.so")) return 0;
        var buf: [4096]u8 = undefined;
        const real = if (std.c.realpath(raw, &buf)) |p| std.mem.span(p) else path;
        for (0..@min(r.count, 2)) |i| if (std.mem.eql(u8, r.names[i][0..r.lens[i]], real[0..@min(real.len, 512)])) return 0;
        if (r.count < 2) {
            const n = @min(real.len, 512);
            @memcpy(r.names[r.count][0..n], real[0..n]);
            r.lens[r.count] = n;
        }
        r.count += 1;
        return 0;
    }
};

test "the runtime scan finds no HIP runtime in a process that opened none" {
    var rt: Runtimes = .{};
    rt.scan();
    try std.testing.expectEqual(@as(usize, 0), rt.count);
}
