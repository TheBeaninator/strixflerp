//! The build file of a front end recipe whose executor is StrixFold's (`zig build --build-file flerp.serve.build.zig`):
//! every step of the origin's build.zig, unchanged, plus `flerp-serve`: one executable per [binaries] entry of the
//! recipe, made of
//!   - flerp_serve_strixfold.zig, the root: the executor ABI adapter and the model text module, both exported from
//!     Zig, with the module graph flerp.build.zig gives the adapter;
//!   - the C and C++ compile units of the binary's module closure - the front end, the decode loop, the modules
//!     they require - as flerp/link.zig lists them. `flerp assemble` writes that file from the module manifests
//!     ([build] c_link), so this file names no source of the front end.
//! The C++ main is compiled with FLERP_HOSTED: Zig owns process start-up and calls flerp_host_main.
const std = @import("std");
const origin = @import("build.zig");
const adapter = @import("flerp.build.zig");
const link = @import("flerp/link.zig");

pub fn build(b: *std.Build) void {
    origin.build(b);
    const g = adapter.graph(b);
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = g.target, .optimize = g.optimize, .link_libc = true });
    const text_h = b.addTranslateC(.{ .root_source_file = b.path("include/flerp/text.h"), .target = g.target, .optimize = g.optimize });
    text_h.addIncludePath(b.path("include"));
    const text_abi = text_h.createModule();
    const step = b.step("flerp-serve", "serve.flerp.http linked with the StrixFold executor ABI adapter and text module, into zig-out/flerp/bin");
    for (link.binaries) |bin| {
        const root = b.createModule(.{
            .root_source_file = b.path("flerp_serve_strixfold.zig"),
            .target = g.target,
            .optimize = g.optimize,
            .link_libc = true,
            .link_libcpp = true,
            .imports = &.{
                .{ .name = "gpu", .module = g.gpu },             .{ .name = "lanes", .module = g.lanes },
                .{ .name = "dsv41", .module = g.dsv41 },         .{ .name = "flerp_abi", .module = g.abi },
                .{ .name = "flerp_text_abi", .module = text_abi }, .{ .name = "tokenizer", .module = g.tokenizer },
                .{ .name = "template", .module = template },
            },
        });
        // include roots: the union over the units (Zig gives a module one set of include paths for its C sources)
        var seen: std.ArrayList([]const u8) = .empty;
        for (bin.units) |u| {
            for ([_][]const []const u8{ u.include, u.system }, 0..) |dirs, is_system| {
                for (dirs) |d| {
                    for (seen.items) |have| {
                        if (std.mem.eql(u8, have, d)) break;
                    } else {
                        seen.append(b.allocator, d) catch @panic("OOM");
                        if (is_system == 1) root.addSystemIncludePath(b.path(d)) else root.addIncludePath(b.path(d));
                    }
                }
            }
            var flags: std.ArrayList([]const u8) = .empty;
            flags.appendSlice(b.allocator, u.flags) catch @panic("OOM");
            flags.append(b.allocator, "-DFLERP_HOSTED") catch @panic("OOM");
            root.addCSourceFile(.{ .file = b.path(u.file), .flags = flags.items });
        }
        const exe = b.addExecutable(.{ .name = bin.name, .root_module = root });
        const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "flerp/bin" } } });
        step.dependOn(&install.step);
    }
}
