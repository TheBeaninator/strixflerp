//! The build file of the StrixFold ABI variant (`zig build --build-file flerp.build.zig`): every step of the origin's
//! build.zig, unchanged, plus `flerp-abi`: the executor ABI adapter over the dsv41 lane backend
//! (flerp/executor/strixfold_abi.zig) linked with the conformance driver (flerp/executor/conformance.c) and the
//! decode loop it runs (flerp/decode_loop.c).
//! The module graph below is the one zig/build/hip_dsv41.zig gives the family (its familyWith, which is not pub);
//! `graph` hands it to another build file that links the adapter (flerp.serve.build.zig).
const std = @import("std");
const origin = @import("build.zig");
const hip = @import("zig/build/hip_dsv41.zig");

pub const Graph = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    gpu: *std.Build.Module,
    tokenizer: *std.Build.Module,
    lanes: *std.Build.Module,
    dsv41: *std.Build.Module,
    abi: *std.Build.Module, // include/flerp/executor.h, translated
};

/// The Zig modules strixfold_abi.zig imports by name. Declares the option -Dflerp-rocm; call it once, after
/// origin.build.
pub fn graph(b: *std.Build) Graph {
    // origin.build declared -Drocm; an option cannot be declared twice, so the adapter's headers come from its own
    const rocm = b.option([]const u8, "flerp-rocm", "ROCm root whose HIP headers the ABI adapter's gpu module reads") orelse "/opt/rocm";
    const target = b.graph.host;
    const optimize: std.builtin.OptimizeMode = .fast; // the origin's default for the engine (build.zig)
    const o: hip.Options = .{ .arch = "gfx1151", .hipcc = "", .include = b.pathJoin(&.{ rocm, "include" }), .device_lib = "", .rocm = rocm };

    const gpu = hip.gpuModule(b, target, optimize, o);
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const dsv41 = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/dsv41.zig"), .target = target, .optimize = optimize, .link_libc = true });
    dsv41.addImport("gpu", gpu);
    dsv41.addImport("lanes", lanes);
    dsv41.addImport("core", core);
    dsv41.addImport("verbs", b.createModule(.{ .root_source_file = b.path("zig/src/fabric/verbs.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    dsv41.addIncludePath(.{ .cwd_relative = o.include });
    dsv41.addCSourceFile(.{ .file = b.path("zig/src/families/dsv41/hipblas_abi_check.c"), .flags = &.{ "-std=c11", "-D__HIP_PLATFORM_AMD__", "-Wall", "-Werror" } });
    dsv41.addAnonymousImport("dsv41_registry", .{ .root_source_file = b.path("zig/kernels/hip/dsv41/registry.json") });

    const abi = b.addTranslateC(.{ .root_source_file = b.path("include/flerp/executor.h"), .target = target, .optimize = optimize }).createModule();
    return .{ .target = target, .optimize = optimize, .gpu = gpu, .tokenizer = tokenizer, .lanes = lanes, .dsv41 = dsv41, .abi = abi };
}

pub fn build(b: *std.Build) void {
    origin.build(b);
    const g = graph(b);
    const root = b.createModule(.{
        .root_source_file = b.path("flerp/executor/strixfold_abi.zig"),
        .target = g.target,
        .optimize = g.optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "gpu", .module = g.gpu }, .{ .name = "lanes", .module = g.lanes }, .{ .name = "dsv41", .module = g.dsv41 }, .{ .name = "flerp_abi", .module = g.abi } },
    });
    root.addIncludePath(b.path("include"));
    // the driver's main becomes a function the Zig entry point calls (Zig owns process start-up: allocator, io)
    root.addCSourceFile(.{ .file = b.path("flerp/executor/conformance.c"), .flags = &.{ "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-Dmain=flerp_host_main", "-Wall" } });
    root.addCSourceFile(.{ .file = b.path("flerp/decode_loop.c"), .flags = &.{ "-std=c11", "-Wall" } });
    const exe = b.addExecutable(.{ .name = "flerp-executor-conformance.strixfold", .root_module = root });
    const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "flerp/bin" } } });
    b.step("flerp-abi", "The executor ABI adapter over the dsv41 lane backend with the conformance driver, into zig-out/flerp/bin").dependOn(&install.step);
}
