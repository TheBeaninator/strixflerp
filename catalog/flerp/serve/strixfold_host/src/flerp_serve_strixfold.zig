//! The root file of a front end binary whose executor is StrixFold's (flerp.serve.build.zig): Zig owns process
//! start-up (allocator, io) and hands both to the two modules that export C functions from Zig - the executor ABI
//! adapter and the model text module - then runs the front end's own main (C++, compiled with FLERP_HOSTED) on this
//! thread, the one that drives the executor.
const std = @import("std");
const abi = @import("flerp/executor/strixfold_abi.zig");
const text = @import("flerp/text/text_strixfold.zig");

comptime { // the files' `export fn`s are the binary's C symbols
    _ = abi;
    _ = text;
}

extern fn flerp_host_main(argc: c_int, argv: [*]const [*:0]const u8) c_int;

pub fn main(init: std.process.Init) !u8 {
    abi.setProcess(init.gpa, init.io);
    text.setProcess(init.gpa, init.io);
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    const argv = try a.alloc([*:0]const u8, args.len);
    for (args, argv) |arg, *p| p.* = arg.ptr;
    // the adapter's way out: the last line, stdio flushed, then _exit. A return from here reaches libc's exit() and
    // libomp.so's fini, which faults once hipBLAS has been closed (strixfold_abi.zig, leave; card 91a9cda544)
    abi.leave(flerp_host_main(@intCast(args.len), argv.ptr));
}
