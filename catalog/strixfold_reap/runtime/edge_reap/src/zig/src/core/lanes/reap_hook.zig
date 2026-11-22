//! Process-wide REAP hooks. The production binary shares this lanes module between the
//! dsv41 family (which installs the hooks at model open) and engine_api (which the HTTP
//! server calls). A server test that does not link dsv41 leaves both function pointers null.

const std = @import("std");

pub const Query = struct {
    domain: []const u8,
    modality: []const u8,
    allocator: std.mem.Allocator,
    out: ?[]u8 = null,
    status: u16 = 200,
};

pub const QueryFn = *const fn (*anyopaque, *Query) void;
/// 0 and `index` on success. On failure, the message length written into `err`.
pub const AdmitFn = *const fn (*anyopaque, []const u8, *u32, []u8) usize;
/// Engine thread: refresh the host mirror when a query is waiting, and hand a snapshot to the writer.
pub const ServiceFn = *const fn (*anyopaque) void;

var hook_ctx: ?*anyopaque = null;
var query_fn: ?QueryFn = null;
var admit_fn: ?AdmitFn = null;
var service_fn: ?ServiceFn = null;
var mu: std.atomic.Mutex = .unlocked;

fn lock() void {
    while (!mu.tryLock()) std.Thread.yield() catch {};
}

pub fn install(c: *anyopaque, q: QueryFn, a: AdmitFn, s: ServiceFn) void {
    lock();
    defer mu.unlock();
    hook_ctx = c;
    query_fn = q;
    admit_fn = a;
    service_fn = s;
}

pub fn clear(c: *anyopaque) void {
    lock();
    defer mu.unlock();
    if (hook_ctx == c) {
        hook_ctx = null;
        query_fn = null;
        admit_fn = null;
        service_fn = null;
    }
}

/// The lane host calls this every turn, including while idle. Nothing installed: no work.
pub fn service() void {
    lock();
    const f = service_fn;
    const c = hook_ctx;
    mu.unlock();
    if (f == null or c == null) return;
    f.?(c.?);
}

/// False when nothing is installed. The caller then answers an empty bucket list.
pub fn query(q: *Query) bool {
    lock();
    const f = query_fn;
    const c = hook_ctx;
    mu.unlock();
    if (f == null or c == null) return false;
    f.?(c.?, q);
    return true;
}

/// No hook: success, index 0 (untagged traffic, and tests with no engine).
pub fn admit(domain: []const u8, index: *u32, err: []u8) usize {
    lock();
    const f = admit_fn;
    const c = hook_ctx;
    mu.unlock();
    if (f == null or c == null) {
        index.* = 0;
        return 0;
    }
    return f.?(c.?, domain, index, err);
}
