//! Per-request routing record ids and the optional client tag.
//! The server owns this so a reply test does not import the dsv41 family.

const std = @import("std");

pub const invalid_tag = "route_tag must match [A-Za-z0-9_.:/=+-]{1,160}";

pub fn tagOk(s: []const u8) bool {
    if (s.len < 1 or s.len > 160) return false;
    for (s) |ch| {
        const ok = (ch >= 'A' and ch <= 'Z') or (ch >= 'a' and ch <= 'z') or (ch >= '0' and ch <= '9') or
            ch == '_' or ch == '.' or ch == ':' or ch == '/' or ch == '=' or ch == '+' or ch == '-';
        if (!ok) return false;
    }
    return true;
}

var boot_s: std.atomic.Value(u64) = .init(0);
var seq: std.atomic.Value(u64) = .init(0);

fn bootSeconds() u64 {
    const cur = boot_s.load(.acquire);
    if (cur != 0) return cur;
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.REALTIME, &ts);
    const raw: u64 = if (ts.sec <= 0) 1 else @intCast(ts.sec);
    if (boot_s.cmpxchgStrong(0, raw, .acq_rel, .acquire)) |old| return old;
    return raw;
}

/// `rr-<boot unix seconds, hex>-<sequence>`, unique across restarts of this process.
pub fn formatId(buf: []u8) []const u8 {
    const n = seq.fetchAdd(1, .monotonic) + 1;
    return std.fmt.bufPrint(buf, "rr-{x}-{d}", .{ bootSeconds(), n }) catch "rr-0-1";
}
