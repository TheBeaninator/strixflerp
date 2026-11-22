//! GET /stats/reap: Edge-REAP routing statistics. Same key as the /cache routes.
//! Absent (404) when REAP stats are off or --cache-transfer-key is unset.
const std = @import("std");
const api = @import("engine_api");
const http_conn = @import("http_conn.zig");
const routes = @import("routes.zig");
const Server = @import("server.zig").Server;
const Conn = http_conn.Conn;
const Allocator = std.mem.Allocator;

pub const invalid_domain = "reap_domain must match [a-z0-9_.-]{1,32}";

pub fn domainOk(s: []const u8) bool {
    if (s.len < 1 or s.len > 32) return false;
    for (s) |ch| {
        const ok = (ch >= 'a' and ch <= 'z') or (ch >= '0' and ch <= '9') or ch == '_' or ch == '.' or ch == '-';
        if (!ok) return false;
    }
    return true;
}

/// 0 and `index` on success. On failure, the message length written into `err`.
/// No hook: index 0 (untagged traffic, and a test engine that does not admit).
pub fn admit(srv: *Server, domain: []const u8, index: *u32, err: []u8) usize {
    const f = srv.engine.vtable.reap_admit orelse {
        index.* = 0;
        return 0;
    };
    return f(srv.engine.ctx, domain, index, err);
}

pub fn get(srv: *Server, conn: *Conn, a: Allocator) void {
    if (!srv.config.reap_stats) return routes.unknown(conn, a);
    const key = srv.config.transfer orelse return routes.unknown(conn, a);
    if (!key.accepts(single(conn, "Authorization"), single(conn, "x-cache-transfer-key"))) return unauthorized(conn);
    const q = queryOf(conn.path);
    var query: api.ReapQuery = .{ .domain = q.domain, .modality = q.modality, .allocator = a };
    if (srv.engine.vtable.reap) |f| {
        f(srv.engine.ctx, &query);
    } else {
        query.status = 200;
        query.out = a.dupe(u8, "{\"buckets\":[]}") catch {
            conn.sendJson(500, "{\"error\":{\"message\":\"out of memory\"}}");
            return;
        };
    }
    if (query.status == 503) {
        const body = query.out orelse "{\"error\":{\"message\":\"reap stats are refreshing\"}}";
        sendUnavailable(conn, body);
        return;
    }
    if (query.out) |body| {
        conn.sendJson(query.status, body);
        return;
    }
    const message: []const u8 = switch (query.status) {
        400 => "modality must be text, vision, or all",
        404 => "unknown reap domain",
        else => "reap export failed",
    };
    const body = std.fmt.allocPrint(a, "{{\"error\":{{\"message\":\"{s}\"}}}}", .{message}) catch return;
    conn.sendJson(query.status, body);
}

fn queryOf(path: []const u8) struct { domain: []const u8, modality: []const u8 } {
    const at = std.mem.indexOfScalar(u8, path, '?') orelse return .{ .domain = "", .modality = "" };
    var domain: []const u8 = "";
    var modality: []const u8 = "";
    var it = std.mem.splitScalar(u8, path[at + 1 ..], '&');
    while (it.next()) |pair| {
        if (std.mem.startsWith(u8, pair, "domain=")) domain = pair["domain=".len..];
        if (std.mem.startsWith(u8, pair, "modality=")) modality = pair["modality=".len..];
    }
    return .{ .domain = domain, .modality = modality };
}

fn single(conn: *const Conn, name: []const u8) ?[]const u8 {
    if (conn.headerCount(name) != 1) return null;
    return conn.header(name);
}

fn sendUnavailable(conn: *Conn, body: []const u8) void {
    conn.startResponse(503, null) catch return;
    conn.addHeader("Retry-After", "1") catch return;
    conn.addHeader("Content-Type", "application/json") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{body.len}) catch return) catch return;
    if (conn.close) conn.addHeader("Connection", "close") catch return;
    conn.finish(body) catch {};
}

fn unauthorized(conn: *Conn) void {
    const body = "{\"error\": {\"message\": \"the cache transfer key is required\"}}";
    conn.close = true;
    conn.startResponse(401, null) catch return;
    conn.addHeader("WWW-Authenticate", "Bearer") catch return;
    conn.addHeader("Content-Type", "application/json") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{body.len}) catch unreachable) catch return;
    conn.addHeader("Connection", "close") catch return;
    conn.finish(body) catch {};
}
