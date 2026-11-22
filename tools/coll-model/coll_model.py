#!/usr/bin/env python3
"""COLL-MODEL: an exhaustive explicit-state model checker for StrixFold's proxy slot protocol.

Models, at fork commit 003f679c28b1:
  world 2: tcp_proxy.zig (TF_COLL_KIND=proxy) and rdma_proxy.zig RdmaProxy + rdma_core.zig Core (kind=rdma), with the
           coll.hip stage/collect kernels, the second proxy thread (TF_COLL_PROXY_STRIPE_KIB > 0, tcp_proxy.Second) and
           tcp_host gathers larger than a slot over two links with one or two wire threads (TF_COLL_WIRE_THREADS).
  world 4: rdma_proxy.zig RdmaProxyN + rdma_core.zig CoreN/Layout (TP4-RDMA-N, lanes TF_COLL_LANES), the deployed TP4
           transport (tp4-serve1*.conf: TRANSPORT=rdma, WORLD=4).
Properties:
  S1  no RX slot region is overwritten while it holds a generation its collect (or, for trailers, the proxy's check)
      has not finished reading, on a rank that has not failed. S1-TX: no transmit source is overwritten before the
      NIC/socket has read it.
  S2  every collect copies exactly gather q's bytes (generation, writer rank and call all as expected).
  S3  remote writes stay inside the peer's registered RX region, which excludes the slab header (static arithmetic
      check over a parameter grid, s3_check).
  D1  with all ranks live and no deadline firing, every reachable terminal state has every rank finished and healthy
      (no deadlock, no spurious failure).
  T1  with one rank dying or wedging at any point and every deadline allowed to fire wherever the code has one, every
      terminal state has every live rank finished; a live rank whose calls did not all complete reports PeerLost or
      Timeout (never OK, never silence).
Exhaustive: breadth-first over every interleaving of the atomic steps below; a run that hits --max-states is reported
as NOT EXHAUSTED, never as a pass. Pure Python, CPU only (run under nice 19).
"""
import argparse
import json
import sys
import time
from collections import deque

CODES = {0: "OK", 1: "PeerLost", 2: "Timeout", 3: "Mismatch", 4: "Aborted", 5: "CollectiveFailed"}


def tset(t, i, v):
    return t[:i] + (v,) + t[i + 1:]


class Violation(Exception):
    def __init__(self, prop, msg):
        super().__init__(prop + ": " + msg)
        self.prop = prop
        self.msg = msg


# ------------------------------------------------------------------------------------------------ world 2 (two ranks)
# Rank fields. Every name maps to code in the CORRESPONDENCE table of docs/experiments/coll-model.md.
W2F = ["pc", "sub", "cq", "seq", "abort", "failed", "ready", "done", "cons", "sinfo", "tx", "txrd", "rx",
       "px", "sx", "hseq", "hph", "st", "lost", "out1", "out2", "hout0", "hout1", "unack1", "unack2",
       "scq1", "rcq1", "scq2", "rcq2", "shut", "qperr", "ahead"]
for _i, _n in enumerate(W2F):
    globals()["F_" + _n.upper()] = _i
EMPTY_RX = (0, -1, -1, 0)  # (generation, call, writer rank, pending: not yet read by this rank's collect)


class World2:
    """Two ranks of tcp_proxy (transport='tcp') or rdma_proxy RdmaProxy (transport='rdma').

    cfg: nslot, calls (script length), kinds (subset of 'SPH': S slot gather, P slot gather of at least the stripe size,
    H gather larger than a slot: tcp_host over two links), stripe (0/1: TF_COLL_PROXY_STRIPE_KIB > 0, a second proxy
    thread), wire (1/2: TF_COLL_WIRE_THREADS), mode ('D1' or 'T1'), mut (set of mutation names, see MUTATIONS)."""

    def __init__(self, transport, nslot, calls, kinds, stripe, wire, mode, mut=()):
        self.tp = transport
        self.N = nslot
        self.W = calls
        self.kinds = kinds
        self.stripe = stripe
        self.wire = wire
        self.mode = mode
        self.mut = set(mut)
        self.por = True
        self.level = "full"

    EAGER_ALWAYS = ("NIC: ack", "reads seq =")
    EAGER_D1 = ("polls its send CQ", "polls its receive CQ", "proxy: done[", "both completions, trailer checked",
                "caller sends part", "helper sends part", "post part 1 to the helper", "complete", "second thread: done =")

    # T1: only steps whose effect is unobservable once their own rank fails or stops (canonical form clears the proxy's
    # loop state and queues) and that commute with a fault of the other rank; host sends are not among them
    EAGER_T1 = ("polls its send CQ", "polls its receive CQ", "proxy: done[", "both completions, trailer checked",
                "post part 1 to the helper", "second thread: done =")

    def is_eager(self, lab):
        if any(x in lab for x in self.EAGER_ALWAYS):
            return True
        return any(x in lab for x in (self.EAGER_D1 if self.mode == "D1" else self.EAGER_T1))

    # -- state ------------------------------------------------------------------------------------------------------
    def initial(self):
        N = self.N
        z = (0,) * N
        r = [None] * len(W2F)
        r[F_PC] = 0
        r[F_SUB] = 0
        r[F_CQ] = 0
        r[F_SEQ] = 0
        r[F_ABORT] = 0
        r[F_FAILED] = 0
        r[F_READY] = z
        r[F_DONE] = z
        r[F_CONS] = z
        r[F_SINFO] = (-1,) * N
        r[F_TX] = ((0, 0, 0, 0),) * N  # data half 0, half 1, trailer half 0, trailer half 1 (generations)
        r[F_TXRD] = z  # NIC reads of TX[s] in flight (posted, not yet placed at the peer)
        r[F_RX] = ((EMPTY_RX, EMPTY_RX, (0, -1), (0, -1)),) * N
        r[F_PX] = (1, 0, 0, 0, 0)  # q, phase, flag a, flag b, split
        r[F_SX] = (0, 0, 0, 0, 0, 1, 0, -1, 0)  # todo, last, phase, flag a, flag b, wire seq, done, call, slot
        r[F_HSEQ] = 1
        r[F_HPH] = (0, 0, 0, 0, 0)  # sent0, recvd0, sent1, recvd1, helper job posted
        r[F_ST] = "L"
        r[F_LOST] = 0
        for f in (F_OUT1, F_OUT2, F_HOUT0, F_HOUT1, F_UNACK1, F_UNACK2, F_SCQ1, F_RCQ1, F_SCQ2, F_RCQ2):
            r[f] = ()
        r[F_SHUT] = 0
        r[F_QPERR] = 0
        r[F_AHEAD] = 0
        r = tuple(r)
        if self.tp == "tcp":
            r = tset(r, F_PX, (1, 0, 0, 0, 0, 1))  # + wire seq
        return ((), (r, r), 0)

    def describe(self, s):
        script, ranks, faults = s
        out = {"script": "".join(script)}
        for k, r in enumerate(ranks):
            d = {"st": r[F_ST], "call": r[F_PC], "op": ("stage/host", "collect")[r[F_SUB]], "seq": r[F_SEQ],
                 "ready": r[F_READY], "done": r[F_DONE], "consumed": r[F_CONS], "abort": r[F_ABORT],
                 "failed": CODES[r[F_FAILED]], "proxy": r[F_PX], "rx_gen": tuple((x[0][0], x[1][0]) for x in r[F_RX])}
            if self.stripe:
                d["second"] = r[F_SX]
            if self.tp == "rdma":
                d["posted"] = r[F_OUT1] + r[F_OUT2]
            else:
                d["wire1_out"] = r[F_OUT1]
            out["rank%d" % k] = d
        return out

    # -- helpers ----------------------------------------------------------------------------------------------------
    def kind(self, script, c):
        return script[c]

    def ops(self, script, c):
        return ("STAGE", "COLLECT") if script[c] in "SP" else ("HOST",)

    def live(self, r):
        return r[F_ST] == "L"

    def fail(self, R, k, code):
        """tcp_proxy Core.fail / rdma_core Core.fail (raiseAbort) then the thread's poison: failed (first wins)."""
        r = R[k]
        if r[F_FAILED] == 0:
            r[F_FAILED] = code
        if r[F_ABORT] == 0:
            r[F_ABORT] = 1

    def rx_write(self, R, k, s, halves, gen, call, writer):
        """A write into rank k's RX[s] (a peer's NIC placement, or k's own recvmsg): S1 check, then the new tags."""
        r = R[k]
        slot = list(r[F_RX][s])
        for h in halves:
            old = slot[h]
            if old[3] and old[0] != gen and r[F_FAILED] == 0 and r[F_ABORT] == 0:
                raise Violation("S1", "rank %d RX[%d] half %d still holds gather %d (call %d) for its collect when gather %d (call %d) from rank %d lands" % (k, s, h, old[0], old[1], gen, call, writer))
            slot[h] = (gen, call, writer, 1)
        r[F_RX] = tset(r[F_RX], s, tuple(slot))

    def choose(self, script, R):
        """The shared script grows lazily: both ranks issue the same calls (coll.zig contract). Entries every rank has
        finished are forgotten ('*'): nothing in the state refers to their kind any more."""
        lo = min(r[F_PC] for r in R)
        if any(r[F_ABORT] or r[F_FAILED] or r[F_ST] != "L" for r in R):
            return script  # a rank's GPU may have skipped ahead of its proxy: keep every call's kind
        if any(c != "*" for c in script[:lo]):
            script = tuple("*" if i < lo else c for i, c in enumerate(script))
        return script

    # -- transitions ------------------------------------------------------------------------------------------------
    def successors(self, s):
        script, ranks, faults = s
        self._cur_script = script
        self._faults = faults
        out = []
        need = max(r[F_PC] + 1 + (1 if "stream" in self.mut else 0) for r in ranks)
        need = min(need, self.W)
        if len(script) < need:
            for kd in self.kinds:
                out.append(("script call %d = %s" % (len(script), kd), (script + (kd,), ranks, faults)))
            return out
        cands = []
        for k in range(2):
            self.gpu_steps(script, ranks, k, cands)
            self.proxy_steps(script, ranks, k, cands)
            if self.stripe:
                self.second_steps(script, ranks, k, cands)
            if self.tp == "rdma":
                self.nic_steps(script, ranks, k, cands)
            self.fail_steps(script, ranks, k, cands)
            if self.mode == "T1" and faults == 0 and ranks[k][F_ST] == "L":
                cands.append(("rank %d dies" % k, lambda R, k=k: self.die(R, k)))
                cands.append(("rank %d wedges" % k, lambda R, k=k: self.wedge(R, k)))
                cands.append(("rank %d: a deadline fires (Timeout)" % k, lambda R, k=k: self.spont(R, k)))
        if self.por:
            # partial-order reduction (persistent sets of size one): a step that commutes with every other step and is
            # invisible to S1/S2 is taken alone (see EAGER_ALWAYS / EAGER_D1 and the report's "Reduction" section)
            for c in cands:
                if (len(c) == 3 and c[2]) or self.is_eager(c[0]):
                    cands = [c]
                    break
        for c in cands:
            lab, fn = c[0], c[1]
            R = [list(r) for r in ranks]
            try:
                f2 = fn(R)
            except Violation as v:
                out.append((lab, v))
                continue
            nf = faults + (1 if f2 == "fault" else 0)
            for r in R:
                # history the protocol can no longer read (which call an already-collected generation carried; the
                # site word of a slot the proxy has finished) is erased so equal futures merge (canonical form)
                # (a region nobody will read again holds nothing that matters: every later check compares against
                # the generation it expects, which a cleared region never equals)
                rx = r[F_RX]
                nrx = tuple((EMPTY_RX if not a[3] else a, EMPTY_RX if not b[3] else b,
                             (0, -1) if not a[3] else t0, (0, -1) if not b[3] else t1) for (a, b, t0, t1) in rx)
                if nrx != rx:
                    r[F_RX] = nrx
                pq = r[F_PX][0]
                tx = r[F_TX]
                ntx = tuple((0, 0, 0, 0) if (r[F_READY][x] < pq and not r[F_TXRD][x]) else tx[x] for x in range(self.N))
                if ntx != tx:
                    r[F_TX] = ntx
                si = r[F_SINFO]
                nsi = tuple(-1 if r[F_READY][x] < pq else si[x] for x in range(self.N))
                if nsi != si:
                    r[F_SINFO] = nsi
                sx = r[F_SX]
                if sx[2] == 0 and sx[0] == sx[1] and (r[F_PX][1] == 0 or not r[F_PX][4]) and sx[0] != 0:
                    # an idle second thread whose last gather the first thread no longer waits for: only "todo !=
                    # last" and "done == q" read these words, and the next hand writes todo afresh
                    r[F_SX] = (0, 0, 0, 0, 0, sx[5], 0, -1, 0)
                if r[F_FAILED] or r[F_ST] != "L":
                    # a failed rank's proxy threads have returned and a wedged/dead rank's never run again: their
                    # completion queues and loop state are read by nothing (canonical form)
                    r[F_SCQ1] = r[F_RCQ1] = r[F_SCQ2] = r[F_RCQ2] = ()
                    r[F_PX] = (0,) * len(r[F_PX])
                    r[F_SX] = (0,) * len(r[F_SX])
            self.rebase(R)
            R = tuple(tuple(r) for r in R)
            out.append((lab, (self.choose(script, R), R, nf)))
        return out

    def rebase(self, R):
        """The second cable's and tcp_host's wire sequence numbers are only ever compared for equality between the
        ranks (tcp_wire header check): both ranks' counters and the headers in flight shift by a common base."""
        b2 = min(r[F_SX][5] for r in R)
        if b2 > 1:
            for r in R:
                sx = r[F_SX]
                r[F_SX] = sx[:5] + (sx[5] - b2 + 1,) + sx[6:]
                if self.tp == "tcp":
                    r[F_OUT2] = tuple((m[0], m[1] - b2 + 1) + m[2:] for m in r[F_OUT2])
        bh = min(r[F_HSEQ] for r in R)
        if bh > 1:
            for r in R:
                r[F_HSEQ] = r[F_HSEQ] - bh + 1
                r[F_HOUT0] = tuple((m[0], m[1] - bh + 1) + m[2:] for m in r[F_HOUT0])
                r[F_HOUT1] = tuple((m[0], m[1] - bh + 1) + m[2:] for m in r[F_HOUT1])

    def die(self, R, k):
        R[k][F_ST] = "D"
        R[k][F_QPERR] = 1
        R[k][F_SHUT] = 1
        for f in (F_OUT1, F_OUT2, F_UNACK1, F_UNACK2):
            R[k][f] = ()
        return "fault"

    def spont(self, R, k):
        self.fail(R, k, 2)
        return "fault"

    def wedge(self, R, k):
        R[k][F_ST] = "W"
        return "fault"

    def fail_steps(self, script, ranks, k, cands):
        """The rest of poison after the failure code: QP to error (rdma: posted writes flush) and the sockets shut
        (tcp_host's and the proxy's wires: the peer's next read is end of stream)."""
        r = ranks[k]
        if r[F_ST] != "L" or (r[F_FAILED] == 0) or (r[F_SHUT] and r[F_QPERR]):
            return

        def fin(R):
            R[k][F_QPERR] = 1
            R[k][F_SHUT] = 1
            for f in (F_OUT1, F_OUT2, F_UNACK1, F_UNACK2):
                R[k][f] = ()
        cands.append(("rank %d poison: QP to error, sockets shut" % k, fin))

    # GPU: coll.hip tf_ds_coll_stage / tf_ds_coll_collect, and tcp_host's begin/end for H calls, in stream order.
    def gpu_steps(self, script, ranks, k, cands):
        r = ranks[k]
        if r[F_ST] != "L" or r[F_PC] >= self.W:
            return
        c = r[F_PC]
        if r[F_ABORT] or r[F_FAILED]:
            # the abort word is sticky: every kernel still queued returns without touching data (coll.hip
            # tf_ds_coll_aborted) and every later begin/end/check returns the error (status()); one step for all
            def rest(R):
                R[k][F_PC] = self.W
                R[k][F_SUB] = 0
                R[k][F_CQ] = 0
                R[k][F_LOST] = 1
            cands.append(("rank %d: abort word set, the remaining kernels return untouched, the host gets the error" % k, rest))
            return
        op = self.ops(script, c)[r[F_SUB]]
        N = self.N
        T1 = self.mode == "T1" and self._faults > 0
        if op == "STAGE":
            q = r[F_SEQ] + 1
            s = q % N
            gate = q <= N or r[F_CONS][s] + N >= q or "nogate" in self.mut
            if r[F_ABORT]:
                cands.append(("rank %d stage call %d: abort word set, returns without staging" % (k, c), lambda R: self.skip(R, k)))
            elif gate:
                cands.append(("rank %d stage call %d: gather %d into TX[%d], ready[%d] = %d" % (k, c, q, s, s, q), lambda R: self.stage(R, k, c, q, s), self.mode == "D1" and not r[F_TXRD][s]))
            elif T1:
                cands.append(("rank %d stage call %d: consumed wait past its deadline, abort = 2" % (k, c), lambda R: self.dev_timeout(R, k)))
        elif op == "COLLECT":
            if r[F_CQ] == 0:
                # thread 0 reads seq once (the last stage on this stream), then waits on that slot
                if r[F_ABORT]:
                    cands.append(("rank %d collect call %d: abort word set" % (k, c), lambda R: self.skip(R, k)))
                else:
                    cands.append(("rank %d collect call %d reads seq = %d" % (k, c, r[F_SEQ]), lambda R: R[k].__setitem__(F_CQ, R[k][F_SEQ])))
                if "stream" in self.mut and not r[F_AHEAD] and c + 1 < self.W and script[c + 1] in "SP":
                    self.stream_ahead(script, ranks, k, c, cands)
                return
            q = r[F_CQ]
            s = q % N
            d = r[F_DONE][s]
            if r[F_ABORT]:
                cands.append(("rank %d collect call %d: abort word set" % (k, c), lambda R: self.skip(R, k)))
            elif d == q:
                # (as World4.gpu_steps: the peer writes q + N into RX[s] only after this rank's write of q + N - 1,
                # which needs a stage later in this stream; not with one slot, not under the stream control)
                eager = self.mode == "D1" and self.level == "full" and "stream" not in self.mut and r[F_SEQ] < q + N - 1
                cands.append(("rank %d collect call %d: done[%d] = %d, copies RX[%d], consumed[%d] = %d" % (k, c, s, q, s, s, q), lambda R: self.collect(R, k, c, q, s, script), eager))
            elif d > q:
                cands.append(("rank %d collect call %d: done[%d] = %d past %d, abort = 2" % (k, c, s, d, q), lambda R: self.dev_timeout(R, k)))
            elif T1:
                cands.append(("rank %d collect call %d: done wait past its deadline, abort = 2" % (k, c), lambda R: self.dev_timeout(R, k)))
            if "stream" in self.mut and not r[F_AHEAD] and c + 1 < self.W and script[c + 1] in "SP":
                self.stream_ahead(script, ranks, k, c, cands)
        else:
            self.host_steps(script, ranks, k, c, cands)

    def stream_ahead(self, script, ranks, k, c, cands):
        """MUTATION stream: the next gather's stage on another stream runs before this collect finished (what
        gatherEnd's same-stream check, rdma_proxy.zig:156, refuses)."""
        r = ranks[k]
        q = r[F_SEQ] + 1
        s = q % self.N
        if r[F_ABORT] or not (q <= self.N or r[F_CONS][s] + self.N >= q):
            return

        def go(R):
            R[k][F_AHEAD] = 1
            self.stage(R, k, c + 1, q, s, advance=False)
        cands.append(("rank %d MUTATION stage call %d ahead of collect call %d (other stream)" % (k, c + 1, c), go))

    def skip(self, R, k):
        R[k][F_LOST] = 1
        self.next_op(R, k)

    def next_op(self, R, k, script=None):
        r = R[k]
        r[F_CQ] = 0
        if r[F_SUB] == 0 and self._slot_call(R, k):
            r[F_SUB] = 1
        else:
            r[F_SUB] = 0
            r[F_PC] += 1
            if r[F_AHEAD]:
                r[F_AHEAD] = 0
                r[F_SUB] = 1  # its stage already ran (mutation stream)

    def _slot_call(self, R, k):
        return self._cur_script[R[k][F_PC]] in "SP"

    def dev_timeout(self, R, k):
        """tf_ds_coll_fail: abort = 2 unless set; the kernel returns without touching data."""
        if R[k][F_ABORT] == 0:
            R[k][F_ABORT] = 2
        self.skip(R, k)

    def stage(self, R, k, c, q, s, advance=True):
        r = R[k]
        if r[F_TXRD][s] and r[F_FAILED] == 0:
            raise Violation("S1-TX", "rank %d stage of gather %d rewrites TX[%d] while the NIC still reads gather %d from it" % (k, q, s, r[F_TX][s][0]))
        r[F_TX] = tset(r[F_TX], s, (q, q, 0, 0))
        r[F_SINFO] = tset(r[F_SINFO], s, c)
        r[F_SEQ] = q
        r[F_READY] = tset(r[F_READY], s, q)
        if advance:
            self.next_op(R, k)

    def collect(self, R, k, c, q, s, script):
        r = R[k]
        peer = 1 - k
        slot = list(r[F_RX][s])
        for h in (0, 1):
            g = slot[h]
            if g[0] != q or g[1] != c or g[2] != peer:
                raise Violation("S2", "rank %d collect of call %d (gather %d) copies RX[%d] half %d holding gather %d of call %d written by rank %d" % (k, c, q, s, h, g[0], g[1], g[2]))
            slot[h] = g[:3] + (0,)
        r[F_RX] = tset(r[F_RX], s, tuple(slot))
        r[F_CONS] = tset(r[F_CONS], s, q)
        self.next_op(R, k)

    # tcp_host (tcp_coll.zig / tcp_wire.zig): a gather larger than a slot, two links (striped), one or two wire threads.
    def host_steps(self, script, ranks, k, c, cands):
        r = ranks[k]
        p = 1 - k
        pr = ranks[p]
        h = r[F_HPH]
        T1 = self.mode == "T1" and self._faults > 0
        if h == (0, 0, 0, 0, 0) and (r[F_FAILED] or r[F_ABORT]):
            cands.append(("rank %d host gather call %d: status() already failed, returns the error" % (k, c), lambda R: self.skip(R, k)))
            return
        if r[F_SHUT]:
            cands.append(("rank %d host gather call %d: its wire was aborted (poison)" % (k, c), lambda R: self.skip(R, k)))
            return
        if self.wire == 2 and not h[4]:
            cands.append(("rank %d host call %d: post part 1 to the helper thread (runBoth)" % (k, c), lambda R: R[k].__setitem__(F_HPH, tset(R[k][F_HPH], 4, 1))))
        if h[0] and h[1] and h[2] and h[3]:
            def fin(R):
                R[k][F_HSEQ] += 1
                R[k][F_HPH] = (0, 0, 0, 0, 0)
                self.next_op(R, k)
            cands.append(("rank %d host gather call %d complete" % (k, c), fin))
            return
        for part in (0, 1):
            who = "helper" if (part == 1 and self.wire == 2) else "caller"
            if part == 1 and self.wire == 2 and not h[4]:
                continue
            si, ri = 2 * part, 2 * part + 1
            fo = F_HOUT0 if part == 0 else F_HOUT1
            if not h[si]:
                def send(R, si=si, fo=fo, part=part):
                    R[k][fo] = R[k][fo] + (("H", r[F_HSEQ], c, part),)
                    R[k][F_HPH] = tset(R[k][F_HPH], si, 1)
                cands.append(("rank %d host call %d %s sends part %d" % (k, c, who, part), send))
            if not h[ri]:
                if pr[fo]:
                    def recv(R, ri=ri, fo=fo, part=part):
                        m = R[p][fo][0]
                        R[p][fo] = R[p][fo][1:]
                        if m != ("H", R[k][F_HSEQ], c, part):
                            self.fail(R, k, 3)
                            self.skip(R, k)
                            R[k][F_HPH] = (0, 0, 0, 0, 0)
                            return
                        R[k][F_HPH] = tset(R[k][F_HPH], ri, 1)
                    cands.append(("rank %d host call %d %s receives part %d" % (k, c, who, part), recv))
                else:
                    if pr[F_SHUT]:
                        cands.append(("rank %d host call %d part %d: end of stream (PeerLost)" % (k, c, part), lambda R: self.host_fail(R, k, 1)))
                    elif T1:
                        cands.append(("rank %d host call %d part %d: no progress within the deadline (Timeout)" % (k, c, part), lambda R: self.host_fail(R, k, 2)))

    def host_fail(self, R, k, code):
        self.fail(R, k, code)
        R[k][F_HPH] = (0, 0, 0, 0, 0)
        self.skip(R, k)

    # proxy thread 1: tcp_proxy.zig Core.run / rdma_core.zig Core.run
    def proxy_steps(self, script, ranks, k, cands):
        r = ranks[k]
        if r[F_ST] != "L" or r[F_FAILED]:
            return
        if self.tp == "tcp":
            self.tcp_proxy(script, ranks, k, cands)
        else:
            self.rdma_proxy(script, ranks, k, cands)

    def tcp_proxy(self, script, ranks, k, cands):
        r = ranks[k]
        p = 1 - k
        pr = ranks[p]
        N = self.N
        T1 = self.mode == "T1" and self._faults > 0
        q, ph, sent, recvd, split, wseq = r[F_PX]
        s = q % N
        if ph == 0:
            rd = r[F_READY][s]
            if rd == q:
                c = r[F_SINFO][s]
                sp = 1 if (self.stripe and script[c] == "P") else 0

                def start(R):
                    R[k][F_PX] = (q, 1, 0, 0, sp, wseq)
                    if sp:  # Second.hand: off, len, site, then todo (release)
                        sx = R[k][F_SX]
                        R[k][F_SX] = (q, sx[1], sx[2], sx[3], sx[4], sx[5], sx[6], c, s)
                cands.append(("rank %d proxy sees ready[%d] = %d (call %d)%s" % (k, s, q, c, ", hands half 1 to the second thread" if sp else ""), start, self.mode == "D1"))
            elif rd > q:
                cands.append(("rank %d proxy: ready[%d] = %d ahead of %d (Mismatch)" % (k, s, rd, q), lambda R: self.fail(R, k, 3)))
            elif r[F_ABORT]:
                cands.append(("rank %d proxy idle: abort word set (deviceGaveUp)" % k, lambda R: self.fail(R, k, 2)))
            return
        c = r[F_SINFO][s]
        halves = (0,) if split else (0, 1)
        if not sent:
            def send(R):
                tx = R[k][F_TX][s]
                for h in halves:
                    if tx[h] != q:
                        raise Violation("S1-TX", "rank %d sends gather %d from TX[%d] half %d, which already holds gather %d" % (k, q, s, h, tx[h]))
                R[k][F_OUT1] = R[k][F_OUT1] + (("P", wseq, c, len(halves), q),)
                R[k][F_PX] = (q, 1, 1, recvd, split, wseq)
            cands.append(("rank %d proxy sends gather %d (call %d) on cable 1" % (k, q, c), send, self.mode == "D1"))
        if not recvd:
            if pr[F_OUT1]:
                def recv(R):
                    m = R[p][F_OUT1][0]
                    R[p][F_OUT1] = R[p][F_OUT1][1:]
                    if m[:4] != ("P", wseq, c, len(halves)):
                        self.fail(R, k, 3)  # tcp_wire header check: Mismatch
                        return
                    self.rx_write(R, k, s, halves, m[4], m[2], p)
                    R[k][F_PX] = (q, 1, R[k][F_PX][2], 1, split, wseq)
                cands.append(("rank %d proxy receives the peer's gather into RX[%d]" % (k, s), recv, self.mode == "D1" and not any(r[F_RX][s][h][3] for h in halves)))
            elif pr[F_SHUT]:
                cands.append(("rank %d proxy cable 1: end of stream (PeerLost)" % k, lambda R: self.fail(R, k, 1)))
            elif T1:
                cands.append(("rank %d proxy cable 1: no progress within the deadline (Timeout)" % k, lambda R: self.fail(R, k, 2)))
        if sent and recvd:
            if split and "nowait" not in self.mut and r[F_SX][6] != q:
                if r[F_ABORT]:
                    cands.append(("rank %d proxy waiting for the second cable: abort word set" % k, lambda R: self.fail(R, k, 2)))
                return

            def fin(R):
                R[k][F_DONE] = tset(R[k][F_DONE], s, q)
                R[k][F_PX] = (q + 1, 0, 0, 0, 0, wseq + 1)
            cands.append(("rank %d proxy: done[%d] = %d" % (k, s, q), fin))

    def second_steps(self, script, ranks, k, cands):
        """tcp_proxy.Second.run (and, for rdma, the hypothetical same thread over a second QP)."""
        r = ranks[k]
        if r[F_ST] != "L" or r[F_FAILED]:
            return
        p = 1 - k
        pr = ranks[p]
        T1 = self.mode == "T1" and self._faults > 0
        todo, last, ph, a, b, wseq2, sdone, c, s = r[F_SX]
        if ph == 0:
            if todo != last:
                def start(R):
                    if self.tp == "rdma":
                        R[k][F_OUT2] = R[k][F_OUT2] + ((todo, s, c, (1,)),)
                        R[k][F_TXRD] = tset(R[k][F_TXRD], s, R[k][F_TXRD][s] + 1)
                    R[k][F_SX] = (todo, last, 1, 0, 0, wseq2, sdone, c, s)
                cands.append(("rank %d second thread starts half 1 of gather %d" % (k, todo), start))
            elif r[F_ABORT]:
                cands.append(("rank %d second thread idle: abort word set" % k, lambda R: self.fail(R, k, 2)))
            return
        if self.tp == "tcp":
            if not a:
                def send(R):
                    if R[k][F_TX][s][1] != todo:
                        raise Violation("S1-TX", "rank %d second thread sends half 1 of gather %d from TX[%d], which holds gather %d" % (k, todo, s, R[k][F_TX][s][1]))
                    R[k][F_OUT2] = R[k][F_OUT2] + (("S", wseq2, c, 1, todo),)
                    R[k][F_SX] = (todo, last, 1, 1, R[k][F_SX][4], wseq2, sdone, c, s)
                cands.append(("rank %d second thread sends half 1 of gather %d on cable 2" % (k, todo), send, self.mode == "D1"))
            if not b:
                if pr[F_OUT2]:
                    def recv(R):
                        m = R[p][F_OUT2][0]
                        R[p][F_OUT2] = R[p][F_OUT2][1:]
                        if m[:4] != ("S", wseq2, c, 1):
                            self.fail(R, k, 3)
                            return
                        self.rx_write(R, k, s, (1,), m[4], m[2], p)
                        R[k][F_SX] = (todo, last, 1, R[k][F_SX][3], 1, wseq2, sdone, c, s)
                    cands.append(("rank %d second thread receives half 1 into RX[%d]" % (k, s), recv, self.mode == "D1" and not r[F_RX][s][1][3]))
                elif pr[F_SHUT]:
                    cands.append(("rank %d second thread cable 2: end of stream (PeerLost)" % k, lambda R: self.fail(R, k, 1)))
                elif T1:
                    cands.append(("rank %d second thread cable 2: Timeout" % k, lambda R: self.fail(R, k, 2)))
            if a and b:
                cands.append(("rank %d second thread: done = %d" % (k, todo), lambda R: R[k].__setitem__(F_SX, (todo, todo, 0, 0, 0, wseq2 + 1, todo, c, s))))
        else:
            self.rdma_wait(ranks, k, cands, which=2)

    def rdma_proxy(self, script, ranks, k, cands):
        r = ranks[k]
        N = self.N
        q, ph, hs, hr, split = r[F_PX]
        s = q % N
        if ph == 0:
            rd = r[F_READY][s]
            if rd == q:
                c = r[F_SINFO][s]
                sp = 1 if (self.stripe and script[c] == "P") else 0

                def start(R):
                    rr = R[k]
                    if rr[F_TXRD][s]:
                        raise Violation("S1-TX", "rank %d proxy writes gather %d's trailer into TX[%d] while the NIC still reads it" % (k, q, s))
                    tx = rr[F_TX][s]
                    rr[F_TX] = tset(rr[F_TX], s, (tx[0], tx[1], q, q if sp else tx[3]))
                    rr[F_OUT1] = rr[F_OUT1] + ((q, s, c, (0,) if sp else (0, 1)),)
                    rr[F_TXRD] = tset(rr[F_TXRD], s, rr[F_TXRD][s] + 1)
                    rr[F_PX] = (q, 1, 0, 0, sp)
                    if sp:
                        sx = rr[F_SX]
                        rr[F_SX] = (q, sx[1], sx[2], sx[3], sx[4], sx[5], sx[6], c, s)
                cands.append(("rank %d proxy sees ready[%d] = %d (call %d): trailer, posts the write with immediate%s" % (k, s, q, c, " (half 0; half 1 to the second thread)" if sp else ""), start, self.mode == "D1" and not r[F_TXRD][s]))
            elif rd > q:
                cands.append(("rank %d proxy: ready[%d] = %d ahead of %d (Mismatch)" % (k, s, rd, q), lambda R: self.fail(R, k, 3)))
            elif r[F_ABORT]:
                cands.append(("rank %d proxy idle: abort word set (deviceGaveUp)" % k, lambda R: self.fail(R, k, 2)))
            return
        self.rdma_wait(ranks, k, cands, which=1)

    def rdma_wait(self, ranks, k, cands, which):
        """rdma_core.zig Core.completions (thread 1, QP 1) or the hypothetical second thread on QP 2."""
        r = ranks[k]
        p = 1 - k
        pr = ranks[p]
        T1 = self.mode == "T1" and self._faults > 0
        if which == 1:
            q, ph, hs, hr, split = r[F_PX]
            s = q % self.N
            c = r[F_SINFO][s]
            scq, rcq, half_set, trl = F_SCQ1, F_RCQ1, ((0,) if split else (0, 1)), 2
        else:
            todo, last, ph, hs, hr, wseq2, sdone, c, s = r[F_SX]
            q = todo
            scq, rcq, half_set, trl = F_SCQ2, F_RCQ2, (1,), 3

        def setflags(R, a=None, b=None):
            if which == 1:
                x = R[k][F_PX]
                R[k][F_PX] = (x[0], x[1], x[2] if a is None else a, x[3] if b is None else b, x[4])
            else:
                x = R[k][F_SX]
                R[k][F_SX] = x[:3] + (x[3] if a is None else a, x[4] if b is None else b) + x[5:]
        if not hs and r[scq]:
            def ps(R):
                wr, st = R[k][scq][0]
                R[k][scq] = R[k][scq][1:]
                if st:
                    self.fail(R, k, st)
                    return
                if wr == q:
                    setflags(R, a=1)
            cands.append(("rank %d proxy%s polls its send CQ" % (k, "" if which == 1 else " 2"), ps))
        if not hr and r[rcq]:
            def pr_(R):
                imm, ln, st = R[k][rcq][0]
                R[k][rcq] = R[k][rcq][1:]
                if st:
                    self.fail(R, k, st)
                    return
                if imm != q or ln != c:
                    self.fail(R, k, 3)
                    return
                setflags(R, b=1)
            cands.append(("rank %d proxy%s polls its receive CQ" % (k, "" if which == 1 else " 2"), pr_))
        if hs and hr:
            if which == 1 and split and "nowait" not in self.mut and r[F_SX][6] != q:
                if r[F_ABORT]:
                    cands.append(("rank %d proxy waiting for QP 2's half: abort word set" % k, lambda R: self.fail(R, k, 2)))
                return

            def fin(R):
                tr = R[k][F_RX][s][trl]
                if tr[0] != q or tr[1] != c:
                    self.fail(R, k, 3)  # trailer mismatch
                    return
                if which == 1:
                    R[k][F_DONE] = tset(R[k][F_DONE], s, q)
                    R[k][F_PX] = (q + 1, 0, 0, 0, 0)
                else:
                    R[k][F_SX] = (q, q, 0, 0, 0, R[k][F_SX][5], q, c, s)
            cands.append(("rank %d proxy%s: both completions, trailer checked, %s" % (k, "" if which == 1 else " 2", ("done[%d] = %d" % (s, q)) if which == 1 else ("half 1 of %d done" % q)), fin))
            return
        if pr[F_SHUT] and which == 1:
            cands.append(("rank %d proxy: the peer's socket is closed (PeerLost)" % k, lambda R: self.fail(R, k, 1)))
        if r[F_ABORT]:
            cands.append(("rank %d proxy waiting for completions: abort word set" % k, lambda R: self.fail(R, k, 2)))
        if T1:
            cands.append(("rank %d proxy: no completion within the progress deadline (Timeout)" % k, lambda R: self.fail(R, k, 2)))

    # The NIC (RC QP, one per direction; QP 2 only in the hypothetical striped rdma)
    def nic_steps(self, script, ranks, k, cands):
        r = ranks[k]
        if r[F_ST] == "D" or r[F_QPERR]:
            return
        p = 1 - k
        pr = ranks[p]
        for (fo, fu, fsc, frc, tri) in ((F_OUT1, F_UNACK1, F_SCQ1, F_RCQ1, 2), (F_OUT2, F_UNACK2, F_SCQ2, F_RCQ2, 3)):
            if r[fo]:
                q, s, c, halves = r[fo][0]
                if pr[F_ST] == "D" or pr[F_QPERR]:
                    def err(R, fo=fo, fsc=fsc, s=s, q=q):
                        R[k][fo] = R[k][fo][1:]
                        R[k][F_TXRD] = tset(R[k][F_TXRD], s, R[k][F_TXRD][s] - 1)
                        R[k][fsc] = R[k][fsc] + ((q, 1),)  # RETRY_EXC_ERR: PeerLost (rdma_nic.statusCode)
                    cands.append(("rank %d NIC: write of gather %d unanswered, retries exhausted" % (k, q), err))
                else:
                    def deliver(R, fo=fo, fu=fu, frc=frc, tri=tri, q=q, s=s, c=c, halves=halves):
                        tx = R[k][F_TX][s]
                        R[k][fo] = R[k][fo][1:]
                        R[k][F_TXRD] = tset(R[k][F_TXRD], s, R[k][F_TXRD][s] - 1)
                        # the NIC read TX[s] some time before this placement: what it carries is TX[s] now
                        for h in halves:
                            self.rx_write(R, p, s, (h,), tx[h], c if tx[h] == q else R[k][F_SINFO][s], k)
                        slot = R[p][F_RX][s]
                        R[p][F_RX] = tset(R[p][F_RX], s, tset(slot, tri, (tx[tri], c)))
                        R[p][frc] = R[p][frc] + ((q, c, 0),)
                        R[k][fu] = R[k][fu] + (q,)
                    cands.append(("rank %d NIC places gather %d in rank %d's RX[%d]%s" % (k, q, p, s, "" if fo == F_OUT1 else " (QP 2)"), deliver,
                                  self.mode == "D1" and not any(pr[F_RX][s][h][3] for h in halves)))
            if r[fu]:
                def ack(R, fu=fu, fsc=fsc):
                    q = R[k][fu][0]
                    R[k][fu] = R[k][fu][1:]
                    R[k][fsc] = R[k][fsc] + ((q, 0),)
                cands.append(("rank %d NIC: ack, send completion%s" % (k, "" if fu == F_UNACK1 else " (QP 2)"), ack))

    # -- terminal properties ----------------------------------------------------------------------------------------
    def check_terminal(self, s):
        script, ranks, faults = s
        for k, r in enumerate(ranks):
            if r[F_ST] != "L":
                continue
            fin = r[F_PC] >= self.W
            code = r[F_FAILED] or (2 if r[F_ABORT] else 0)
            if self.mode == "D1":
                if not fin:
                    raise Violation("D1", "rank %d stopped at call %d of %d with nothing enabled (deadlock)" % (k, r[F_PC], self.W))
                if code or r[F_LOST]:
                    raise Violation("D1", "rank %d failed (%s) in a run with live ranks and no deadline" % (k, CODES[code]))
            else:
                if not fin:
                    raise Violation("T1", "rank %d is live and stuck at call %d with no deadline left to fire (silent hang)" % (k, r[F_PC]))
                if code == 0 and r[F_LOST]:
                    raise Violation("T1", "rank %d finished OK although a call's data never arrived" % k)
                if code not in (0, 1, 2):
                    raise Violation("T1", "rank %d reports %s for a dead/wedged peer or a deadline" % (k, CODES[code]))


# ------------------------------------------------------------------------------------------------ world 4 (RdmaProxyN)
W4F = ["pc", "opi", "cq", "seq", "abort", "failed", "ready", "done", "cons", "sinfo", "mem", "turns", "out", "unack",
       "scq", "rcq", "st", "lost", "qperr", "poisoned", "pby"]
for _i, _n in enumerate(W4F):
    globals()["G_" + _n.upper()] = _i
QEMPTY = (0, -1, -1, 0, 0)  # quarter of a slot's data area: generation, writer rank, call, pending (a local collect
#                              has not read it), NIC reads in flight (this rank's block as a write's source)
TEMPTY = (0, -1, 0, 0)  # trailer place of one rank: generation, call, pending (the proxy has not checked it), NIC reads
PAIRS = {"x": (1, 0, 3, 2), "z": (3, 2, 1, 0), "y": (2, 3, 0, 1)}


class World4:
    """Four (or three) ranks of rdma_proxy.zig RdmaProxyN over rdma_core.zig CoreN: one QP per peer, lanes of slots.

    Script kinds: a gather k=1; s gather k=1 swap12; c gather of L+1 chunks (k>1: one collect a rank, chunks round
    robin over the lanes); x exchange k=1 pairs (0,1)(2,3); z exchange k=1 pairs (0,3)(1,2), the lower rank sends 0
    bytes; y exchange of 2 chunks pairs (0,2)(1,3)."""

    def __init__(self, world, nslot, lanes, calls, kinds, mode, mut=()):
        assert nslot % lanes == 0 and (nslot // lanes >= 2 or "lanes" in mut)
        self.n = world
        self.N = nslot
        self.L = lanes
        self.m = nslot // lanes
        self.W = calls
        self.kinds = kinds
        self.mode = mode
        self.mut = set(mut)
        self.por = True
        self.level = "full"  # reduction level: full, basic (no structural collect/stage/start rules), none
        self.all = (1 << world) - 1
        self._ops = {}

    # -- the call shapes (Layout.gatherStep / exchangeStep / depth) -------------------------------------------------
    def chunks(self, kind):
        return {"a": 1, "s": 1, "c": self.L + 1, "x": 1, "z": 1, "y": 2}[kind]

    def place(self, kind, g):
        return 2 * (g % 2) + g // 2 if kind == "s" else g  # Word.place (swap12)

    def partner(self, kind, g):
        return PAIRS[kind][g] if kind in PAIRS else None

    def sizes(self, kind, g):
        """(send, recv) of a step for rank g: 0 or 1 (= some bytes)."""
        if kind == "z":
            p = PAIRS["z"][g]
            return (0, 1) if g < p else (1, 0)
        return (1, 1)

    def own_quarters(self, kind, g):
        p = self.partner(kind, g)
        if p is None:
            return (self.place(kind, g),)
        return (0, 1) if g < p else (2, 3)  # Layout.exchangePos: the lower rank at 0, the higher at exchangeHalf

    def ops(self, kind, g):
        key = (kind, g)
        if key in self._ops:
            return self._ops[key]
        k = self.chunks(kind)
        L = self.L
        d = min(k, L) if "depth" not in self.mut else min(k, L + 1)
        out = [("S", ci) for ci in range(d)]
        for ci in range(k):
            p = self.partner(kind, g)
            if p is not None:
                out.append(("C", ci, "x"))
            elif k == 1:
                out.append(("C", ci, "all"))
            else:
                for pl in range(self.n):
                    out.append(("C", ci, pl))
            if ci + d < k:
                out.append(("S", ci + d))
        self._ops[key] = tuple(out)
        return self._ops[key]

    # -- state ------------------------------------------------------------------------------------------------------
    def initial(self):
        L, m, n = self.L, self.m, self.n
        r = [None] * len(W4F)
        r[G_PC] = 0
        r[G_OPI] = 0
        r[G_CQ] = 0
        r[G_SEQ] = (0,) * L
        r[G_ABORT] = (0,) * L
        r[G_FAILED] = 0
        r[G_READY] = ((0,) * m,) * L
        r[G_DONE] = ((0,) * m,) * L
        r[G_CONS] = ((0,) * m,) * L
        r[G_SINFO] = ((-1,) * m,) * L
        r[G_MEM] = (((QEMPTY,) * 4, (TEMPTY,) * n),) * self.N
        r[G_TURNS] = ((0, 1, 0, 0, 0),) * L  # active, q, have_send mask, have_recv mask, early mask
        r[G_OUT] = ((),) * n
        r[G_UNACK] = ((),) * n
        r[G_SCQ] = ()
        r[G_RCQ] = ()
        r[G_ST] = "L"
        r[G_LOST] = 0
        r[G_QPERR] = 0
        r[G_POISONED] = 0
        r[G_PBY] = -1
        r = tuple(r)
        return ((), tuple(r for _ in range(n)), 0)

    def describe(self, s):
        script, ranks, faults = s
        out = {"script": "".join(script)}
        for k, r in enumerate(ranks):
            out["rank%d" % k] = {"st": r[G_ST], "call": r[G_PC], "op": r[G_OPI], "seq": r[G_SEQ], "ready": r[G_READY],
                                 "done": r[G_DONE], "consumed": r[G_CONS], "abort": r[G_ABORT],
                                 "failed": CODES[r[G_FAILED]], "turns": r[G_TURNS],
                                 "posted": tuple(len(x) for x in r[G_OUT]), "rcq": r[G_RCQ]}
        return out

    def slot_of(self, li, q):
        return li * self.m + q % self.m  # Layout.slotIndex

    def call_id(self, c, ci):
        return c * 16 + ci

    # -- transitions ------------------------------------------------------------------------------------------------
    def successors(self, s):
        script, ranks, faults = s
        self._faults = faults
        self._script = script
        need = min(self.W, max(r[G_PC] + 1 for r in ranks))
        if len(script) < need:
            return [("script call %d = %s" % (len(script), kd), (script + (kd,), ranks, faults)) for kd in self.kinds]
        cands = []
        for k in range(self.n):
            r = ranks[k]
            if r[G_ST] == "L":
                self.gpu_steps(script, ranks, k, cands)
                if not r[G_FAILED]:
                    self.proxy_steps(script, ranks, k, cands)
                elif r[G_QPERR] == 0:
                    cands.append(("rank %d: QPs to error (proxyLoop: nics.toErr)" % k, lambda R, k=k: self.to_err(R, k)))
            if r[G_ST] != "D" and not r[G_QPERR]:
                self.nic_steps(ranks, k, cands)
            if self.mode == "T1" and faults == 0 and r[G_ST] == "L":
                cands.append(("rank %d dies" % k, lambda R, k=k: self.die(R, k)))
                cands.append(("rank %d wedges" % k, lambda R, k=k: self.wedge(R, k)))
                cands.append(("rank %d: a deadline fires (Timeout)" % k, lambda R, k=k: self.spont(R, k)))
        if self.por:
            for c in cands:
                if len(c) == 3 and c[2]:
                    if "tpor" in self.mut and faults == 0:
                        # T1 with reduced fault timing: the eager step plus every fault, in every visited state
                        cands = [c] + [x for x in cands if x[0].endswith(("dies", "wedges", "(Timeout)")) and "deadline fires" in x[0] or x[0].endswith(("dies", "wedges"))]
                    else:
                        cands = [c]
                    break
        out = []
        for c in cands:
            lab, fn = c[0], c[1]
            R = [list(r) for r in ranks]
            try:
                f2 = fn(R)
            except Violation as v:
                out.append((lab, v))
                continue
            nf = faults + (1 if f2 == "fault" else 0)
            self.canon(R)
            R = tuple(tuple(r) for r in R)
            lo = min(r[G_PC] for r in R)
            if any(r[G_FAILED] or any(r[G_ABORT]) or r[G_ST] != "L" for r in R):
                lo = 0  # a rank's GPU may have skipped ahead of its proxy: keep every call's kind
            sc = script
            if any(x != "*" for x in sc[:lo]):
                sc = tuple("*" if i < lo else x for i, x in enumerate(sc))
            out.append((lab, (sc, R, nf)))
        return out

    def canon(self, R):
        for r in R:
            mem = r[G_MEM]
            # a generation the lane's proxy has finished, nobody reads any more and no NIC reads: its call is history
            tq = [r[G_TURNS][si // self.m][1] for si in range(self.N)]
            # (its generation too: every later reader compares against the generation it expects, which 0 never is)
            nm = tuple((tuple(qq if (qq[3] or qq[4] or qq[0] >= tq[si]) else QEMPTY for qq in qs),
                        tuple(tt if (tt[2] or tt[3] or tt[0] >= tq[si]) else TEMPTY for tt in ts)) for si, (qs, ts) in enumerate(mem))
            if nm != mem:
                r[G_MEM] = nm
            si = r[G_SINFO]
            nsi = tuple(tuple(-1 if r[G_READY][li][x] < r[G_TURNS][li][1] else si[li][x] for x in range(self.m)) for li in range(self.L))
            if nsi != si:
                r[G_SINFO] = nsi
            if r[G_FAILED] or r[G_ST] != "L":
                r[G_SCQ] = ()
                r[G_RCQ] = ()
                r[G_TURNS] = ((0, 0, 0, 0, 0),) * self.L

    def die(self, R, k):
        r = R[k]
        r[G_ST] = "D"
        r[G_QPERR] = 1
        r[G_OUT] = ((),) * self.n
        r[G_UNACK] = ((),) * self.n
        return "fault"

    def wedge(self, R, k):
        R[k][G_ST] = "W"
        return "fault"

    def spont(self, R, k):
        self.fail(R, k, 2)
        return "fault"

    def fail(self, R, k, code):
        """CoreN.fail: abort word up on every lane, then poison: a zero-byte write with immediate to every peer once
        (not back to the one that poisoned this rank); RdmaProxyN.poison keeps the first code."""
        r = R[k]
        if r[G_FAILED] == 0:
            r[G_FAILED] = code
        r[G_ABORT] = tuple(a or 1 for a in r[G_ABORT])
        if not r[G_POISONED]:
            r[G_POISONED] = 1
            out = list(r[G_OUT])
            for j in range(self.n):
                if j != k and j != r[G_PBY]:
                    out[j] = out[j] + (("X", code),)
            r[G_OUT] = tuple(out)

    def to_err(self, R, k):
        r = R[k]
        r[G_QPERR] = 1
        r[G_OUT] = ((),) * self.n
        r[G_UNACK] = ((),) * self.n

    def skip(self, R, k):
        R[k][G_LOST] = 1
        self.next_op(R, k)

    def next_op(self, R, k):
        r = R[k]
        r[G_CQ] = 0
        r[G_OPI] += 1
        if r[G_OPI] >= len(self.ops(self._script[r[G_PC]], k)):
            r[G_OPI] = 0
            r[G_PC] += 1

    # GPU: coll.stage / coll.collect on each lane's header (RdmaProxyN.launchStage / launchCollects)
    def gpu_steps(self, script, ranks, k, cands):
        r = ranks[k]
        if r[G_PC] >= self.W:
            return
        if r[G_FAILED]:
            def rest(R):
                R[k][G_PC] = self.W
                R[k][G_OPI] = 0
                R[k][G_CQ] = 0
                R[k][G_LOST] = 1
            cands.append(("rank %d: abort words up, the remaining kernels return untouched, the host gets the error" % k, rest))
            return
        c = r[G_PC]
        kind = script[c]
        op = self.ops(kind, k)[r[G_OPI]]
        ci = op[1]
        li = ci % self.L  # Layout.laneOf
        m = self.m
        T1 = self.mode == "T1" and self._faults > 0
        if r[G_ABORT][li]:
            cands.append(("rank %d %s lane %d: abort word set, returns" % (k, "stage" if op[0] == "S" else "collect", li), lambda R: self.skip(R, k)))
            return
        if op[0] == "S":
            q = r[G_SEQ][li] + 1
            ls = q % m
            gate = q <= m or r[G_CONS][li][ls] + m >= q
            if gate:
                slot = self.slot_of(li, q)
                eager = (self.mode == "D1" or (self._faults == 0 and "T1nopor" not in self.mut)) and self.stage_eager(ranks, k, slot, kind)
                cands.append(("rank %d stage call %d chunk %d: gather %d of lane %d into slot %d, ready = %d" % (k, c, ci, q, li, slot, q),
                              lambda R: self.stage(R, k, c, ci, kind, li, q, ls, slot), eager))
            elif T1:
                cands.append(("rank %d stage lane %d: consumed wait past its deadline, abort = 2" % (k, li), lambda R: self.dev_timeout(R, k, li)))
            return
        if r[G_CQ] == 0:
            def rd(R):
                R[k][G_CQ] = R[k][G_SEQ][li]
            cands.append(("rank %d collect call %d chunk %d reads lane %d seq = %d" % (k, c, ci, li, r[G_SEQ][li]), rd, True))
            return
        q = r[G_CQ]
        ls = q % m
        d = r[G_DONE][li][ls]
        if d == q:
            # a collect at the head of the stream commutes with every step that can precede it: all of gather q's
            # writes are placed (done needs every trailer, each after its data on the same QP), and a peer can write
            # gather q + m into this slot only after this rank's trailer of q + m - 1, which needs this rank's stage of
            # q + m - 1, later in this stream (seq < q + m - 1 now); with one slot a lane this fails and it is expanded
            eager = self.mode == "D1" and self.level == "full" and r[G_SEQ][li] < q + m - 1
            cands.append(("rank %d collect call %d chunk %d %s: done = %d, copies slot %d, consumed = %d" % (k, c, ci, op[2], q, self.slot_of(li, q), q),
                          lambda R: self.collect(R, k, c, ci, kind, li, q, ls, op[2]), eager))
        elif d > q:
            cands.append(("rank %d collect lane %d: done %d past %d, abort = 2" % (k, li, d, q), lambda R: self.dev_timeout(R, k, li)))
        elif T1:
            cands.append(("rank %d collect lane %d: done wait past its deadline, abort = 2" % (k, li), lambda R: self.dev_timeout(R, k, li)))

    def stage_eager(self, ranks, k, slot, kind):
        """coll.stage writes only this rank's own block of the slot. A peer's write of the same gather lands in the
        peer's own block (disjoint); one of an earlier gather in this slot was placed before this rank's done of it,
        hence before its collect and this stage; one of a later gather needs this stage first (its trailer). So only
        this rank's own NIC reads of the slot (an earlier gather's source, still queued) can conflict."""
        r = ranks[k]
        for j in range(self.n):
            if any(w[0] in "DT" and w[1] == slot for w in r[G_OUT][j]):
                return False
            if self.level != "full" and j != k and any(w[0] in "DT" and w[1] == slot for w in ranks[j][G_OUT][k]):
                return False  # the basic level: also no peer write toward this slot in flight
        return True

    def dev_timeout(self, R, k, li):
        r = R[k]
        if r[G_ABORT][li] == 0:
            r[G_ABORT] = tset(r[G_ABORT], li, 2)
        self.skip(R, k)

    def stage(self, R, k, c, ci, kind, li, q, ls, slot):
        r = R[k]
        call = self.call_id(c, ci)
        send, _ = self.sizes(kind, k) if self.partner(kind, k) is not None else (1, 1)
        qs, ts = r[G_MEM][slot]
        qs = list(qs)
        if send:
            for x in self.own_quarters(kind, k):
                old = qs[x]
                if (old[3] and old[0] != q) or old[4]:
                    raise Violation("S1", "rank %d stage of lane %d gather %d overwrites slot %d quarter %d holding gather %d (pending read %d, NIC reads %d)" % (k, li, q, slot, x, old[0], old[3], old[4]))
                qs[x] = (q, k, call, 1 if self.partner(kind, k) is None else 0, 0)
        r[G_MEM] = tset(r[G_MEM], slot, (tuple(qs), ts))
        r[G_SINFO] = tset(r[G_SINFO], li, tset(r[G_SINFO][li], ls, call))
        r[G_SEQ] = tset(r[G_SEQ], li, q)
        r[G_READY] = tset(r[G_READY], li, tset(r[G_READY][li], ls, q))
        self.next_op(R, k)

    def collect(self, R, k, c, ci, kind, li, q, ls, blk):
        r = R[k]
        slot = self.slot_of(li, q)
        call = self.call_id(c, ci)
        qs, ts = r[G_MEM][slot]
        qs = list(qs)
        if blk == "x":
            p = self.partner(kind, k)
            reads = [(x, p) for x in self.own_quarters(kind, p)] if self.sizes(kind, k)[1] else []
        elif blk == "all":
            reads = [(self.place(kind, g), g) for g in range(self.n)]
        else:
            g = [g for g in range(self.n) if self.place(kind, g) == blk][0]
            reads = [(blk, g)]
        for x, g in reads:
            t = qs[x]
            if t[0] != q or t[1] != g or t[2] != call:
                raise Violation("S2", "rank %d collect of call %d chunk %d (lane %d gather %d) reads slot %d quarter %d holding gather %d of call %s from rank %d (expected rank %d)" % (k, c, ci, li, q, slot, x, t[0], divmod(t[2], 16) if t[2] >= 0 else "?", t[1], g))
            qs[x] = (t[0], t[1], t[2], 0, t[4])
        r[G_MEM] = tset(r[G_MEM], slot, (tuple(qs), ts))
        r[G_CONS] = tset(r[G_CONS], li, tset(r[G_CONS][li], ls, q))
        self.next_op(R, k)

    # proxy: rdma_core.zig CoreN.run / start / check
    def proxy_steps(self, script, ranks, k, cands):
        r = ranks[k]
        T1 = self.mode == "T1" and self._faults > 0
        D1 = self.mode == "D1" or (self._faults == 0 and "T1nopor" not in self.mut)
        inflight = 0
        for li in range(self.L):
            act, q, hs, hr, early = r[G_TURNS][li]
            if act:
                inflight += 1
                if hs == self.all and hr == self.all:
                    cands.append(("rank %d proxy: lane %d gather %d complete, done = %d" % (k, li, q, q), lambda R, li=li: self.finish(R, k, li), D1))
                continue
            ls = q % self.m
            rd = r[G_READY][li][ls]
            if rd == q:
                slot = self.slot_of(li, q)
                # (early trailers it checks were placed already, and their places are not rewritten before this
                # rank's own trailer of this gather; a recv poll and this start commute, see poll_recv's branches)
                eager = D1 and (early == 0 or self.level == "full") and not any(w[0] in "DT" and w[1] == slot for j in range(self.n) for w in r[G_OUT][j])
                cands.append(("rank %d proxy starts lane %d gather %d (slot %d)" % (k, li, q, slot), lambda R, li=li: self.start(R, k, li), eager))
            elif rd > q:
                cands.append(("rank %d proxy: lane %d ready %d ahead of %d (Mismatch)" % (k, li, rd, q), lambda R: self.fail(R, k, 3)))
        if any(r[G_ABORT]):
            cands.append(("rank %d proxy: an abort word is up (deviceGaveUp)" % k, lambda R: self.fail(R, k, 2)))
        if r[G_SCQ]:
            cands.append(("rank %d proxy polls its send CQ" % k, lambda R: self.poll_send(R, k), D1))
        if r[G_RCQ]:
            cands.append(("rank %d proxy polls its receive CQ" % k, lambda R: self.poll_recv(R, k), D1))
        if inflight:
            if T1:
                cands.append(("rank %d proxy: no progress within the deadline (Timeout)" % k, lambda R: self.fail(R, k, 2)))
            watch = range(1, self.n) if k == 0 else (0,)
            if any(ranks[j][G_ST] == "D" for j in watch):
                cands.append(("rank %d proxy: a peer closed its control socket (PeerLost)" % k, lambda R: self.fail(R, k, 1)))

    def start(self, R, k, li):
        r = R[k]
        act, q, hs, hr, early = r[G_TURNS][li]
        ls = q % self.m
        slot = self.slot_of(li, q)
        call = r[G_SINFO][li][ls]
        c, ci = divmod(call, 16)
        kind = self._script[c]
        p = self.partner(kind, k)
        send = self.sizes(kind, k)[0] if p is not None else 1
        qs, ts = r[G_MEM][slot]
        qs, ts = list(qs), list(ts)
        if ts[k][3]:
            raise Violation("S1-TX", "rank %d proxy rewrites its trailer of slot %d (gather %d) while the NIC still reads gather %d's" % (k, slot, q, ts[k][0]))
        npeer = self.n - 1
        ts[k] = (q, call, 0, npeer)
        out = list(r[G_OUT])
        mine = self.own_quarters(kind, k)
        for j in range(self.n):
            if j == k:
                continue
            if send and (p is None or p == j):
                out[j] = out[j] + (("D", slot, mine, q, call),)
                for x in mine:
                    qq = qs[x]
                    qs[x] = qq[:4] + (qq[4] + 1,)
            out[j] = out[j] + (("T", slot, li, q, call),)
        r[G_OUT] = tuple(out)
        r[G_MEM] = tset(r[G_MEM], slot, (tuple(qs), tuple(ts)))
        hs = hr = 1 << k
        for j in range(self.n):
            if early >> j & 1:
                hr |= 1 << j
                if not self.check(R, k, slot, q, call, j):
                    r[G_TURNS] = tset(r[G_TURNS], li, (1, q, hs, hr, 0))
                    self.fail(R, k, 3)
                    return
        r[G_TURNS] = tset(r[G_TURNS], li, (1, q, hs, hr, 0))

    def check(self, R, k, slot, q, call, j):
        """CoreN.check: peer j's trailer of gather q against this rank's call; marks it read."""
        r = R[k]
        qs, ts = r[G_MEM][slot]
        t = ts[j]
        ok = t[0] == q and t[1] == call
        r[G_MEM] = tset(r[G_MEM], slot, (qs, tset(ts, j, (t[0], t[1], 0, t[3]))))
        return ok

    def finish(self, R, k, li):
        r = R[k]
        act, q, hs, hr, early = r[G_TURNS][li]
        ls = q % self.m
        r[G_DONE] = tset(r[G_DONE], li, tset(r[G_DONE][li], ls, q))
        r[G_TURNS] = tset(r[G_TURNS], li, (0, q + 1, 0, 0, early))

    def poll_send(self, R, k):
        r = R[k]
        wr, st = r[G_SCQ][0]
        r[G_SCQ] = r[G_SCQ][1:]
        if st:
            self.fail(R, k, st)
            return
        if wr == "X":
            return
        li, q, j = wr
        act, tq, hs, hr, early = r[G_TURNS][li]
        if act and tq == q:
            r[G_TURNS] = tset(r[G_TURNS], li, (act, tq, hs | (1 << j), hr, early))

    def poll_recv(self, R, k):
        r = R[k]
        j, imm, ln, st = r[G_RCQ][0]
        r[G_RCQ] = r[G_RCQ][1:]
        if st:
            self.fail(R, k, st)
            return
        if ln == 0:  # a peer failed and says so
            r[G_PBY] = j
            self.fail(R, k, 1)
            return
        li, q = imm
        act, tq, hs, hr, early = r[G_TURNS][li]
        b = 1 << j
        if act and q == tq and not hr & b:
            r[G_TURNS] = tset(r[G_TURNS], li, (act, tq, hs, hr | b, early))
            slot = self.slot_of(li, tq)
            call = r[G_SINFO][li][tq % self.m]
            if not self.check(R, k, slot, tq, call, j):
                self.fail(R, k, 3)
        elif act and q == tq + 1 and hr & b and not early & b:
            r[G_TURNS] = tset(r[G_TURNS], li, (act, tq, hs, hr, early | b))
        elif not act and q == tq and not early & b:
            r[G_TURNS] = tset(r[G_TURNS], li, (act, tq, hs, hr, early | b))
        else:
            self.fail(R, k, 3)

    # NIC: one RC QP per ordered pair; data (unsignaled) then trailer (signaled, with immediate) in order
    def nic_steps(self, ranks, k, cands):
        r = ranks[k]
        for j in range(self.n):
            if r[G_OUT][j]:
                w = r[G_OUT][j][0]
                if ranks[j][G_ST] == "D" or ranks[j][G_QPERR]:
                    cands.append(("rank %d NIC -> rank %d: retries exhausted (RETRY_EXC_ERR)" % (k, j), lambda R, j=j: self.nic_err(R, k, j)))
                else:
                    what = "data of gather %d" % w[3] if w[0] == "D" else "trailer of lane %d gather %d" % (w[2], w[3]) if w[0] == "T" else "poison"
                    cands.append(("rank %d NIC places %s in rank %d" % (k, what, j), lambda R, j=j: self.deliver(R, k, j), self.deliver_eager(ranks, k, j, w)))
            if r[G_UNACK][j]:
                cands.append(("rank %d NIC: ack from rank %d, send completion" % (k, j), lambda R, j=j: self.ack(R, k, j), True))

    def deliver_eager(self, ranks, k, j, w):
        """A placement whose target holds nothing unread commutes with every step that can precede it (the receiver
        reads that generation only after this QP's trailer; the sender restages its block only after the ack); a
        placement onto unread data is the S1 question itself and is never reduced."""
        if w[0] == "X" or not (self.mode == "D1" or ("tpor" in self.mut and self._faults == 0)):
            return False
        dq, dt = ranks[j][G_MEM][w[1]]
        if w[0] == "D":
            return all(not dq[x][3] and not dq[x][4] for x in w[2])
        return not dt[k][2]

    def nic_err(self, R, k, j):
        r = R[k]
        w = r[G_OUT][j][0]
        wr = "X" if w[0] == "X" else (w[2], w[3], j) if w[0] == "T" else ("D", w[3], j)
        # the failed write's error completion, then the QP toward j is in error: the rest flush (no further reads)
        for x in r[G_OUT][j]:
            if x[0] in "DT":
                self.release_reads(R, k, x)
        r[G_OUT] = tset(r[G_OUT], j, ())
        r[G_SCQ] = r[G_SCQ] + ((wr, 1),)

    def release_reads(self, R, k, w):
        r = R[k]
        slot = w[1]
        qs, ts = r[G_MEM][slot]
        if w[0] == "D":
            qs = list(qs)
            for x in w[2]:
                qs[x] = qs[x][:4] + (qs[x][4] - 1,)
            qs = tuple(qs)
        else:
            ts = tset(ts, k, ts[k][:3] + (ts[k][3] - 1,))
        r[G_MEM] = tset(r[G_MEM], slot, (qs, ts))

    def deliver(self, R, k, j):
        r = R[k]
        w = r[G_OUT][j][0]
        r[G_OUT] = tset(r[G_OUT], j, r[G_OUT][j][1:])
        pj = R[j]
        pfail = pj[G_FAILED] or any(pj[G_ABORT])
        if w[0] == "X":
            pj[G_RCQ] = pj[G_RCQ] + ((k, w[1], 0, 0),)
            r[G_UNACK] = tset(r[G_UNACK], j, r[G_UNACK][j] + ("X",))
            return
        slot = w[1]
        q = w[3]
        call = w[4]
        sq, st = r[G_MEM][slot]
        dq, dt = pj[G_MEM][slot]
        if w[0] == "D":
            dq = list(dq)
            for x in w[2]:
                src = sq[x]
                if src[0] != q and not r[G_FAILED]:
                    raise Violation("S1-TX", "rank %d's block in slot %d quarter %d was restaged (gather %d) before the NIC read gather %d" % (k, slot, x, src[0], q))
                old = dq[x]
                if ((old[3] and old[0] != q) or old[4]) and not pfail:
                    raise Violation("S1", "rank %d's write of gather %d lands in rank %d's slot %d quarter %d, which still holds gather %d for its collect" % (k, q, j, slot, x, old[0]))
                dq[x] = (src[0], k, src[2] if src[0] == q else -1, 1, 0)
            pj[G_MEM] = tset(pj[G_MEM], slot, (tuple(dq), dt))
            self.release_reads(R, k, w)
            return
        src = st[k]
        if src[0] != q and not r[G_FAILED]:
            raise Violation("S1-TX", "rank %d's trailer in slot %d was rewritten (gather %d) before the NIC read gather %d's" % (k, slot, src[0], q))
        old = dt[k]
        if old[2] and old[0] != q and not pfail:
            raise Violation("S1", "rank %d's trailer of gather %d lands in rank %d's slot %d before rank %d checked gather %d's" % (k, q, j, slot, j, old[0]))
        pj[G_MEM] = tset(pj[G_MEM], slot, (dq, tset(dt, k, (src[0], src[1], 1, 0))))
        self.release_reads(R, k, w)
        pj[G_RCQ] = pj[G_RCQ] + ((k, (w[2], q), 64, 0),)
        r[G_UNACK] = tset(r[G_UNACK], j, r[G_UNACK][j] + ((w[2], q, j),))

    def ack(self, R, k, j):
        r = R[k]
        wr = r[G_UNACK][j][0]
        r[G_UNACK] = tset(r[G_UNACK], j, r[G_UNACK][j][1:])
        if not r[G_FAILED]:
            r[G_SCQ] = r[G_SCQ] + ((wr, 0),)

    def check_terminal(self, s):
        script, ranks, faults = s
        for k, r in enumerate(ranks):
            if r[G_ST] != "L":
                continue
            fin = r[G_PC] >= self.W
            code = r[G_FAILED] or (2 if any(r[G_ABORT]) else 0)
            if self.mode == "D1":
                if not fin:
                    raise Violation("D1", "rank %d stopped at call %d op %d of %d calls with nothing enabled (deadlock)" % (k, r[G_PC], r[G_OPI], self.W))
                if code or r[G_LOST]:
                    raise Violation("D1", "rank %d failed (%s) in a run with live ranks and no deadline" % (k, CODES[code]))
            else:
                if not fin:
                    raise Violation("T1", "rank %d is live and stuck at call %d with no deadline left to fire (silent hang)" % (k, r[G_PC]))
                if code == 0 and r[G_LOST]:
                    raise Violation("T1", "rank %d finished OK although a call's data never arrived" % k)
                if code not in (0, 1, 2):
                    raise Violation("T1", "rank %d reports %s for a dead/wedged peer or a deadline" % (k, CODES[code]))


# ------------------------------------------------------------------------------------------------ S3 (static)
SHARED_BYTES = 64 + 16 * 64  # abi.h tf_ds_coll_shared (1088), tcp_proxy.Shared
TRAILER = 64  # rdma_nic.trailer_bytes


def align_fwd(x, a):
    return (x + a - 1) // a * a


def align_back(x, a):
    return x // a * a


def s3_check():
    """Every remote write's byte range against the receiver's registered RX region, which must exclude the slab header,
    over a grid of settings and sizes; the arithmetic is rdma_proxy.zig open/openN, rdma_nic.zig trailerAt/wireLen/
    capacity and rdma_core.zig Layout (gatherChunk, exchangeHalf, exchangePos, trailerOff, the start() bound)."""
    checked = 0
    problems = []
    slot_kibs = (1, 4, 5, 64, 4352, 8704)
    for nslot in range(2, 17):
        for kib in slot_kibs:
            sb = align_fwd(kib << 10, 4096)
            # world 2: header [0, 1088), TX MR [4096, 4096 + n sb) local only, RX MR [rx_off, rx_off + n sb) remote write
            tx_off = 4096
            rx_off = tx_off + nslot * sb
            if rx_off < SHARED_BYTES or tx_off < SHARED_BYTES:
                problems.append(("w2", nslot, kib, "an MR overlaps the header"))
            cap = sb - TRAILER
            for b in sorted({0, 1, 63, 64, 65, cap - 65, cap - 64, cap - 1, cap, cap + 1, sb}):
                if b < 0 or b > cap:
                    continue  # gatherBegin: larger than capacity goes to tcp_host (rdma_proxy.zig:129)
                if b == 0:
                    continue  # gatherBegin returns before the stage (rdma_proxy.zig:141): nothing is posted
                ln = align_fwd(b, 64) + TRAILER  # wireLen
                if ln > sb:
                    continue  # the proxy refuses it (rdma_core.zig:58): nothing is posted
                for s in range(nslot):
                    lo, hi = s * sb, s * sb + ln
                    checked += 1
                    if not (0 <= lo and hi <= nslot * sb):
                        problems.append(("w2", nslot, kib, b, s, "write outside the RX MR"))
            # world 3/4
            for world in (3, 4):
                for lanes in (1, 2, 3, 4):
                    if not (nslot % lanes == 0 and nslot // lanes >= 2):
                        continue
                    if sb <= world * TRAILER:
                        continue  # openN refuses (rdma_proxy.zig:507)
                    rx4 = align_fwd(lanes * SHARED_BYTES, 4096)
                    if rx4 < lanes * SHARED_BYTES:
                        problems.append(("w4", nslot, kib, "MR overlaps a lane header"))
                    data = sb - world * TRAILER
                    chunk = align_back(data // world, 64)
                    half = align_back(data // 2, 64)
                    if chunk == 0:
                        continue  # openN refuses
                    tr = [sb - (world - g) * TRAILER for g in range(world)]
                    for g in range(world):
                        if tr[g] < data or tr[g] + TRAILER > sb:
                            problems.append(("w4", nslot, kib, world, "trailer place outside the trailer area"))
                    for n in sorted({1, 63, 64, chunk - 1, chunk}):
                        if n <= 0 or n > chunk:
                            continue
                        for swap in (False, True):
                            if swap and world != 4:
                                continue
                            places = [2 * (g % 2) + g // 2 if swap else g for g in range(world)]
                            if sorted(places) != list(range(world)):
                                problems.append(("w4", "places not a permutation"))
                            ranges = [(p * n, p * n + n) for p in places]
                            for (a, b_) in ranges:
                                checked += 1
                                if b_ > data:
                                    problems.append(("w4", nslot, kib, world, lanes, n, "gather block past the data area"))
                            rs = sorted(ranges)
                            for x in range(len(rs) - 1):
                                if rs[x][1] > rs[x + 1][0]:
                                    problems.append(("w4", nslot, kib, world, n, "gather blocks overlap"))
                    for sn in sorted({0, 1, 64, half - 1, half}):
                        if sn < 0 or sn > half:
                            continue
                        for g, p in ((0, 1), (1, 0)):
                            pos = 0 if g < p else half
                            checked += 1
                            if pos + sn > data:
                                problems.append(("w4", nslot, kib, world, sn, "exchange half past the data area"))
                        if half + 0 < sn:
                            problems.append(("w4", "exchange halves overlap"))
                    if 2 * half > data:
                        problems.append(("w4", nslot, kib, world, "exchange halves overlap the trailers"))
                    for s in range(nslot):
                        base = s * sb
                        if base + sb > nslot * sb:
                            problems.append(("w4", "slot past the MR"))
    return {"checked": checked, "problems": problems}


class Checker:
    """Breadth-first exhaustive search with parent links for counterexample traces."""

    def __init__(self, model, max_states):
        self.m = model
        self.max_states = max_states

    def run(self):
        m = self.m
        t0 = time.time()
        init = m.initial()
        ids = {init: 0}
        parent = [-1]
        label = [-1]  # the index of the step among the parent's successors (the text is rebuilt for a trace)
        states = [init]
        q = deque([0])
        terminals = 0
        transitions = 0
        result = {"exhausted": False, "violation": None}
        while q:
            i = q.popleft()
            s = states[i]
            try:
                succ = m.successors(s)
            except Violation as v:
                result["violation"] = {"prop": v.prop, "msg": v.msg, "trace": self.trace(i, parent, label, states)}
                break
            if not succ:
                terminals += 1
                try:
                    m.check_terminal(s)
                except Violation as v:
                    result["violation"] = {"prop": v.prop, "msg": v.msg, "trace": self.trace(i, parent, label, states)}
                    break
                continue
            for si, (lab, ns) in enumerate(succ):
                transitions += 1
                if isinstance(ns, Violation):
                    tr = self.trace(i, parent, label, states)
                    tr.append({"step": lab, "state": "VIOLATION"})
                    result["violation"] = {"prop": ns.prop, "msg": ns.msg, "trace": tr}
                    q.clear()
                    break
                if ns not in ids:
                    ids[ns] = len(states)
                    states.append(ns)
                    parent.append(i)
                    label.append(si)
                    q.append(len(states) - 1)
            if result["violation"]:
                break
            if len(states) > self.max_states:
                result["stopped"] = "max-states"
                break
        else:
            result["exhausted"] = result["violation"] is None
        if result["violation"] is None and q:
            result["exhausted"] = False
        result.update(states=len(states), transitions=transitions, terminals=terminals, seconds=round(time.time() - t0, 1))
        return result

    def trace(self, i, parent, label, states):
        path = []
        while i > 0:
            path.append(i)
            i = parent[i]
        path.reverse()
        out = []
        for j in path:
            lab = self.m.successors(states[parent[j]])[label[j]][0]
            out.append({"step": lab, "state": self.m.describe(states[j])})
        return out


# ------------------------------------------------------------------------------------------------ trace annotation
# step text -> code (fork 003f679c28b1, zig/src/families/dsv41/ unless named): the CORRESPONDENCE table's rows, first match
STEP_CODE = [
    ("script call", "coll.zig contract: both ranks issue the same calls (COLLECTIVES-DESIGN 2.2)"),
    ("reads lane", "coll.hip:93 (collect reads its lane header's seq; rdma_proxy.zig:402-410 launchCollects)"),
    ("reads seq", "coll.hip:93 (tf_ds_coll_collect: q = seq)"),
    ("MUTATION", "not in the code (a control)"),
    ("abort words up, the remaining kernels", "coll.hip:20-22,59,97 (tf_ds_coll_aborted) + rdma_proxy.zig:337-347 status()"),
    ("abort word set, the remaining kernels", "coll.hip:20-22,59,97 (tf_ds_coll_aborted) + tcp_proxy.zig:333-344 / rdma_proxy.zig:68-79 status()"),
    ("consumed wait past its deadline", "coll.hip:62 (tf_ds_coll_fail, abort = 2)"),
    ("done wait past its deadline", "coll.hip:100 (tf_ds_coll_fail)"),
    ("past", "coll.hip:100 (d > q: tf_ds_coll_fail)"),
    ("stage call", "coll.hip:47-84 (q = seq + 1 :54, consumed wait :59-64, copy :70-72, bytes/site/seq/ready :78-81); launched by tcp_proxy.zig:426 / rdma_proxy.zig:144 / rdma_proxy.zig:393-400"),
    ("stage lane", "coll.hip:59-64"),
    ("collect", "coll.hip:86-117 (done wait :97-102, copy :108, consumed :114); launched by tcp_proxy.zig:442 / rdma_proxy.zig:162 / rdma_proxy.zig:402-410"),
    ("host gather", "tcp_coll.zig:282-330 (tcp_host gatherBegin/End), status() first"),
    ("host call", "tcp_wire.zig:203-224 exchange, :230-265 runBoth (helper), :266 run, :328 header check"),
    ("hands half 1", "tcp_proxy.zig:113, 134-136 (split), :237-242 Second.hand"),
    ("proxy sees ready", "tcp_proxy.zig:111-136 / rdma_core.zig:40-71 (ready :42, trailer :63-65, postWrite :68 -> rdma_nic.zig:206-211)"),
    ("proxy sends gather", "tcp_proxy.zig:138 -> tcp_wire.zig:203 (sendmsg half of the exchange)"),
    ("proxy receives", "tcp_proxy.zig:138 -> tcp_wire.zig:203 (recvmsg into RX[s]; header check :328)"),
    ("cable 1: end of stream", "tcp_wire.zig (EOF -> PeerLost) -> tcp_proxy.zig:139-140, :205-213"),
    ("cable 1: no progress", "tcp_wire.zig progress deadline (TF_COLL_PROGRESS_MS) -> tcp_proxy.zig:139-140"),
    ("waiting for the second cable", "tcp_proxy.zig:142-151"),
    ("proxy: done[", "tcp_proxy.zig:153"),
    ("second thread", "tcp_proxy.zig:244-268 (Second.run)"),
    ("proxy 2", "hypothetical: tcp_proxy.Second over a second QP (refused at rdma_proxy.zig:211-215)"),
    ("polls its send CQ", "rdma_core.zig:97-109 (world 2) / :490-501 (world 4, CoreN)"),
    ("polls its receive CQ", "rdma_core.zig:110-132 (world 2) / :503-531 (world 4: poison :511-515, turn branches :520-531, check :646-663)"),
    ("both completions, trailer checked", "rdma_core.zig:74-85 (trailer check, done[s] = q)"),
    ("waiting for QP 2", "hypothetical (mirrors tcp_proxy.zig:142-151)"),
    ("peer's socket is closed", "rdma_core.zig:146-149, peerGone :169-178"),
    ("no completion within the progress deadline", "rdma_core.zig:139-145"),
    ("waiting for completions: abort", "rdma_core.zig:135-138"),
    ("proxy idle: abort", "tcp_proxy.zig:119-122 / rdma_core.zig:48-51"),
    ("ahead of", "tcp_proxy.zig:115-117 / rdma_core.zig:44-47 / :473-476 (Mismatch)"),
    ("proxy starts lane", "rdma_core.zig:465-472 + start :592-643 (own trailer :613-614, posts :627-635 -> rdma_nic.zig:443-457, early trailers :636-640)"),
    ("complete, done", "rdma_core.zig:533-543"),
    ("an abort word is up", "rdma_core.zig:480-484, :546-549"),
    ("no progress within the deadline", "rdma_core.zig:551-558"),
    ("closed its control socket", "rdma_core.zig:559-564"),
    ("QPs to error", "rdma_proxy.zig:380-383 (proxyLoop: poison, nics.toErr)"),
    ("poison: QP to error", "tcp_proxy.zig:346-352 / rdma_proxy.zig:81-86 (poison)"),
    ("NIC places poison", "rdma_core.zig:673-694 poison -> rdma_nic.zig:460-466 postPoison (zero-byte write with immediate)"),
    ("NIC places", "RC write (with immediate) placed in order at the peer: rdma_nic.zig:206-211 (world 2), :443-457 (world 4: data unsignaled, then the signaled trailer)"),
    ("retries exhausted", "rdma_nic.zig:184 / :416 (retry_cnt 7) -> IBV_WC_RETRY_EXC_ERR -> statusCode :96-98 = PeerLost"),
    ("ack", "signaled send completion (rdma_nic.zig:208 / :446)"),
    ("dies", "fault injection: the process is gone (QPs destroyed, sockets closed)"),
    ("wedges", "fault injection: the process stops but its NIC and sockets stay up"),
    ("a deadline fires", "fault injection: any wait of this rank times out (Timeout)"),
]


def code_of(step):
    for pat, ref in STEP_CODE:
        if pat in step:
            return ref
    return "?"


def trace_text(res):
    v = res["violation"]
    lines = ["# %s counterexample: %s" % (v["prop"], v["msg"]),
             "# configuration: %s" % json.dumps({k: res[k] for k in res if k not in ("trace", "violation", "states", "transitions", "terminals", "seconds", "exhausted", "max_states")}),
             "# %d steps; each line: step | the code it models" % len(res["trace"]), ""]
    for i, st in enumerate(res["trace"], 1):
        lines.append("%3d. %s\n     | %s" % (i, st["step"], code_of(st["step"])))
    last = [x for x in res["trace"] if x["state"] != "VIOLATION"]
    if last:
        lines += ["", "# state before the violating step:", json.dumps(last[-1]["state"], indent=1)]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------ CLI
def build(cfg):
    if cfg["world"] == 2:
        m = World2(cfg["transport"], cfg["nslot"], cfg["calls"], cfg["kinds"], cfg["stripe"], cfg["wire"], cfg["mode"], cfg.get("mut", ()))
    else:
        m = World4(cfg["world"], cfg["nslot"], cfg["lanes"], cfg["calls"], cfg["kinds"], cfg["mode"], cfg.get("mut", ()))
    if cfg.get("nopor") or cfg.get("level") == "none":
        m.por = False
    if cfg.get("level"):
        m.level = cfg["level"]
    return m


def run_one(cfg):
    m = build(cfg)
    r = Checker(m, cfg.get("max_states", 4_000_000)).run()
    out = dict(cfg)
    out.update({k: v for k, v in r.items() if k != "violation"})
    if r["violation"]:
        v = r["violation"]
        out["violation"] = {"prop": v["prop"], "msg": v["msg"], "trace_len": len(v["trace"])}
        out["trace"] = v["trace"]
    return out


def sweep_configs(which):
    cfgs = []
    if which in ("w2", "all"):
        for tp in ("tcp", "rdma"):
            for stripe in (0, 1):
                for wire in (1, 2):
                    for n in range(2, 9):
                        for mode in ("D1", "T1"):
                            # D1 (and S1/S2 on every step): every slot reused at least twice; T1: three calls (see the
                            # report: the deadline paths do not depend on slot reuse, and the stage's consumed wait never
                            # blocks under stream order)
                            calls = 2 * n + 2 if mode == "D1" else 3
                            cfgs.append({"world": 2, "transport": tp, "stripe": stripe, "wire": wire, "nslot": n, "mode": mode,
                                         "calls": calls, "kinds": "SPH" if stripe else "SH",
                                         "note": "hypothetical: rdma_proxy.zig:211-215 refuses it" if (tp == "rdma" and stripe) else ""})
    if which in ("w2deep", "w2deep-rdma"):
        # T1 with every slot reused once more (calls = NSLOT + 2) where it fits the state budget
        for tp in (("tcp", "rdma") if which == "w2deep" else ("rdma",)):
            for stripe in (0, 1):
                for n in range(2, 9):
                    cfgs.append({"world": 2, "transport": tp, "stripe": stripe, "wire": 2, "nslot": n, "mode": "T1",
                                 "calls": n + 2, "kinds": "SPH" if stripe else "SH", "deep": 1})
    if which == "w4b":
        # the world-4 D1 rows the first sweep had not reached, and T1 at the deployed size (bounded: see the report)
        for n in range(2, 9):
            for lanes in (1, 2, 3, 4):
                if n % lanes or n // lanes < 2:
                    continue
                if (n, lanes) not in ((2, 1), (3, 1), (4, 1)):
                    m = n // lanes
                    cfgs.append({"world": 4, "nslot": n, "lanes": lanes, "mode": "D1", "calls": 2 * m + 1, "kinds": "axzs",
                                 "shape": "decode", "stripe": 0, "wire": 1})
                    cfgs.append({"world": 4, "nslot": n, "lanes": lanes, "mode": "D1", "calls": 2, "kinds": "cy",
                                 "shape": "prompt", "stripe": 0, "wire": 1})
        cfgs.append({"world": 4, "nslot": 8, "lanes": 4, "mode": "T1", "calls": 1, "kinds": "a", "shape": "decode",
                     "stripe": 0, "wire": 1, "unreduced_faults": 1})
    if which == "w4t1":
        # T1 at world 4: one call (the model is then the same for every NSLOT/lanes with 2+ slots a lane: one slot
        # gather on lane 0), faults in every visited state, pre-fault steps reduced (tpor; see the report)
        for kinds in ("a", "x"):
            cfgs.append({"world": 4, "nslot": 8, "lanes": 4, "mode": "T1", "calls": 1, "kinds": kinds, "mut": ["tpor"],
                         "shape": "decode", "stripe": 0, "wire": 1})
    if which == "porval":
        # the reduction checked against weaker ones: the same verdicts must come out
        for lev in ("none", "basic", "full"):
            for tp, st, n, mode, calls in (("tcp", 1, 2, "D1", 6), ("rdma", 0, 3, "D1", 8), ("rdma", 1, 2, "D1", 6), ("rdma", 0, 2, "T1", 3), ("tcp", 0, 2, "T1", 3)):
                cfgs.append({"world": 2, "transport": tp, "stripe": st, "wire": 2, "nslot": n, "mode": mode, "calls": calls,
                             "kinds": "SPH" if st else "SH", "level": lev})
        for n, lanes, calls, kinds in ((2, 1, 5, "axzs"), (3, 1, 7, "axzs"), (4, 1, 9, "axzs"), (2, 1, 2, "cy"), (4, 1, 2, "cy"), (4, 2, 2, "cy"), (8, 4, 1, "c"), (8, 4, 2, "cy")):
            for lev in ("basic", "full"):
                cfgs.append({"world": 4, "nslot": n, "lanes": lanes, "mode": "D1", "calls": calls, "kinds": kinds, "level": lev})
        for n, lanes, calls, kinds in ((2, 1, 1, "a"), (2, 1, 2, "a")):
            for lev in ("none", "full"):
                cfgs.append({"world": 4, "nslot": n, "lanes": lanes, "mode": "D1", "calls": calls, "kinds": kinds, "level": lev})
    if which in ("w4", "all"):
        for n in range(2, 9):
            for lanes in (1, 2, 3, 4):
                if n % lanes or n // lanes < 2:
                    continue
                m = n // lanes
                # prompt shape, smaller: one gather of lanes + 1 chunks (every lane in flight at once)
                cfgs.append({"world": 4, "nslot": n, "lanes": lanes, "mode": "D1", "calls": 1, "kinds": "c",
                             "shape": "prompt1", "stripe": 0, "wire": 1})
                # decode shape: calls that fit one slot (lane 0 only), every slot of lane 0 reused at least twice
                cfgs.append({"world": 4, "nslot": n, "lanes": lanes, "mode": "D1", "calls": 2 * m + 1, "kinds": "axzs",
                             "shape": "decode", "stripe": 0, "wire": 1})
                # prompt shape: chunked calls over every lane (gather of lanes+1 chunks, exchange of 2 chunks)
                cfgs.append({"world": 4, "nslot": n, "lanes": lanes, "mode": "D1", "calls": 2 * m + 1, "kinds": "cy",
                             "shape": "prompt", "stripe": 0, "wire": 1})

    return cfgs


MUTATIONS = [
    # (name, config, what it removes, expected)
    ("rdma-nslot1", {"world": 2, "transport": "rdma", "stripe": 0, "wire": 1, "nslot": 1, "mode": "D1", "calls": 4, "kinds": "S"},
     "TF_COLL_SLOTS=1, refused at rdma_proxy.zig:204", "S1"),
    ("tcp-nslot1", {"world": 2, "transport": "tcp", "stripe": 0, "wire": 1, "nslot": 1, "mode": "D1", "calls": 4, "kinds": "S"},
     "TF_COLL_SLOTS=1 with tcp (refused at tcp_proxy.zig:486); RX is written by the rank's own recvmsg", "none"),
    ("rdma-stream", {"world": 2, "transport": "rdma", "stripe": 0, "wire": 1, "nslot": 2, "mode": "D1", "calls": 4, "kinds": "S", "mut": ["stream"]},
     "the next stage runs before this collect (gatherEnd on another stream, refused at rdma_proxy.zig:156)", "S2"),
    ("rdma-nogate", {"world": 2, "transport": "rdma", "stripe": 0, "wire": 1, "nslot": 2, "mode": "D1", "calls": 6, "kinds": "S", "mut": ["nogate"]},
     "coll.stage without its consumed wait (coll.hip:59-64): stream order alone already orders the reuse", "none"),
    ("rdma-nogate-stream", {"world": 2, "transport": "rdma", "stripe": 0, "wire": 1, "nslot": 2, "mode": "D1", "calls": 6, "kinds": "S", "mut": ["nogate", "stream"]},
     "both: no consumed wait and the next stage ahead of the collect", "S1"),
    ("rdma-stripe-handoff-T1", {"world": 2, "transport": "rdma", "stripe": 1, "wire": 1, "nslot": 2, "mode": "T1", "calls": 3, "kinds": "SP"},
     "hypothetical striped rdma (tcp_proxy's Second design over a second QP), dead/wedged peer", "none"),
    ("tcp-stripe-nowait", {"world": 2, "transport": "tcp", "stripe": 1, "wire": 1, "nslot": 2, "mode": "D1", "calls": 4, "kinds": "P", "mut": ["nowait"]},
     "the first proxy thread publishes done without waiting for the second cable's half (tcp_proxy.zig:142-151)", "S2"),
    ("rdma-stripe-nowait", {"world": 2, "transport": "rdma", "stripe": 1, "wire": 1, "nslot": 2, "mode": "D1", "calls": 4, "kinds": "P", "mut": ["nowait"]},
     "hypothetical striped rdma whose first thread does not wait for QP 2's half", "S2"),
    ("w4-depth", {"world": 4, "nslot": 4, "lanes": 2, "mode": "D1", "calls": 2, "kinds": "c", "mut": ["depth"]},
     "one stage more ahead than Layout.depth (two stages of one lane before its collect)", "S2"),
    ("w4-lane1slot", {"world": 4, "nslot": 4, "lanes": 4, "mode": "D1", "calls": 3, "kinds": "a", "mut": ["lanes"]},
     "one slot a lane (Layout.lanesFit refuses it, rdma_proxy.zig:502)", "S1"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run", help="one configuration (JSON on stdout)")
    a.add_argument("cfg", help="JSON object: world, transport, nslot, lanes, stripe, wire, mode, calls, kinds, mut")
    a.add_argument("--max-states", type=int, default=4_000_000)
    s = sub.add_parser("sweep", help="the card's sweep; one JSON line per configuration")
    s.add_argument("which", choices=("w2", "w4", "all", "mut", "w2deep", "w4b", "porval", "w4t1", "w2deep-rdma"))
    s.add_argument("--jobs", type=int, default=4)
    s.add_argument("--max-states", type=int, default=4_000_000)
    s.add_argument("--out", required=True)
    s.add_argument("--only", default="", help="substring filter on the configuration's JSON")
    sub.add_parser("s3", help="the static S3 check")
    t = sub.add_parser("traces", help="readable counterexample traces of the controls into a directory")
    t.add_argument("dir")
    args = ap.parse_args()
    if args.cmd == "s3":
        print(json.dumps(s3_check()))
        return
    if args.cmd == "traces":
        import os
        os.makedirs(args.dir, exist_ok=True)
        for name, cfg, what, exp in MUTATIONS:
            c = dict(cfg)
            c.update(name=name, what=what, expected=exp, max_states=3_000_000)
            res = run_one(c)
            if res.get("violation"):
                with open(os.path.join(args.dir, name + ".txt"), "w") as f:
                    f.write(trace_text(res))
                print(name, res["violation"]["prop"], len(res["trace"]), "steps")
            else:
                print(name, "no violation (%s)" % ("exhausted" if res["exhausted"] else "not exhausted"))
        return
    if args.cmd == "run":
        cfg = json.loads(args.cfg)
        cfg["max_states"] = args.max_states
        print(json.dumps(run_one(cfg)))
        return
    if args.which == "mut":
        cfgs = []
        for name, cfg, what, exp in MUTATIONS:
            c = dict(cfg)
            c.update(name=name, what=what, expected=exp)
            cfgs.append(c)
    else:
        cfgs = sweep_configs(args.which)
    cfgs = [c for c in cfgs if args.only in json.dumps(c)]
    for c in cfgs:
        c["max_states"] = args.max_states
    import multiprocessing as mp
    with mp.Pool(args.jobs) as pool, open(args.out, "a") as f:
        for res in pool.imap_unordered(run_one, cfgs):
            f.write(json.dumps(res) + "\n")
            f.flush()
            v = res.get("violation")
            print("%-70s states=%-9d exhausted=%s %ss %s" % (json.dumps({k: res[k] for k in ("world", "transport", "nslot", "lanes", "stripe", "wire", "mode", "calls", "name") if k in res}),
                                                          res["states"], res["exhausted"], res["seconds"], (v["prop"] + ": " + v["msg"]) if v else ""), flush=True)


if __name__ == "__main__":
    main()
