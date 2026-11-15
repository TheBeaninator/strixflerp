// SPDX-License-Identifier: MIT
// Self-test of runtime.flerp.stage_relay without a GPU (tests/test_stage_relay.py builds and runs it):
// loopback ring, frame checks, injected faults, idempotent abort, and a TCP ring of forked processes on localhost
// with a killed rank. Prints one line per check, "ok <name>" or "FAIL <name>: why"; exits 1 on any failure.
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "flerp/stage_relay/relay.hpp"

using namespace flerp::stage_relay;
namespace {
int failures = 0;
void Check(bool ok, const std::string& name, const std::string& why = "") {
  std::printf("%s %s%s%s\n", ok ? "ok" : "FAIL", name.c_str(), ok || why.empty() ? "" : ": ", ok ? "" : why.c_str());
  if (!ok) ++failures;
}

Config Base(std::uint32_t world, std::uint32_t rank) {
  Config c;
  c.world = world;
  c.rank = rank;
  c.epoch = 7;
  c.hidden = 8;
  c.vocab = 16;
  c.max_rows = 4;
  c.plan.fill(0xAB);
  c.deadline = std::chrono::milliseconds(2000);
  return c;
}

std::vector<Config> Ring(std::uint32_t world) {
  std::vector<Config> v;
  for (std::uint32_t r = 0; r < world; ++r) v.push_back(Base(world, r));
  return v;
}

std::vector<std::uint8_t> Rows(std::uint32_t rows, float seed) {
  std::vector<float> f(rows * 8);
  for (std::size_t i = 0; i < f.size(); ++i) f[i] = seed + static_cast<float>(i);
  std::vector<std::uint8_t> b(f.size() * 4);
  std::memcpy(b.data(), f.data(), b.size());
  return b;
}

// Rank 0 sends a chunk; ranks 1..n-1 forward it; the last replies. Returns the number of ranks that threw.
int PassOnce(std::vector<std::unique_ptr<Relay>>& ring, Op op, std::uint32_t rows) {
  int threw = 0;
  const auto payload = Rows(rows, 1.0F);
  try {
    ring[0]->Send(op, 0, rows, kWantLogits, 0, payload);
  } catch (const RelayError&) {
    ++threw;
  }
  for (std::size_t r = 1; r < ring.size(); ++r) {
    try {
      const Frame f = ring[r]->Recv();
      if (r + 1 == ring.size()) ring[r]->Send(Op::kReply, 0, 1, kWantLogits, 42, {});
      else ring[r]->Send(f.op(), f.header.start_pos, f.header.rows, f.header.flags, 0, f.payload);
    } catch (const RelayError&) {
      ++threw;
    }
  }
  try {
    const Frame f = ring[0]->Recv();
    if (f.op() != Op::kReply || f.header.token != 42) ++threw;
  } catch (const RelayError&) {
    ++threw;
  }
  return threw;
}

void LoopbackTests() {
  {
    auto ring = LoopbackRing(Ring(4));
    bool ok = true;
    for (auto& r : ring) ok = ok && !r->poisoned();
    Check(ok, "loopback handshake world 4");
    Check(PassOnce(ring, Op::kPrefillChunk, 3) == 0 && PassOnce(ring, Op::kDecode, 1) == 0, "loopback chunk and decode round trip");
    // payload bytes arrive unchanged
    const auto payload = Rows(2, 5.0F);
    ring[0]->Send(Op::kPrefillChunk, 9, 2, 0, 0, payload);
    const Frame f = ring[1]->Recv();
    Check(f.payload == payload && f.header.start_pos == 9 && f.header.rows == 2, "payload and fields arrive unchanged");
  }
  for (const auto kind : {Fault::Kind::kDrop, Fault::Kind::kStaleEpoch, Fault::Kind::kReplay, Fault::Kind::kBadBytes}) {
    auto ring = LoopbackRing(Ring(4));
    (void)PassOnce(ring, Op::kDecode, 1);              // ordinals 1 on every link
    bool fired = false;
    ring[1]->SetFault({kind, 2}, [&](const Fault&) { fired = true; });
    int threw = PassOnce(ring, Op::kDecode, 1);
    if (kind == Fault::Kind::kReplay && threw == 0) threw = PassOnce(ring, Op::kDecode, 1);   // the stale copy is read next
    // drain: every rank still unpoisoned is asked for its next call, as its own loop would
    for (std::size_t r = 1; r < ring.size(); ++r) {
      if (!ring[r]->poisoned()) {
        try {
          (void)ring[r]->Recv();
        } catch (const RelayError&) {
        }
      }
    }
    bool all = true;
    std::string reasons;
    for (auto& r : ring) {
      all = all && r->poisoned();
      reasons += "[" + r->reason() + "] ";
    }
    Check(fired && threw > 0 && all, std::string("fault ") + FaultName(kind) + " poisons every rank", reasons);
    if (kind == Fault::Kind::kStaleEpoch) Check(ring[2]->reason().find("epoch 6") != std::string::npos, "stale epoch is named", ring[2]->reason());
    if (kind == Fault::Kind::kReplay) Check(ring[2]->reason().find("ordinal") != std::string::npos, "replayed frame refused by ordinal", ring[2]->reason());
  }
  {
    auto cfgs = Ring(4);
    cfgs[2].epoch = 6;
    auto ring = LoopbackRing(cfgs);
    Check(ring[2]->poisoned() && ring[3]->poisoned(), "wrong epoch refused at the handshake", ring[2]->reason() + " / " + ring[3]->reason());
  }
  {
    auto cfgs = Ring(2);
    cfgs[1].plan[0] = 0;
    auto ring = LoopbackRing(cfgs);
    Check(ring[0]->poisoned() && ring[1]->poisoned(), "another plan refused at the handshake");
  }
  {
    auto ring = LoopbackRing(Ring(2));
    ring[0]->Abort("first");
    ring[0]->Abort("second");
    bool refused = false;
    try {
      ring[0]->Send(Op::kDecode, 0, 1, 0, 0, Rows(1, 0));
    } catch (const RelayError&) {
      refused = true;
    }
    Check(ring[0]->reason() == "first" && refused, "abort is idempotent and a poisoned relay refuses");
    bool peer = false;
    try {
      (void)ring[1]->Recv();
    } catch (const RelayError&) {
      peer = true;
    }
    Check(peer && ring[1]->poisoned(), "abort reaches the neighbour");
  }
  {
    auto ring = LoopbackRing(Ring(2));
    bool refused = false;
    try {
      ring[0]->Send(Op::kDecode, 0, 2, 0, 0, Rows(2, 0));   // a decode frame of two rows
    } catch (const RelayError&) {
      refused = true;
    }
    Check(refused, "a sender refuses a malformed frame");
  }
}

// TCP: world 3 of forked processes on 127.0.0.1. `fault` on rank 1. Returns {exit codes, seconds to exit}.
struct TcpResult {
  std::vector<int> codes;
  double seconds{0};
};
// late_ms: the last rank starts that much later (a stage still loading), as in FLERP-P5's serve step where rank 3
// loaded 16 min after rank 2 and rank 1's HELLO wait ran out on the 120 s frame deadline.
TcpResult TcpRun(std::uint16_t port, const std::string& fault, std::uint64_t epoch1 = 7, std::uint32_t world = 3, int late_ms = 0) {
  std::vector<pid_t> pids;
  const auto t0 = std::chrono::steady_clock::now();
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    const pid_t pid = ::fork();
    if (pid == 0) {
      try {
        if (late_ms > 0 && rank + 1 == world) ::usleep(static_cast<useconds_t>(late_ms) * 1000U);
        Config c = Base(world, rank);
        if (rank == 1) c.epoch = epoch1;
        auto relay = ConnectRing(c, {"127.0.0.1"}, port, std::chrono::milliseconds(5000));
        if (rank == 1 && !fault.empty()) relay->SetFault(ParseFault(fault));
        for (int step = 0; step < 20; ++step) {
          if (rank == 0) {
            relay->Send(Op::kDecode, static_cast<std::uint32_t>(step), 1, kWantLogits, 0, Rows(1, 0));
            (void)relay->Recv();
          } else {
            const Frame f = relay->Recv();
            if (rank + 1 == world) relay->Send(Op::kReply, f.header.start_pos, 1, 0, 1, {});
            else relay->Send(f.op(), f.header.start_pos, f.header.rows, f.header.flags, 0, f.payload);
          }
        }
        ::_exit(0);
      } catch (const RelayError&) {
        ::_exit(3);
      }
    }
    pids.push_back(pid);
  }
  TcpResult res;
  for (const pid_t pid : pids) {
    int status = 0;
    (void)::waitpid(pid, &status, 0);
    res.codes.push_back(WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status));
  }
  res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return res;
}

std::string Codes(const TcpResult& r) {
  std::string s;
  for (const int c : r.codes) s += std::to_string(c) + " ";
  return s + "in " + std::to_string(r.seconds) + " s";
}

// The idle wait and the signal interrupt (executor.flerp.qwen35moe_abi's stage loop): world 2, rank 1 waits for
// frames with a long header deadline; rank 0 sends after `idle_ms` (longer than the 2 s frame deadline), then waits.
// interrupt: the parent sends SIGTERM to rank 1 while it waits; its handler calls Relay::Interrupt.
Relay* g_interrupt_relay = nullptr;
void OnTerm(int) {
  if (g_interrupt_relay != nullptr) g_interrupt_relay->Interrupt();
}

TcpResult IdleRun(std::uint16_t port, int idle_ms, bool interrupt) {
  std::vector<pid_t> pids;
  const auto t0 = std::chrono::steady_clock::now();
  for (std::uint32_t rank = 0; rank < 2; ++rank) {
    const pid_t pid = ::fork();
    if (pid == 0) {
      try {
        auto relay = ConnectRing(Base(2, rank), {"127.0.0.1"}, port, std::chrono::milliseconds(5000));
        if (rank == 1) {
          g_interrupt_relay = relay.get();
          struct sigaction sa {};
          sa.sa_handler = OnTerm;
          ::sigaction(SIGTERM, &sa, nullptr);
          for (int step = 0; step < 2; ++step) {
            const Frame f = relay->Recv(std::chrono::hours(1));
            relay->Send(Op::kReply, f.header.start_pos, 1, 0, 1, {});
          }
          ::_exit(0);
        }
        for (int step = 0; step < 2; ++step) {
          ::usleep(static_cast<useconds_t>(idle_ms) * 1000U);
          relay->Send(Op::kDecode, static_cast<std::uint32_t>(step), 1, kWantLogits, 0, Rows(1, 0));
          (void)relay->Recv();
        }
        ::_exit(0);
      } catch (const RelayError&) {
        ::_exit(rank == 1 && interrupt ? 5 : 3);
      }
    }
    pids.push_back(pid);
  }
  if (interrupt) {
    ::usleep(1000000);
    ::kill(pids[1], SIGTERM);
  }
  TcpResult res;
  for (const pid_t pid : pids) {
    int status = 0;
    (void)::waitpid(pid, &status, 0);
    res.codes.push_back(WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status));
  }
  res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return res;
}

void TcpTests(std::uint16_t port) {
  {
    auto r = IdleRun(port + 50, 3000, false);
    Check(r.codes == std::vector<int>{0, 0}, "tcp: a stage waits idle past the frame deadline with a long header deadline", Codes(r));
    r = IdleRun(port + 60, 4000, true);
    Check(r.codes == std::vector<int>{3, 5} && r.seconds < 6.0, "tcp: Interrupt from a signal handler ends a waiting receive", Codes(r));
  }
  auto r = TcpRun(port, "");
  Check(r.codes == std::vector<int>{0, 0, 0}, "tcp world 3: 20 round trips", Codes(r));
  r = TcpRun(port + 10, "kill:5");
  Check(r.codes == std::vector<int>{3, 128 + SIGKILL, 3} && r.seconds < 4.0, "tcp: a killed rank makes the others fail at once", Codes(r));
  r = TcpRun(port + 20, "drop:5");
  Check(r.codes == std::vector<int>{3, 3, 3} && r.seconds < 6.0, "tcp: a dropped frame fails every rank within the deadline", Codes(r));
  r = TcpRun(port + 30, "stale-epoch:5");
  Check(r.codes == std::vector<int>{3, 3, 3} && r.seconds < 4.0, "tcp: a stale-epoch frame fails every rank", Codes(r));
  r = TcpRun(port + 70, "", 7, 4, 3500);
  Check(r.codes == std::vector<int>{0, 0, 0, 0}, "tcp world 4: a last rank later than the frame deadline still joins (HELLO waits on the setup deadline)", Codes(r));
  r = TcpRun(port + 40, "", 6);
  Check(r.codes == std::vector<int>{3, 3, 3} && r.seconds < 6.0, "tcp: a rank of another epoch is refused at setup", Codes(r));
}
}  // namespace

int main(int argc, char** argv) {
  ::signal(SIGPIPE, SIG_IGN);
  LoopbackTests();
  TcpTests(static_cast<std::uint16_t>(argc > 1 ? std::stoi(argv[1]) : 39400));
  std::printf("%d failure(s)\n", failures);
  return failures == 0 ? 0 : 1;
}
