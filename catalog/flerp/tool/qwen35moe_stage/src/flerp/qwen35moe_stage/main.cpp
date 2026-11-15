// SPDX-License-Identifier: MIT
// tool.flerp.qwen35moe_stage: the process of a stage 1..N-1 of executor.flerp.qwen35moe_abi (docs/PHASE5-FOURNODE-
// DESIGN.md section 3.9). Rank 0 is the executor inside whatever opened it (the front end, the conformance driver);
// this program loads one block range and executes rank 0's mirrored calls until CLOSE.
//
//   flerp-qwen35moe-stage KEY=VALUE ...
//       the keys of flerp_open (stages.hpp) with rank = 1..world-1 and peers = HOST[,HOST...]:PORT. Exit status 0
//       after CLOSE or SIGTERM / SIGINT (at any point: a signal during the load ends the process when the load
//       returns), 3 when the relay failed, 1 when the load or the setup failed. One JSON line a step on stdout.
//   flerp-qwen35moe-stage --loopback [--max-tokens N] --ids-file F [--ids-file F ...] KEY=VALUE ...
//       every stage in this process: the executor is opened through the ABI with peers = loopback (when world > 1)
//       and each prompt (token ids, white space separated) is fed in forwards of caps.max_forward, then decoded
//       greedily one id a forward for N tokens (default 32). One {"event": "generated"} line a prompt.
#include <signal.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "flerp/executor.h"
#include "flerp/qwen35moe_abi/stages.hpp"

namespace qabi = flerp::qwen35moe_abi;

namespace {

std::atomic<bool> g_stop{false};

void OnSignal(int) {
  g_stop.store(true);
  qabi::InterruptStage();
}

int Usage() {
  std::fprintf(stderr,
               "usage: flerp-qwen35moe-stage KEY=VALUE ...    (rank=1..world-1 peers=HOST[,HOST...]:PORT weights=F world=W ...)\n"
               "       flerp-qwen35moe-stage --loopback [--max-tokens N] --ids-file F ... KEY=VALUE ...\n");
  return 2;
}

int Loopback(const std::vector<std::pair<std::string, std::string>>& keys, const std::vector<std::string>& files, long max_tokens) {
  std::vector<flerp_kv> kv;
  bool peers = false, world_gt_1 = false;
  for (const auto& [k, v] : keys) {
    kv.push_back({k.c_str(), v.c_str()});
    peers = peers || k == "peers";
    world_gt_1 = world_gt_1 || (k == "world" && v != "1");
  }
  static const char* const kPeers = "peers";
  static const char* const kLoop = "loopback";
  if (world_gt_1 && !peers) kv.push_back({kPeers, kLoop});
  flerp_open_args oa{sizeof(oa), "flerp-qwen35moe-stage --loopback", kv.data(), kv.size()};
  flerp_executor* ex = nullptr;
  if (flerp_open(&oa, &ex) != FLERP_OK) {
    std::fprintf(stderr, "flerp_open: %s\n", flerp_last_error(nullptr));
    return 1;
  }
  flerp_caps caps{};
  caps.size = sizeof(caps);
  int rc = 0;
  if (flerp_get_caps(ex, &caps) != FLERP_OK) rc = 1;
  for (const std::string& path : files) {
    if (rc != 0) break;
    std::ifstream in(path);
    std::vector<flerp_token> prompt;
    for (long long id = 0; in >> id;) prompt.push_back(static_cast<flerp_token>(id));
    int32_t seq = -1;
    flerp_seq_args sa{};
    sa.size = sizeof(sa);
    if (prompt.empty() || flerp_seq_new(ex, &sa, &seq) != FLERP_OK) {
      std::fprintf(stderr, "%s: %s\n", path.c_str(), prompt.empty() ? "no ids" : flerp_last_error(ex));
      rc = 1;
      break;
    }
    flerp_token tok = 0;
    flerp_rows rows{};
    rows.size = sizeof(rows), rows.tokens = &tok, rows.tokens_cap = 1;
    flerp_want want{};
    want.size = sizeof(want), want.delivery = FLERP_OUT_TOKENS, want.rows = 1;
    flerp_want none{};
    none.size = sizeof(none);
    std::vector<flerp_token> out;
    bool ok = true;
    for (size_t off = 0; ok && off < prompt.size(); off += (size_t)caps.max_forward) {
      const size_t n = std::min(prompt.size() - off, (size_t)caps.max_forward);
      const bool last = off + n == prompt.size();
      ok = flerp_forward(ex, seq, prompt.data() + off, n, last ? &want : &none, &rows, nullptr, 0) == FLERP_OK;
    }
    while (ok && (long)out.size() < max_tokens) {
      out.push_back(tok);
      const flerp_token in_id = tok;
      if ((long)out.size() < max_tokens) ok = flerp_forward(ex, seq, &in_id, 1, &want, &rows, nullptr, 0) == FLERP_OK;
    }
    if (!ok) {
      std::fprintf(stderr, "%s: %s\n", path.c_str(), flerp_last_error(ex));
      rc = 1;
    }
    std::string ids = "[";
    for (size_t i = 0; i < out.size(); ++i) ids += (i ? "," : "") + std::to_string(out[i]);
    qabi::Emit("{\"event\":\"generated\",\"prompt_tokens\":" + std::to_string(prompt.size()) + ",\"ids\":" + ids + "]}");
    flerp_seq_free(ex, seq);
  }
  flerp_close(ex);
  qabi::Emit("{\"event\":\"exit\",\"rc\":" + std::to_string(rc) + "}");
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  struct sigaction sa {};
  sa.sa_handler = OnSignal;
  sigemptyset(&sa.sa_mask);
  ::sigaction(SIGTERM, &sa, nullptr);
  ::sigaction(SIGINT, &sa, nullptr);

  bool loopback = false;
  long max_tokens = 32;
  std::vector<std::string> files;
  std::vector<std::pair<std::string, std::string>> keys;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool more = i + 1 < argc;
    if (a == "--loopback") loopback = true;
    else if (a == "--max-tokens" && more) max_tokens = std::atol(argv[++i]);
    else if (a == "--ids-file" && more) files.emplace_back(argv[++i]);
    else if (a.find('=') != std::string::npos && a.rfind("--", 0) != 0) keys.emplace_back(a.substr(0, a.find('=')), a.substr(a.find('=') + 1));
    else return Usage();
  }
  if (loopback) return files.empty() ? Usage() : Loopback(keys, files, max_tokens);
  if (!files.empty()) return Usage();
  qabi::Options o;
  for (const auto& [k, v] : keys) {
    std::string error;
    if (!qabi::SetKey(o, k, v, &error)) {
      std::fprintf(stderr, "flerp-qwen35moe-stage: %s\n", error.c_str());
      return 2;
    }
  }
  const int rc = qabi::StageMain(o, g_stop);
  qabi::Emit("{\"event\":\"exit\",\"rank\":" + std::to_string(o.rank) + ",\"rc\":" + std::to_string(rc) + "}");
  return rc;
}
