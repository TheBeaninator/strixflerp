// SPDX-License-Identifier: MIT
// SHAPE-TRANSFERS R6 part 2 (criteria gate/xfer-r6-spec.toml): the cost of a speculative verify forward of the
// qwen35moe executor at 1/2/4/8 rows against a 1-row decode step, on one GPU, one weights load.
//   xfer-verify-bench --model FILE.gguf --ids-file FILE [--context N] [--repeats N]
// Prefills the ids (ForwardPromptBatch), then at the following positions times, each a median of --repeats:
//   ForwardToken (graph on and off), ForwardVerificationChunk(rows) and ForwardPromptBatch(rows, start_pos).
// One JSON object a line on stdout. Built beside flerp-qwen35moe-generate (gate/xfer_verify_build.sh).
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "flerp/qwen35moe/stage.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using gufo::tokenization::TokenId;

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}
template <typename F>
std::vector<double> Time(F&& f, int warm, int reps) {
  std::vector<double> ms;
  for (int i = 0; i < warm + reps; ++i) {
    const auto t0 = Clock::now();
    f(i);
    (void)hipDeviceSynchronize();
    const double v = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    if (i >= warm) ms.push_back(v);
  }
  return ms;
}
std::string J(const std::vector<double>& v) {
  std::vector<double> s = v;
  std::sort(s.begin(), s.end());
  return "{\"median_ms\":" + std::to_string(Median(v)) + ",\"min_ms\":" + std::to_string(s.front()) + ",\"max_ms\":" +
         std::to_string(s.back()) + ",\"n\":" + std::to_string(v.size()) + "}";
}
}  // namespace

int main(int argc, char** argv) {
  std::string model, ids_file;
  std::uint32_t context = 4096;
  int reps = 30;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--model") model = argv[i + 1];
    else if (a == "--ids-file") ids_file = argv[i + 1];
    else if (a == "--context") context = static_cast<std::uint32_t>(std::stoul(argv[i + 1]));
    else if (a == "--repeats") reps = std::stoi(argv[i + 1]);
  }
  if (model.empty() || ids_file.empty()) {
    std::cerr << "usage: xfer-verify-bench --model F --ids-file F [--context N] [--repeats N]\n";
    return 2;
  }
  std::vector<TokenId> ids;
  {
    std::ifstream in(ids_file);
    for (long v; in >> v;) ids.push_back(static_cast<TokenId>(v));
  }
  std::string error;
  auto owner = gufo::core::GgufReader::OpenFile(model, &error);
  if (!owner) {
    std::cerr << "error: " << error << "\n";
    return 1;
  }
  const std::shared_ptr<const gufo::core::GgufReader> reader(std::move(owner));
  const auto t0 = Clock::now();
  auto gpu = gufo::hip::QwenGpuExecutor::CreateFromGguf(reader, &error, context);
  if (!gpu) {
    std::cerr << "error: " << error << "\n";
    return 1;
  }
  std::cout << "{\"event\":\"loaded\",\"load_s\":" << std::chrono::duration<double>(Clock::now() - t0).count()
            << ",\"max_prompt_batch\":" << gpu->GetMaxPromptBatch() << ",\"prompt\":" << ids.size() << "}" << std::endl;
  const auto P = static_cast<std::uint32_t>(ids.size());
  const auto tok = [&](std::size_t i) { return ids[i % ids.size()]; };
  try {
    gpu->Reset();
    (void)gpu->ForwardPromptBatch(ids, 0, true);
    (void)hipDeviceSynchronize();
    std::uint32_t pos = P;
    // decode step: graph on (the production path), then eager
    for (const bool graph : {true, false}) {
      flerp::qwen35moe::SetDecodeGraph(graph);
      const auto v = Time([&](int i) { (void)gpu->ForwardToken(tok(i), pos + static_cast<std::uint32_t>(i), true); }, 5, reps);
      pos += static_cast<std::uint32_t>(5 + reps);
      std::cout << "{\"event\":\"decode\",\"graph\":" << (graph ? "true" : "false") << ",\"from_pos\":" << pos << ",\"t\":" << J(v) << "}" << std::endl;
    }
    flerp::qwen35moe::SetDecodeGraph(true);
    // verify forwards at a fixed start position (no commit: the same rows are verified again each repeat)
    for (const std::size_t rows : {1, 2, 4, 8}) {
      std::vector<TokenId> cand(rows);
      for (std::size_t r = 0; r < rows; ++r) cand[r] = tok(pos + r);
      try {
        const auto v = Time([&](int) { (void)gpu->ForwardVerificationChunk(cand, pos, false); }, 5, reps);
        std::cout << "{\"event\":\"verify\",\"rows\":" << rows << ",\"pos\":" << pos << ",\"t\":" << J(v) << "}" << std::endl;
      } catch (const std::exception& e) {
        std::cout << "{\"event\":\"verify_error\",\"rows\":" << rows << ",\"what\":\"" << e.what() << "\"}" << std::endl;
      }
      try {
        const auto v = Time([&](int) { (void)gpu->ForwardPromptBatch(cand, pos, true); }, 5, reps);
        std::cout << "{\"event\":\"chunk\",\"rows\":" << rows << ",\"pos\":" << pos << ",\"t\":" << J(v) << "}" << std::endl;
      } catch (const std::exception& e) {
        std::cout << "{\"event\":\"chunk_error\",\"rows\":" << rows << ",\"what\":\"" << e.what() << "\"}" << std::endl;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
