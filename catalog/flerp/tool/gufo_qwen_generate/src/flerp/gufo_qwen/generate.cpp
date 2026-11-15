// SPDX-License-Identifier: MIT
// A greedy token generator over gufo's Qwen 3.5 executor, for recipe gates. It prints token ids, not prose:
// one JSON object a line on stdout, everything else on stderr.
//
//   flerp-gufo-qwen-generate --model FILE.gguf [--cpu] [--tokenize] [--context N] [--max-tokens N]
//                            (--prompt-file FILE | --ids-file FILE)...
//
// --prompt-file   a prompt as raw UTF-8 text: no chat template, encoded by the engine's tokenizer.
// --ids-file      a prompt as token ids separated by white space (the tokenizer is not asked).
// --cpu           gufo's CPU reference (models::QwenGenerator) instead of the HIP executor.
// --tokenize      print each prompt's token ids and generate nothing (implies --cpu: no device is touched).
//
// The decode is the engine's own: QwenGpuExecutor::Generate at its default sampling (temperature 0, so the
// unmodified argmax) and its production execution policy. No draft backend is attached.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/families/qwen/text/chat_template.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {

using gufo::tokenization::TokenId;
using Clock = std::chrono::steady_clock;

double Since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string JsonString(std::string_view text) {
  std::string out = "\"";
  char buffer[8];
  for (const unsigned char c : text) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += static_cast<char>(c);
    } else if (c < 0x20) {
      std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
      out += buffer;
    } else {
      out += static_cast<char>(c);
    }
  }
  return out + "\"";
}

template <typename Range>
std::string JsonIds(const Range& ids) {
  std::string out = "[";
  bool first = true;
  for (const auto id : ids) {
    if (!first) out += ",";
    out += std::to_string(static_cast<std::int64_t>(id));
    first = false;
  }
  return out + "]";
}

struct Prompt {
  std::string source;
  bool is_ids{false};
};

struct Result {
  std::vector<TokenId> ids;
  std::string text;
  double first_token_s{0.0};
  double total_s{0.0};
};

template <typename Engine>
Result Run(Engine& engine, const std::vector<TokenId>& prompt, std::size_t max_tokens) {
  Result result;
  gufo::models::GenerationOptions options;
  options.max_new_tokens = max_tokens;
  const auto start = Clock::now();
  (void)engine.Generate(prompt, options, [&](TokenId token, std::string_view piece) {
    if (result.ids.empty()) result.first_token_s = Since(start);
    result.ids.push_back(token);
    result.text.append(piece);
    return true;
  });
  result.total_s = Since(start);
  return result;
}

int Usage() {
  std::cerr << "usage: flerp-gufo-qwen-generate --model FILE.gguf [--cpu] [--tokenize] [--context N] [--max-tokens N] "
               "(--prompt-file FILE | --ids-file FILE)...\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::vector<Prompt> prompts;
  std::size_t max_tokens = 64;
  std::uint32_t context = 4096;
  bool cpu = false;
  bool tokenize_only = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const auto value = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
    if (arg == "--cpu") {
      cpu = true;
    } else if (arg == "--tokenize") {
      cpu = tokenize_only = true;
    } else if (arg == "--model" || arg == "--prompt-file" || arg == "--ids-file" || arg == "--max-tokens" ||
               arg == "--context") {
      const char* v = value();
      if (v == nullptr) return Usage();
      if (arg == "--model") model_path = v;
      else if (arg == "--prompt-file") prompts.push_back({v, false});
      else if (arg == "--ids-file") prompts.push_back({v, true});
      else if (arg == "--max-tokens") max_tokens = std::strtoull(v, nullptr, 10);
      else context = static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
    } else {
      return Usage();
    }
  }
  if (model_path.empty() || prompts.empty() || max_tokens == 0) return Usage();

  const auto load_start = Clock::now();
  std::string error;
  auto owner = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!owner) {
    std::cerr << "error: GGUF " << model_path << ": " << error << "\n";
    return 1;
  }
  const std::shared_ptr<const gufo::core::GgufReader> reader(std::move(owner));
  const std::string architecture(reader->GetMetadataString("general.architecture").value_or(""));

  std::unique_ptr<gufo::hip::QwenGpuExecutor> gpu;
  std::unique_ptr<gufo::models::QwenGenerator> host;
  if (cpu) {
    host = gufo::models::QwenGenerator::CreateFromGguf(*reader, &error);
  } else {
    gpu = gufo::hip::QwenGpuExecutor::CreateFromGguf(reader, &error, context);
  }
  if (!gpu && !host) {
    std::cerr << "error: " << (cpu ? "Qwen CPU generator" : "Qwen GPU executor") << " (general.architecture="
              << architecture << "): " << error << "\n";
    return 1;
  }
  const double load_s = Since(load_start);
  const auto& config = cpu ? host->GetConfig() : gpu->GetConfig();
  const auto& tokenizer = cpu ? host->GetTokenizer() : gpu->GetTokenizer();
  std::cout << "{\"event\":\"loaded\",\"engine\":" << JsonString(cpu ? "gufo-cpu" : "gufo-hip")
            << ",\"architecture\":" << JsonString(architecture) << ",\"layers\":" << config.num_layers
            << ",\"hidden\":" << config.hidden_size << ",\"intermediate\":" << config.intermediate_size
            << ",\"heads\":" << config.num_attention_heads << ",\"kv_heads\":" << config.num_key_value_heads
            << ",\"vocab\":" << config.vocab_size << ",\"mtp_layers\":" << config.mtp_num_layers
            << ",\"context\":" << context << ",\"load_s\":" << load_s;
  if (gpu) {
    const auto usage = gpu->GetMemoryUsage();
    std::cout << ",\"device_bytes\":" << usage.TotalBytes()
              << ",\"weight_resident_bytes\":" << gpu->GetModel().GetResidentBytes()
              << ",\"weight_regions\":" << gpu->GetModel().GetWeightRegionCount();
  }
  std::cout << "}" << std::endl;

  for (std::size_t index = 0; index < prompts.size(); ++index) {
    std::ifstream file(prompts[index].source, std::ios::binary);
    if (!file) {
      std::cerr << "error: cannot read " << prompts[index].source << "\n";
      return 1;
    }
    const std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::vector<TokenId> ids;
    if (prompts[index].is_ids) {
      std::istringstream in(content);
      for (long long id = 0; in >> id;) ids.push_back(static_cast<TokenId>(id));
    } else {
      ids = gufo::tokenization::QwenChatTemplate::EncodeRendered(tokenizer, content, 0, content.size(), {}, {});
    }
    if (ids.empty() || ids.size() + max_tokens >= context) {
      std::cerr << "error: prompt " << index << " has " << ids.size() << " tokens; with " << max_tokens
                << " new tokens it must fit a context of " << context << "\n";
      return 1;
    }
    if (tokenize_only) {
      std::cout << "{\"event\":\"tokenized\",\"prompt\":" << index << ",\"source\":"
                << JsonString(prompts[index].source) << ",\"prompt_ids\":" << JsonIds(ids) << "}" << std::endl;
      continue;
    }
    Result result;
    try {
      result = cpu ? Run(*host, ids, max_tokens) : Run(*gpu, ids, max_tokens);
    } catch (const std::exception& exception) {
      std::cerr << "error: prompt " << index << ": " << exception.what() << "\n";
      return 1;
    }
    const double decode_s = result.total_s - result.first_token_s;
    const double decode_tps =
        result.ids.size() > 1 && decode_s > 0.0 ? static_cast<double>(result.ids.size() - 1) / decode_s : 0.0;
    std::cout << "{\"event\":\"generated\",\"prompt\":" << index << ",\"source\":" << JsonString(prompts[index].source)
              << ",\"prompt_ids\":" << JsonIds(ids) << ",\"ids\":" << JsonIds(result.ids)
              << ",\"text\":" << JsonString(result.text) << ",\"first_token_s\":" << result.first_token_s
              << ",\"total_s\":" << result.total_s << ",\"decode_tok_s\":" << decode_tps << "}" << std::endl;
  }
  return 0;
}
