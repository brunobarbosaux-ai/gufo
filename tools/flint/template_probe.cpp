// G2 probe: render the packed flint fixtures with FlintChatTemplate, compare
// the rendered bytes to the oracle's /apply-template output, then tokenize and
// compare ids to the oracle's /tokenize output.
//
// Cases file layout: see tools/flint/g2_cases.py.
//
// Usage: template_probe MODEL.gguf CASES.bin [MAX_REPORTS]
//   exits 0 when every fixture matches, 1 on mismatch.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

bool ReadPod(std::ifstream& in, void* out, std::size_t bytes) {
  in.read(static_cast<char*>(out), static_cast<std::streamsize>(bytes));
  return in.gcount() == static_cast<std::streamsize>(bytes);
}

bool ReadString(std::ifstream& in, std::string* out) {
  std::uint32_t size = 0;
  if (!ReadPod(in, &size, sizeof(size))) {
    return false;
  }
  out->assign(size, '\0');
  return size == 0 || ReadPod(in, &(*out)[0], size);
}

std::string Readable(std::string_view text, std::size_t limit = 120) {
  std::string out;
  for (const char c : text) {
    if (out.size() >= limit) {
      out += "...";
      break;
    }
    if (c == '\n') {
      out += "\\n";
    } else if (c == '\r') {
      out += "\\r";
    } else {
      out += c;
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::fprintf(stderr, "usage: %s MODEL.gguf CASES.bin [MAX_REPORTS]\n",
                 argv[0]);
    return 2;
  }
  const int max_reports = argc == 4 ? std::atoi(argv[3]) : 10;

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (reader == nullptr) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return 2;
  }
  if (!gufo::tokenization::FlintChatTemplate::ValidateGgufTemplate(*reader,
                                                                  &error)) {
    std::fprintf(stderr, "template validation failed: %s\n", error.c_str());
    return 1;
  }
  const auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (tokenizer == nullptr) {
    std::fprintf(stderr, "tokenizer load failed: %s\n", error.c_str());
    return 2;
  }

  std::ifstream in(argv[2], std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "cannot open cases: %s\n", argv[2]);
    return 2;
  }
  std::uint32_t fixture_count = 0;
  if (!ReadPod(in, &fixture_count, sizeof(fixture_count))) {
    std::fprintf(stderr, "truncated cases header\n");
    return 2;
  }

  std::size_t render_failures = 0;
  std::size_t token_failures = 0;
  std::size_t reported = 0;
  for (std::uint32_t index = 0; index < fixture_count; ++index) {
    using gufo::tokenization::ChatMessage;
    using gufo::tokenization::ChatRole;
    using gufo::tokenization::ChatTool;

    std::string name;
    std::string expected_rendered;
    std::vector<std::uint32_t> expected_ids;
    if (!ReadString(in, &name) || !ReadString(in, &expected_rendered)) {
      std::fprintf(stderr, "truncated fixture %u\n", index);
      return 2;
    }
    std::uint32_t id_count = 0;
    if (!ReadPod(in, &id_count, sizeof(id_count))) {
      return 2;
    }
    expected_ids.resize(id_count);
    if (id_count != 0 &&
        !ReadPod(in, &expected_ids[0], id_count * sizeof(std::uint32_t))) {
      return 2;
    }

    std::uint32_t tool_count = 0;
    if (!ReadPod(in, &tool_count, sizeof(tool_count))) {
      return 2;
    }
    std::vector<ChatTool> tools(tool_count);
    for (auto& tool : tools) {
      if (!ReadString(in, &tool.name) || !ReadString(in, &tool.description) ||
          !ReadString(in, &tool.parameters_json)) {
        return 2;
      }
    }

    std::uint32_t message_count = 0;
    if (!ReadPod(in, &message_count, sizeof(message_count))) {
      return 2;
    }
    std::vector<ChatMessage> messages;
    messages.reserve(message_count);
    for (std::uint32_t m = 0; m < message_count; ++m) {
      std::uint8_t role = 0;
      ChatMessage message;
      std::uint32_t call_count = 0;
      if (!ReadPod(in, &role, sizeof(role))) {
        return 2;
      }
      message.role = static_cast<ChatRole>(role);
      if (!ReadString(in, &message.content) ||
          !ReadString(in, &message.thought) ||
          !ReadPod(in, &call_count, sizeof(call_count))) {
        return 2;
      }
      message.tool_calls.resize(call_count);
      for (auto& call : message.tool_calls) {
        std::uint32_t arg_count = 0;
        if (!ReadString(in, &call.name) ||
            !ReadPod(in, &arg_count, sizeof(arg_count))) {
          return 2;
        }
        call.arguments.resize(arg_count);
        for (auto& argument : call.arguments) {
          std::uint8_t is_string = 1;
          if (!ReadString(in, &argument.name) ||
              !ReadPod(in, &is_string, sizeof(is_string)) ||
              !ReadString(in, &argument.value)) {
            return 2;
          }
          argument.is_string = is_string != 0;
        }
      }
      messages.push_back(std::move(message));
    }

    // The oracle suppresses the generation prompt when the conversation ends on
    // an assistant turn (continue mode).
    gufo::tokenization::ChatTemplateOptions options;
    options.add_generation_prompt =
        !messages.empty() && messages.back().role != ChatRole::kAssistant;

    const auto rendered =
        gufo::tokenization::FlintChatTemplate::Render(messages, tools, options,
                                                      &error);
    if (!rendered.has_value()) {
      std::printf("RENDER FAILED %s: %s\n", name.c_str(), error.c_str());
      ++render_failures;
      continue;
    }
    if (*rendered != expected_rendered) {
      ++render_failures;
      if (reported < static_cast<std::size_t>(max_reports)) {
        ++reported;
        std::size_t diff = 0;
        while (diff < rendered->size() && diff < expected_rendered.size() &&
               (*rendered)[diff] == expected_rendered[diff]) {
          ++diff;
        }
        std::printf("BYTES DIFF %s (first at %zu, lengths %zu vs %zu)\n",
                    name.c_str(), diff, rendered->size(),
                    expected_rendered.size());
        std::printf("  expected bytes:");
        for (std::size_t k = diff; k < expected_rendered.size() && k < diff + 12;
             ++k) {
          std::printf(" %02X", static_cast<unsigned char>(expected_rendered[k]));
        }
        std::printf("\n  actual bytes  :");
        for (std::size_t k = diff; k < rendered->size() && k < diff + 12; ++k) {
          std::printf(" %02X", static_cast<unsigned char>((*rendered)[k]));
        }
        std::printf("\n");
        std::printf("  expected: %s\n",
                    Readable(std::string_view(expected_rendered).substr(diff))
                        .c_str());
        std::printf("  actual  : %s\n",
                    Readable(std::string_view(*rendered).substr(diff)).c_str());
      }
      continue;
    }

    const auto ids = tokenizer->Encode(*rendered);
    if (ids != expected_ids) {
      ++token_failures;
      if (reported < static_cast<std::size_t>(max_reports)) {
        ++reported;
        std::size_t diff = 0;
        while (diff < ids.size() && diff < expected_ids.size() &&
               ids[diff] == expected_ids[diff]) {
          ++diff;
        }
        std::printf("ID DIFF %s at %zu (%zu vs %zu ids)\n", name.c_str(), diff,
                    expected_ids.size(), ids.size());
      }
    }
  }

  std::printf("G2: %zu/%u fixtures byte-identical, %zu token mismatches%s\n",
              static_cast<std::size_t>(fixture_count) - render_failures,
              fixture_count, token_failures,
              (render_failures == 0 && token_failures == 0) ? "" : " — FAILED");
  return (render_failures == 0 && token_failures == 0) ? 0 : 1;
}
