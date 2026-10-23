// G1 probe: encode the packed corpus with Gufo's tokenizer and compare the
// ids against the llama.cpp golden.
//
// Cases file layout (written by tools/flint/g1_cases.py):
//   u32 case_count
//   per case: u32 text_bytes, text, u32 id_count, u32 ids[id_count]
//
// Usage: tokenizer_probe MODEL.gguf CASES.bin [MAX_REPORTS]
//   exits 0 when every case matches, 1 on mismatch (reports the first
//   MAX_REPORTS divergences, default 10).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

bool ReadPod(std::ifstream& in, void* out, std::size_t bytes) {
  in.read(static_cast<char*>(out), static_cast<std::streamsize>(bytes));
  return in.gcount() == static_cast<std::streamsize>(bytes);
}

std::string Snippet(const std::string& text, std::size_t limit = 60) {
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
    } else if (static_cast<unsigned char>(c) < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\x%02X", static_cast<unsigned char>(c));
      out += buf;
    } else {
      out += c;
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::fprintf(stderr,
                 "usage: %s MODEL.gguf CASES.bin [MAX_REPORTS]\n"
                 "       %s MODEL.gguf --text TEXT\n",
                 argv[0], argv[0]);
    return 2;
  }
  const bool encode_text = std::string(argv[2]) == "--text";
  const int max_reports = argc == 4 ? std::atoi(argv[3]) : 10;

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (reader == nullptr) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return 2;
  }
  const auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (tokenizer == nullptr) {
    std::fprintf(stderr, "tokenizer load failed: %s\n", error.c_str());
    return 2;
  }

  if (encode_text) {
    const auto ids = tokenizer->Encode(argv[3]);
    for (const auto id : ids) {
      std::printf("%u ", id);
    }
    std::printf("\n");
    return 0;
  }

  std::ifstream in(argv[2], std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "cannot open cases: %s\n", argv[2]);
    return 2;
  }
  std::uint32_t case_count = 0;
  if (!ReadPod(in, &case_count, sizeof(case_count))) {
    std::fprintf(stderr, "truncated cases header\n");
    return 2;
  }

  std::size_t mismatches = 0;
  std::size_t reported = 0;
  std::size_t total_ids = 0;
  for (std::uint32_t index = 0; index < case_count; ++index) {
    std::uint32_t text_bytes = 0;
    std::uint32_t id_count = 0;
    if (!ReadPod(in, &text_bytes, sizeof(text_bytes))) {
      std::fprintf(stderr, "truncated case %u\n", index);
      return 2;
    }
    std::string text(text_bytes, '\0');
    if (text_bytes != 0 && !ReadPod(in, &text[0], text_bytes)) {
      std::fprintf(stderr, "truncated text in case %u\n", index);
      return 2;
    }
    if (!ReadPod(in, &id_count, sizeof(id_count))) {
      std::fprintf(stderr, "truncated ids header in case %u\n", index);
      return 2;
    }
    std::vector<std::uint32_t> expected(id_count);
    if (id_count != 0 &&
        !ReadPod(in, &expected[0], id_count * sizeof(std::uint32_t))) {
      std::fprintf(stderr, "truncated ids in case %u\n", index);
      return 2;
    }

    const auto actual = tokenizer->Encode(text);
    total_ids += expected.size();
    bool equal = actual.size() == expected.size();
    if (equal) {
      for (std::size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
          equal = false;
          break;
        }
      }
    }
    if (equal) {
      continue;
    }
    ++mismatches;
    if (reported < static_cast<std::size_t>(max_reports)) {
      ++reported;
      std::size_t first_diff = 0;
      while (first_diff < actual.size() && first_diff < expected.size() &&
             actual[first_diff] == expected[first_diff]) {
        ++first_diff;
      }
      std::printf("MISMATCH case %u: \"%s\"\n", index, Snippet(text).c_str());
      std::printf("  expected %zu ids, got %zu (first diff at %zu)\n",
                  expected.size(), actual.size(), first_diff);
      std::printf("  expected[%zu..] =", first_diff);
      for (std::size_t i = first_diff;
           i < expected.size() && i < first_diff + 8; ++i) {
        std::printf(" %u", expected[i]);
      }
      std::printf("\n  actual[%zu..]   =", first_diff);
      for (std::size_t i = first_diff; i < actual.size() && i < first_diff + 8;
           ++i) {
        std::printf(" %u", actual[i]);
      }
      std::printf("\n");
    }
  }

  std::printf("G1: %zu/%u cases match (%zu ids)%s\n", case_count - mismatches,
              case_count, total_ids, mismatches == 0 ? "" : " — FAILED");
  return mismatches == 0 ? 0 : 1;
}
