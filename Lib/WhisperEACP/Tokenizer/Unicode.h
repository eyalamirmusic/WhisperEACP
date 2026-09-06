#pragma once

#include <WhisperEACP/Core/Core.h>

#include <cstddef>
#include <string>
#include <string_view>

namespace WSP::Unicode
{
// The three classes GPT-2's pre-tokenizer pattern is written in — \p{L},
// \p{N}, \s — and everything else.
enum class Category
{
    other,
    letter,
    number,
    whitespace
};

struct CodePoint
{
    char32_t value = 0;
    int byteLength = 1;
    Category category = Category::other;
};

Category categoryOf(char32_t codePoint);

// A byte that starts no valid UTF-8 sequence is reported as one byte of
// Category::other, so a caller slicing the input keeps it verbatim and a
// round trip through the byte-level layer stays lossless.
CodePoint readCodePoint(std::string_view text, std::size_t position);

void appendUtf8(std::string& text, char32_t codePoint);
} // namespace WSP::Unicode
