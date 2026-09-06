#pragma once

#include <WhisperEACP/Core/Core.h>

#include <string_view>

// The split BPE is applied to, one pre-token at a time. It is GPT-2's pattern,
// which HuggingFace's ByteLevel pre-tokenizer still carries verbatim:
//
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// Written as a scanner rather than a regex because no standard C++ regex knows
// \p{L}. The returned views point into the caller's text, so a byte that is
// not valid UTF-8 survives the split untouched.
namespace WSP::PreTokenizer
{
Vector<std::string_view> split(std::string_view text);
} // namespace WSP::PreTokenizer
