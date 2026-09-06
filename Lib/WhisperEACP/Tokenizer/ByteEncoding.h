#pragma once

#include <WhisperEACP/Core/Core.h>

#include <string>
#include <string_view>

// GPT-2's byte-level layer, which Whisper inherits: the 256 input bytes are
// mapped onto printable code points before BPE ever sees them, and mapped back
// on the way out. The mapping is a bijection, so the round trip is exact for
// any byte sequence, valid UTF-8 or not.
namespace WSP::ByteEncoding
{
char32_t codePointForByte(unsigned char byte);

// -1 for a code point outside the table.
int byteForCodePoint(char32_t codePoint);

std::string toPrintable(std::string_view bytes);
std::string fromPrintable(std::string_view printableText);
} // namespace WSP::ByteEncoding
