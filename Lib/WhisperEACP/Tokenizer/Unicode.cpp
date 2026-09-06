#include "Unicode.h"

#include <Miro/Unicode.h>

// The one translation unit that knows about Miro's Unicode layer, so the
// header above stays free of its types. Everything here is a rewording of
// Miro's general categories and UTF-8 decoder into the three-class view the
// GPT-2 pre-tokenizer is written in — the table, the decoder and the
// White_Space list that used to live here are Miro's now.
namespace WSP::Unicode
{
Category categoryOf(char32_t codePoint)
{
    if (Miro::Unicode::isWhitespace(codePoint))
        return Category::whitespace;

    if (Miro::Unicode::isLetter(codePoint))
        return Category::letter;

    if (Miro::Unicode::isNumber(codePoint))
        return Category::number;

    return Category::other;
}

CodePoint readCodePoint(std::string_view text, std::size_t position)
{
    const auto decoded = Miro::Unicode::decodeUtf8(text, position);

    // A rejected sequence reports its lead byte, which is a byte and not a
    // code point: classifying its value would make a stray 0xC3 a letter.
    if (!decoded.valid)
        return {decoded.value, 1, Category::other};

    return {decoded.value, decoded.byteLength, categoryOf(decoded.value)};
}

void appendUtf8(std::string& text, char32_t codePoint)
{
    Miro::Unicode::appendUtf8(text, codePoint);
}
} // namespace WSP::Unicode
