#include "ByteEncoding.h"
#include "Unicode.h"

#include <array>

namespace WSP::ByteEncoding
{
namespace
{
constexpr auto byteCount = std::size_t {256};

// The bytes that already have a printable code point of their own. Everything
// else is pushed above U+00FF, in byte order, starting at U+0100.
constexpr bool hasPrintableCodePoint(int byte)
{
    return (byte >= '!' && byte <= '~') || (byte >= 0xA1 && byte <= 0xAC)
           || (byte >= 0xAE && byte <= 0xFF);
}

constexpr auto highestCodePoint = char32_t {0x143};

constexpr std::array<char32_t, byteCount> makeByteToCodePoint()
{
    auto table = std::array<char32_t, byteCount> {};
    auto nextUnused = char32_t {0x100};

    for (auto byte = 0; byte < (int) byteCount; ++byte)
    {
        table[(std::size_t) byte] =
            hasPrintableCodePoint(byte) ? static_cast<char32_t>(byte) : nextUnused++;
    }

    return table;
}

constexpr std::array<int, highestCodePoint + 1> makeCodePointToByte()
{
    auto table = std::array<int, highestCodePoint + 1> {};
    table.fill(-1);

    const auto forward = makeByteToCodePoint();

    for (auto byte = 0; byte < (int) byteCount; ++byte)
        table[forward[(std::size_t) byte]] = byte;

    return table;
}

constexpr auto byteToCodePoint = makeByteToCodePoint();
constexpr auto codePointToByte = makeCodePointToByte();
} // namespace

char32_t codePointForByte(unsigned char byte)
{
    return byteToCodePoint[byte];
}

int byteForCodePoint(char32_t codePoint)
{
    if (codePoint > highestCodePoint)
        return -1;

    return codePointToByte[codePoint];
}

std::string toPrintable(std::string_view bytes)
{
    auto printable = std::string {};
    printable.reserve(bytes.size() * 2);

    for (auto byte: bytes)
        Unicode::appendUtf8(printable, codePointForByte((unsigned char) byte));

    return printable;
}

std::string fromPrintable(std::string_view printableText)
{
    auto bytes = std::string {};
    bytes.reserve(printableText.size());

    for (auto position = std::size_t {}; position < printableText.size();)
    {
        const auto codePoint = Unicode::readCodePoint(printableText, position);
        position += (std::size_t) codePoint.byteLength;

        const auto byte = byteForCodePoint(codePoint.value);

        if (byte >= 0)
            bytes += static_cast<char>(byte);
    }

    return bytes;
}
} // namespace WSP::ByteEncoding
