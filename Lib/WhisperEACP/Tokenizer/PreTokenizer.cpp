#include "PreTokenizer.h"
#include "Unicode.h"

#include <array>

namespace WSP::PreTokenizer
{
namespace
{
using Unicode::Category;

constexpr auto contractions =
    std::array<std::string_view, 7> {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};

std::size_t matchContraction(std::string_view text, std::size_t position)
{
    for (auto contraction: contractions)
    {
        if (text.compare(position, contraction.size(), contraction) == 0)
            return contraction.size();
    }

    return 0;
}

std::size_t skipRun(std::string_view text, std::size_t position, Category category)
{
    while (position < text.size())
    {
        const auto codePoint = Unicode::readCodePoint(text, position);

        if (codePoint.category != category)
            break;

        position += (std::size_t) codePoint.byteLength;
    }

    return position;
}

// ` ?\p{L}+`, ` ?\p{N}+` and ` ?[^\s\p{L}\p{N}]+` differ only in the class they
// then run on. A leading space is a literal U+0020, not any whitespace.
std::size_t matchOptionalSpaceThenRun(std::string_view text,
                                      std::size_t position,
                                      Category category)
{
    const auto afterSpace = text[position] == ' ' ? position + 1 : position;

    if (afterSpace >= text.size())
        return 0;

    if (Unicode::readCodePoint(text, afterSpace).category != category)
        return 0;

    return skipRun(text, afterSpace, category) - position;
}

// `\s+(?!\S)|\s+`: the whole run when it ends the text, and otherwise the run
// minus the one whitespace character the following pre-token is entitled to
// claim as its leading space. One character, not one byte — U+3000 is three.
std::size_t matchWhitespaceRun(std::string_view text, std::size_t position)
{
    auto runEnd = position;
    auto lastCharacterStart = position;

    while (runEnd < text.size())
    {
        const auto codePoint = Unicode::readCodePoint(text, runEnd);

        if (codePoint.category != Category::whitespace)
            break;

        lastCharacterStart = runEnd;
        runEnd += (std::size_t) codePoint.byteLength;
    }

    if (runEnd < text.size() && lastCharacterStart > position)
        return lastCharacterStart - position;

    return runEnd - position;
}

std::size_t lengthOfPreTokenAt(std::string_view text, std::size_t position)
{
    if (const auto contraction = matchContraction(text, position))
        return contraction;

    for (auto category: {Category::letter, Category::number, Category::other})
    {
        if (const auto run = matchOptionalSpaceThenRun(text, position, category))
            return run;
    }

    if (const auto whitespace = matchWhitespaceRun(text, position))
        return whitespace;

    return (std::size_t) Unicode::readCodePoint(text, position).byteLength;
}
} // namespace

Vector<std::string_view> split(std::string_view text)
{
    auto preTokens = Vector<std::string_view> {};

    for (auto position = std::size_t {}; position < text.size();)
    {
        const auto length = lengthOfPreTokenAt(text, position);
        preTokens.add(text.substr(position, length));
        position += length;
    }

    return preTokens;
}
} // namespace WSP::PreTokenizer
