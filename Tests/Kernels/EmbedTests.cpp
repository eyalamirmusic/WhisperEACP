#include "Common.h"

using namespace nano;
using namespace WSP;
using namespace eacp::GPU;

namespace
{
// output[t, c] = tokenTable[token[t], c] + positionTable[firstPosition + t, c],
// written from the definition rather than from the kernel's index arithmetic.
Vector<float> embedReference(const Vector<std::uint32_t>& tokens,
                             const Vector<float>& tokenTable,
                             const Vector<float>& positionTable,
                             int width,
                             int firstPosition)
{
    auto expected = sized(tokens.size() * width);

    for (auto step = 0; step < tokens.size(); ++step)
        for (auto channel = 0; channel < width; ++channel)
            expected[step * width + channel] =
                tokenTable[(int) tokens[step] * width + channel]
                + positionTable[(firstPosition + step) * width + channel];

    return expected;
}

Vector<float> runEmbed(const Vector<std::uint32_t>& tokens,
                       const Vector<float>& tokenTable,
                       const Vector<float>& positionTable,
                       int width,
                       int firstPosition)
{
    auto tokenBuffer = storageOf(tokens);
    auto tokenTableBuffer = storageOf(tokenTable);
    auto positionTableBuffer = storageOf(positionTable);
    auto output = outputFor(tokens.size() * width);

    auto kernel = Embed {};
    kernel.tokens = tokenBuffer;
    kernel.tokenTable = tokenTableBuffer;
    kernel.positionTable = positionTableBuffer;
    kernel.output = output;
    kernel.width = (unsigned) width;
    kernel.firstPosition = (unsigned) firstPosition;

    return runOverGrid(kernel, output, width, tokens.size());
}

Vector<std::uint32_t> tokensOf(std::initializer_list<std::uint32_t> ids)
{
    auto values = unsignedSized((int) ids.size());
    auto at = 0;

    for (auto id: ids)
        values[at++] = id;

    return values;
}

// A table whose every element says which row and which column it is, so a
// gather that read the right number out of the wrong row is a wrong number
// rather than a wrong tolerance. Every value here is an integer plus a quarter
// of one, so float32 holds it exactly up to a vocabulary of eight million.
Vector<float> countedTable(int rowCount, int width, float rowScale)
{
    auto values = sized(rowCount * width);

    for (auto row = 0; row < rowCount; ++row)
        for (auto column = 0; column < width; ++column)
            values[row * width + column] =
                rowScale * (float) row + 0.25f * (float) column;

    return values;
}

// tiny.en's own vocabulary, which is the number that decides whether a token id
// survives the trip: every id below 2^23 has a float bit pattern with an
// exponent field of zero, so the whole vocabulary arrives as what a float
// would call a subnormal.
constexpr auto vocabulary = 51864;
constexpr auto maxPositions = 448;
} // namespace

auto tEmbedMatchesCpu = test("Kernels/embedMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto width = 5;
    constexpr auto smallVocabulary = 13;

    auto tokens = tokensOf({7, 0, 12, 3, 3, 11});
    auto tokenTable = spreadValues(smallVocabulary * width, 4242u, 2.f);
    auto positionTable = spreadValues(maxPositions * width, 2424u, 1.f);

    auto result = runEmbed(tokens, tokenTable, positionTable, width, 0);
    auto expected = embedReference(tokens, tokenTable, positionTable, width, 0);

    check(result.size() == expected.size());

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-6));
};

// The shape a decoder step has once a cache is carrying the sequence so far:
// one token, and a position that is the cache's length rather than zero.
auto tEmbedAtCacheOffset = test("Kernels/embedAtCacheOffset") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto width = 7;
    constexpr auto smallVocabulary = 19;
    constexpr auto firstPosition = 23;

    auto tokens = tokensOf({17});
    auto tokenTable = spreadValues(smallVocabulary * width, 5150u, 3.f);
    auto positionTable = spreadValues(maxPositions * width, 5151u, 2.f);

    auto result = runEmbed(tokens, tokenTable, positionTable, width, firstPosition);
    auto expected =
        embedReference(tokens, tokenTable, positionTable, width, firstPosition);

    check(result.size() == width);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-6));
};

// The two tables are indexed by different things, and a kernel that fed the
// token id to the positional table (or the step to the token table) still
// agrees with itself. Zeroing one table at a time is what separates them: with
// the positional table zero the answer is the token's row, and with the token
// table zero it is firstPosition + t's row, both exactly.
auto tEmbedGathersTokenRowsAndPositionRows =
    test("Kernels/embedGathersTokenRowsAndPositionRows") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto width = 5;
    constexpr auto smallVocabulary = 13;
    constexpr auto firstPosition = 9;

    auto tokens = tokensOf({12, 0, 5, 7});
    auto tokenTable = countedTable(smallVocabulary, width, 8.f);
    auto positionTable = countedTable(maxPositions, width, 64.f);

    auto zeroPositions = sized(maxPositions * width);
    auto zeroTokens = sized(smallVocabulary * width);

    for (auto i = 0; i < zeroPositions.size(); ++i)
        zeroPositions[i] = 0.f;

    for (auto i = 0; i < zeroTokens.size(); ++i)
        zeroTokens[i] = 0.f;

    auto byToken = runEmbed(tokens, tokenTable, zeroPositions, width, firstPosition);
    auto byPosition =
        runEmbed(tokens, zeroTokens, positionTable, width, firstPosition);

    for (auto step = 0; step < tokens.size(); ++step)
    {
        for (auto column = 0; column < width; ++column)
        {
            const auto expectedToken =
                8.f * (float) tokens[step] + 0.25f * (float) column;

            const auto expectedPosition =
                64.f * (float) (firstPosition + step) + 0.25f * (float) column;

            check(byToken[step * width + column] == expectedToken);
            check(byPosition[step * width + column] == expectedPosition);
        }
    }
};

// The id itself, over tiny.en's whole vocabulary rather than a toy one. The
// tables are built so the answer is the token id as a float and nothing else,
// so an id that lost a bit on the way through the buffer — the bitcast reading
// a subnormal, a backend flushing one to zero — comes back as row zero's value
// and fails by 51863 rather than by a tolerance.
auto tEmbedReadsWholeVocabularyExactly =
    test("Kernels/embedReadsWholeVocabularyExactly") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto width = 3;

    auto tokens = tokensOf({0, 1, 220, 50256, 50257, vocabulary - 1});
    auto tokenTable = countedTable(vocabulary, width, 1.f);
    auto positionTable = sized(maxPositions * width);

    for (auto position = 0; position < maxPositions; ++position)
        for (auto column = 0; column < width; ++column)
            positionTable[position * width + column] = -0.25f * (float) column;

    auto result = runEmbed(tokens, tokenTable, positionTable, width, 0);

    for (auto step = 0; step < tokens.size(); ++step)
        for (auto column = 0; column < width; ++column)
            check(result[step * width + column] == (float) tokens[step]);
};
