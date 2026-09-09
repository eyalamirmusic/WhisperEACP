#pragma once

#include "Common.h"

#include <cstring>

namespace WSP
{
// What a tiled product has to answer with, and the checks that ask it. There
// are two programs computing the same TiledMatMulShape — the register-tiled
// one every backend has and the SIMD-group one Metal specialises to — so the
// reference and every assertion against it are written once here and each
// program is run through them.
namespace TiledProduct
{
// The product from its definition, over the same strides the kernel walks, in
// double so that what it disagrees with the kernel about is the kernel.
inline Vector<float> reference(const Vector<float>& a,
                               const Vector<float>& b,
                               const Vector<float>& bias,
                               const TiledMatMulShape& shape,
                               OperandLayout layout,
                               int outputElements,
                               const Vector<float>& previous = {})
{
    auto expected = sized(outputElements);

    for (auto i = 0; i < outputElements; ++i)
        expected[i] = 0.f;

    for (auto batch = 0; batch < shape.batches; ++batch)
        for (auto m = 0; m < shape.rows; ++m)
            for (auto n = 0; n < shape.columns; ++n)
            {
                auto total = 0.0;

                for (auto k = 0; k < shape.inner; ++k)
                {
                    auto left =
                        a[batch * shape.aBatchStride + m * shape.aRowStride + k];
                    auto right =
                        layout == OperandLayout::ContiguousK
                            ? b[batch * shape.bBatchStride + n * shape.bStride + k]
                            : b[batch * shape.bBatchStride + k * shape.bStride + n];

                    total += (double) left * right;
                }

                auto masked = shape.causal && n + shape.rows > m + shape.columns;
                auto at = batch * shape.cBatchStride + m * shape.cRowStride + n;
                auto value = shape.scale * total + bias[n];

                if (shape.gelu)
                    value = 0.5 * value * (1.0 + std::erf(value / std::sqrt(2.0)));

                if (shape.residual)
                    value += previous[at];

                expected[at] = masked ? causalMaskScore : (float) value;
            }

    return expected;
}

template <typename Program>
Vector<float> run(Program& kernel,
                  const Vector<float>& a,
                  const eacp::GPU::Buffer& b,
                  const Vector<float>& bias,
                  const TiledMatMulShape& shape,
                  int outputElements,
                  const Vector<float>& previous = {})
{
    auto aBuffer = storageOf(a);
    auto biasBuffer = storageOf(bias);

    // Poisoned rather than fresh, so an element the kernel should not touch
    // is seen to be untouched and a tile that ran past an edge shows — or
    // holding what a residual product is meant to add to.
    auto poisoned = sized(outputElements);

    for (auto i = 0; i < outputElements; ++i)
        poisoned[i] = previous.size() > 0 ? previous[i] : -777.f;

    auto output = storageOf(poisoned);

    kernel.a = aBuffer;
    kernel.b = b;
    kernel.bias = biasBuffer;
    kernel.output = output;
    kernel.prepare();

    auto commands = eacp::GPU::Device::shared().makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        kernel.dispatch(pass, shape);
    }

    commands.commit();
    return readBack(output, outputElements);
}

inline Vector<float> zeroes(int count)
{
    auto values = sized(count);

    for (auto i = 0; i < count; ++i)
        values[i] = 0.f;

    return values;
}

// Halves as the loader ships them: two to a word, the low half first, rounded
// the way the hardware rounds, and read back widened so the reference sees
// exactly the numbers the kernel does.
inline Vector<float> packedHalves(const Vector<float>& values,
                                  Vector<float>& widened)
{
    auto words = sized((values.size() + 1) / 2);
    widened = sized(values.size());

    for (auto i = 0; i < words.size(); ++i)
        words[i] = 0.f;

    for (auto i = 0; i < values.size(); ++i)
    {
        auto half = (_Float16) values[i];
        widened[i] = (float) half;

        auto bits = std::uint16_t {};
        std::memcpy(&bits, &half, sizeof(bits));

        auto word = std::uint32_t {};
        std::memcpy(&word, &words[i / 2], sizeof(word));
        word |= (std::uint32_t) bits << (16 * (i % 2));
        std::memcpy(&words[i / 2], &word, sizeof(word));
    }

    return words;
}

// What every check below asks of a product whose inner extent is a few dozen
// terms. Long sums want more — see dotProductTolerance.
inline constexpr auto shortSumTolerance = 1e-5;

// A float dot product's error grows with how many terms it summed, and it
// grows with the magnitude of the partial sums rather than of the answer: 1536
// terms of size 2 reach ±80 on the way to an answer that cancels to 0.1. So a
// tolerance relative to the answer alone is the wrong measure once the inner
// extent is long, and this adds the accumulation's own — one float epsilon per
// term, at the spread the checks here feed.
inline double dotProductTolerance(int inner)
{
    return shortSumTolerance + 2.4e-7 * (double) inner;
}

inline void checkMatches(const Vector<float>& result,
                         const Vector<float>& expected,
                         double tolerance = shortSumTolerance)
{
    for (auto i = 0; i < result.size(); ++i)
        nano::check(isClose(result[i], expected[i], tolerance));
}

template <typename Program>
void checkLinear(int rows,
                 int inner,
                 int columns,
                 unsigned seed,
                 double tolerance = shortSumTolerance)
{
    auto shape = TiledMatMulShape::forLinear(rows, inner, columns);
    auto a = spreadValues(rows * inner, seed, 2.f);
    auto b = spreadValues(columns * inner, seed + 1u, 3.f);
    auto bias = spreadValues(columns, seed + 2u, 1.f);

    auto kernel = Program {};
    auto result = run(kernel, a, storageOf(b), bias, shape, rows * columns);

    checkMatches(
        result,
        reference(a, b, bias, shape, OperandLayout::ContiguousK, rows * columns),
        tolerance);
}

template <typename Program>
void checkPackedLinear(int rows, int inner, int columns, unsigned seed)
{
    auto shape = TiledMatMulShape::forLinear(rows, inner, columns);
    auto a = spreadValues(rows * inner, seed, 2.f);
    auto weights = spreadValues(columns * inner, seed + 1u, 3.f);
    auto bias = spreadValues(columns, seed + 2u, 1.f);

    auto widened = Vector<float> {};
    auto packed = packedHalves(weights, widened);

    auto kernel = Program {};
    auto result = run(kernel, a, storageOf(packed), bias, shape, rows * columns);

    checkMatches(
        result,
        reference(
            a, widened, bias, shape, OperandLayout::ContiguousK, rows * columns));
}

// Attention scores as the encoder and decoder compute them: one batch per
// head over the head's slice of the query and key rows, scaled, and causally
// masked over a cache's trapezoid, where query m stands at keys - queries + m.
template <typename Program>
void checkAttentionScores(
    int queries, int keys, int heads, int headWidth, unsigned seed)
{
    const auto width = heads * headWidth;
    const auto outputs = heads * queries * keys;

    auto shape = TiledMatMulShape::forAttentionScores(
        queries, keys, heads, headWidth, width, 0.37f, true);

    auto queryRows = spreadValues(queries * width, seed, 2.f);
    auto keyRows = spreadValues(keys * width, seed + 1u, 2.f);
    auto bias = zeroes(keys);

    auto kernel = Program {};
    auto result = run(kernel, queryRows, storageOf(keyRows), bias, shape, outputs);
    auto expected = reference(
        queryRows, keyRows, bias, shape, OperandLayout::ContiguousK, outputs);

    auto maskedCount = 0;

    for (auto i = 0; i < result.size(); ++i)
    {
        nano::check(isClose(result[i], expected[i], 1e-5));

        if (expected[i] == causalMaskScore)
        {
            nano::check(result[i] == causalMaskScore);
            ++maskedCount;
        }
    }

    nano::check(maskedCount == heads * (queries * (queries - 1) / 2));
}

// The other half: probabilities against the head's columns of the value rows,
// written into the head's columns of the output — the operand that is
// contiguous along n.
template <typename Program>
void checkAttentionApply(
    int queries, int keys, int heads, int headWidth, unsigned seed)
{
    const auto width = heads * headWidth;
    const auto outputs = queries * width;

    auto shape =
        TiledMatMulShape::forAttentionApply(queries, keys, heads, headWidth, width);

    auto probabilities = spreadValues(heads * queries * keys, seed, 1.f);
    auto valueRows = spreadValues(keys * width, seed + 1u, 2.f);
    auto bias = zeroes(headWidth);

    auto kernel = Program {};
    auto result =
        run(kernel, probabilities, storageOf(valueRows), bias, shape, outputs);

    checkMatches(result,
                 reference(probabilities,
                           valueRows,
                           bias,
                           shape,
                           OperandLayout::ContiguousN,
                           outputs));
}

// The two stages folded into the store: the GELU after fc1, and the residual
// sum after every out_proj and fc2, added into what the output held.
template <typename Program>
void checkGeluAndResidual(int rows, int inner, int columns, unsigned seed)
{
    auto a = spreadValues(rows * inner, seed, 2.f);
    auto b = spreadValues(columns * inner, seed + 1u, 1.f);
    auto bias = spreadValues(columns, seed + 2u, 1.f);
    auto stream = spreadValues(rows * columns, seed + 3u, 5.f);

    auto activated = TiledMatMulShape::forLinear(rows, inner, columns);
    activated.gelu = true;

    auto kernel = Program {};
    auto geluResult = run(kernel, a, storageOf(b), bias, activated, rows * columns);

    checkMatches(
        geluResult,
        reference(
            a, b, bias, activated, OperandLayout::ContiguousK, rows * columns));

    auto accumulated = TiledMatMulShape::forLinear(rows, inner, columns);
    accumulated.residual = true;

    auto residualResult =
        run(kernel, a, storageOf(b), bias, accumulated, rows * columns, stream);

    checkMatches(residualResult,
                 reference(a,
                           b,
                           bias,
                           accumulated,
                           OperandLayout::ContiguousK,
                           rows * columns,
                           stream));
}

// One pipeline, two shapes, two dispatches — the encoder's six projections of
// two widths through one program, and what says a shape is a uniform rather
// than something the pipeline was compiled around.
template <typename Program>
void checkTwoShapesThroughOneProgram()
{
    auto first = TiledMatMulShape::forLinear(33, 20, 17);
    auto second = TiledMatMulShape::forLinear(9, 40, 65);

    auto firstA = spreadValues(33 * 20, 700u, 2.f);
    auto firstB = spreadValues(17 * 20, 701u, 2.f);
    auto firstBias = spreadValues(17, 702u, 1.f);
    auto secondA = spreadValues(9 * 40, 703u, 2.f);
    auto secondB = spreadValues(65 * 40, 704u, 2.f);
    auto secondBias = spreadValues(65, 705u, 1.f);

    auto firstABuffer = storageOf(firstA);
    auto firstBBuffer = storageOf(firstB);
    auto firstBiasBuffer = storageOf(firstBias);
    auto firstOutput = outputFor(33 * 17);
    auto secondABuffer = storageOf(secondA);
    auto secondBBuffer = storageOf(secondB);
    auto secondBiasBuffer = storageOf(secondBias);
    auto secondOutput = outputFor(9 * 65);

    auto kernel = Program {};
    kernel.prepare();

    auto commands = eacp::GPU::Device::shared().makeCommandBuffer();

    {
        auto pass = commands.beginCompute();

        kernel.a = firstABuffer;
        kernel.b = firstBBuffer;
        kernel.bias = firstBiasBuffer;
        kernel.output = firstOutput;
        kernel.dispatch(pass, first);

        kernel.a = secondABuffer;
        kernel.b = secondBBuffer;
        kernel.bias = secondBiasBuffer;
        kernel.output = secondOutput;
        kernel.dispatch(pass, second);
    }

    commands.commit();

    checkMatches(
        readBack(firstOutput, 33 * 17),
        reference(
            firstA, firstB, firstBias, first, OperandLayout::ContiguousK, 33 * 17));

    checkMatches(readBack(secondOutput, 9 * 65),
                 reference(secondA,
                           secondB,
                           secondBias,
                           second,
                           OperandLayout::ContiguousK,
                           9 * 65));
}
} // namespace TiledProduct
} // namespace WSP
