#include "Common.h"

#include <cstring>

using namespace nano;
using namespace WSP;
using namespace eacp::GPU;

namespace
{
// The product from its definition, over the same strides the kernel walks, in
// double so that what it disagrees with the kernel about is the kernel.
Vector<float> tiledReference(const Vector<float>& a,
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
Vector<float> runTiled(Program& kernel,
                       const Vector<float>& a,
                       const Buffer& b,
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

    auto commands = Device::shared().makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        kernel.dispatch(pass, shape);
    }

    commands.commit();
    return readBack(output, outputElements);
}

Vector<float> zeroes(int count)
{
    auto values = sized(count);

    for (auto i = 0; i < count; ++i)
        values[i] = 0.f;

    return values;
}

void checkLinear(int rows, int inner, int columns, unsigned seed)
{
    auto shape = TiledMatMulShape::forLinear(rows, inner, columns);
    auto a = spreadValues(rows * inner, seed, 2.f);
    auto b = spreadValues(columns * inner, seed + 1u, 3.f);
    auto bias = spreadValues(columns, seed + 2u, 1.f);

    auto kernel = TiledLinear {};
    auto result = runTiled(kernel, a, storageOf(b), bias, shape, rows * columns);
    auto expected = tiledReference(
        a, b, bias, shape, OperandLayout::ContiguousK, rows * columns);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
}

// Halves as the loader ships them: two to a word, the low half first, rounded
// the way the hardware rounds, and read back widened so the reference sees
// exactly the numbers the kernel does.
Vector<float> packedHalves(const Vector<float>& values, Vector<float>& widened)
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
} // namespace

// Nothing here is a multiple of a tile, a slab or a group: a kernel that ran
// its tile past the edge, or counted a partial slab as a whole one, fails.
auto tTiledLinearOddShape = test("Kernels/tiledLinearOddShape") = []
{
    if (!Device::shared().isValid())
        return;

    checkLinear(5, 7, 3, 100u);
};

// Every extent crosses a tile boundary and none lands on one, and the inner
// count is not a multiple of the slab.
auto tTiledLinearAcrossTiles = test("Kernels/tiledLinearAcrossTiles") = []
{
    if (!Device::shared().isValid())
        return;

    checkLinear(70, 50, 45, 200u);
};

// The model's own projection shape, every extent a multiple of everything.
auto tTiledLinearModelShape = test("Kernels/tiledLinearModelShape") = []
{
    if (!Device::shared().isValid())
        return;

    checkLinear(96, 64, 128, 300u);
};

auto tTiledLinearPackedWeights = test("Kernels/tiledLinearPackedWeights") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto rows = 37;
    constexpr auto inner = 21;
    constexpr auto columns = 35;

    auto shape = TiledMatMulShape::forLinear(rows, inner, columns);
    auto a = spreadValues(rows * inner, 400u, 2.f);
    auto weights = spreadValues(columns * inner, 401u, 3.f);
    auto bias = spreadValues(columns, 402u, 1.f);

    auto widened = Vector<float> {};
    auto packed = packedHalves(weights, widened);

    auto kernel = HalfWeightTiledLinear {};
    auto result =
        runTiled(kernel, a, storageOf(packed), bias, shape, rows * columns);
    auto expected = tiledReference(
        a, widened, bias, shape, OperandLayout::ContiguousK, rows * columns);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
};

// Attention scores as the encoder and decoder compute them: one batch per
// head over the head's slice of the query and key rows, scaled, and here
// causally masked over a cache's trapezoid — three queries against seven
// keys, so query 0 stands at position 4.
auto tTiledAttentionScores = test("Kernels/tiledAttentionScores") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto heads = 3;
    constexpr auto headWidth = 5;
    constexpr auto width = heads * headWidth;
    constexpr auto queries = 35;
    constexpr auto keys = 39;

    auto shape = TiledMatMulShape::forAttentionScores(
        queries, keys, heads, headWidth, width, 0.37f, true);

    auto queryRows = spreadValues(queries * width, 500u, 2.f);
    auto keyRows = spreadValues(keys * width, 501u, 2.f);
    auto bias = zeroes(keys);

    const auto outputs = heads * queries * keys;

    auto kernel = TiledLinear {};
    auto result =
        runTiled(kernel, queryRows, storageOf(keyRows), bias, shape, outputs);
    auto expected = tiledReference(
        queryRows, keyRows, bias, shape, OperandLayout::ContiguousK, outputs);

    auto maskedCount = 0;

    for (auto i = 0; i < result.size(); ++i)
    {
        check(isClose(result[i], expected[i], 1e-5));

        if (expected[i] == causalMaskScore)
        {
            check(result[i] == causalMaskScore);
            ++maskedCount;
        }
    }

    check(maskedCount == heads * (queries * (queries - 1) / 2));
};

// The other half: probabilities against the head's columns of the value rows,
// written into the head's columns of the output — the operand that is
// contiguous along n.
auto tTiledAttentionApply = test("Kernels/tiledAttentionApply") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto heads = 2;
    constexpr auto headWidth = 9;
    constexpr auto width = heads * headWidth;
    constexpr auto queries = 41;
    constexpr auto keys = 37;

    auto shape =
        TiledMatMulShape::forAttentionApply(queries, keys, heads, headWidth, width);

    auto probabilities = spreadValues(heads * queries * keys, 600u, 1.f);
    auto valueRows = spreadValues(keys * width, 601u, 2.f);
    auto bias = zeroes(headWidth);

    const auto outputs = queries * width;

    auto kernel = TiledMatMul {};
    auto result =
        runTiled(kernel, probabilities, storageOf(valueRows), bias, shape, outputs);
    auto expected = tiledReference(
        probabilities, valueRows, bias, shape, OperandLayout::ContiguousN, outputs);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
};

// One pipeline, two shapes, two dispatches — the encoder's six projections
// of two widths through one program.
auto tTiledLinearOneProgramTwoShapes =
    test("Kernels/tiledLinearOneProgramTwoShapes") = []
{
    if (!Device::shared().isValid())
        return;

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

    auto kernel = TiledLinear {};
    kernel.prepare();

    auto commands = Device::shared().makeCommandBuffer();

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

    auto firstResult = readBack(firstOutput, 33 * 17);
    auto secondResult = readBack(secondOutput, 9 * 65);
    auto firstExpected = tiledReference(
        firstA, firstB, firstBias, first, OperandLayout::ContiguousK, 33 * 17);
    auto secondExpected = tiledReference(
        secondA, secondB, secondBias, second, OperandLayout::ContiguousK, 9 * 65);

    for (auto i = 0; i < firstResult.size(); ++i)
        check(isClose(firstResult[i], firstExpected[i], 1e-5));

    for (auto i = 0; i < secondResult.size(); ++i)
        check(isClose(secondResult[i], secondExpected[i], 1e-5));
};

// The two stages folded into the store: the GELU after fc1, and the residual
// sum after every out_proj and fc2, added into what the output held.
auto tTiledLinearFoldsGeluAndResidual =
    test("Kernels/tiledLinearFoldsGeluAndResidual") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto rows = 40;
    constexpr auto inner = 24;
    constexpr auto columns = 36;

    auto a = spreadValues(rows * inner, 800u, 2.f);
    auto b = spreadValues(columns * inner, 801u, 1.f);
    auto bias = spreadValues(columns, 802u, 1.f);
    auto stream = spreadValues(rows * columns, 803u, 5.f);

    auto activated = TiledMatMulShape::forLinear(rows, inner, columns);
    activated.gelu = true;

    auto kernel = TiledLinear {};
    auto geluResult =
        runTiled(kernel, a, storageOf(b), bias, activated, rows * columns);
    auto geluExpected = tiledReference(
        a, b, bias, activated, OperandLayout::ContiguousK, rows * columns);

    for (auto i = 0; i < geluResult.size(); ++i)
        check(isClose(geluResult[i], geluExpected[i], 1e-5));

    auto accumulated = TiledMatMulShape::forLinear(rows, inner, columns);
    accumulated.residual = true;

    auto residualResult =
        runTiled(kernel, a, storageOf(b), bias, accumulated, rows * columns, stream);
    auto residualExpected = tiledReference(
        a, b, bias, accumulated, OperandLayout::ContiguousK, rows * columns, stream);

    for (auto i = 0; i < residualResult.size(); ++i)
        check(isClose(residualResult[i], residualExpected[i], 1e-5));
};
