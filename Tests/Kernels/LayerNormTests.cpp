#include "Common.h"

using namespace nano;
using namespace WSP;
using namespace eacp::GPU;

namespace
{
// The reference, in double so that what it disagrees with the kernel about is
// the kernel and not the CPU's own float rounding. Whisper's normalisation is
// the biased variance — divided by the row length, not by one less than it.
Vector<float> layerNormReference(const Vector<float>& input,
                                 const Vector<float>& weight,
                                 const Vector<float>& bias,
                                 int rowCount,
                                 int rowLength,
                                 float epsilon)
{
    auto expected = sized(rowCount * rowLength);

    for (auto row = 0; row < rowCount; ++row)
    {
        auto base = row * rowLength;
        auto total = 0.0;

        for (auto i = 0; i < rowLength; ++i)
            total += input[base + i];

        auto mean = total / rowLength;
        auto squares = 0.0;

        for (auto i = 0; i < rowLength; ++i)
        {
            auto centred = input[base + i] - mean;
            squares += centred * centred;
        }

        auto scale = 1.0 / std::sqrt(squares / rowLength + epsilon);

        for (auto i = 0; i < rowLength; ++i)
            expected[base + i] =
                (float) ((input[base + i] - mean) * scale * weight[i] + bias[i]);
    }

    return expected;
}

// Every row with statistics of its own, so a kernel that normalised the whole
// buffer at once, or reused one row's mean for the next, would not survive.
Vector<float> rowsWithDifferentStatistics(int rowCount, int rowLength)
{
    auto values = spreadValues(rowCount * rowLength, 20240905u, 4.f);

    for (auto row = 0; row < rowCount; ++row)
        for (auto i = 0; i < rowLength; ++i)
            values[row * rowLength + i] =
                values[row * rowLength + i] * (float) (row + 1) - (float) row * 3.f;

    return values;
}

constexpr auto rowCount = 6;
constexpr auto rowLength = 37;
} // namespace

auto tLayerNormMatchesCpu = test("Kernels/layerNormMatchesCpu") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto input = rowsWithDifferentStatistics(rowCount, rowLength);
    auto weight = spreadValues(rowLength, 11u, 1.5f);
    auto bias = spreadValues(rowLength, 22u, 0.75f);

    auto inputBuffer = storageOf(input);
    auto weightBuffer = storageOf(weight);
    auto biasBuffer = storageOf(bias);
    auto output = outputFor(rowCount * rowLength);

    auto kernel = LayerNorm {};
    kernel.input = inputBuffer;
    kernel.weight = weightBuffer;
    kernel.bias = biasBuffer;
    kernel.output = output;
    kernel.rowLength = (unsigned) rowLength;

    auto result = runOverRows(kernel, output, rowCount, rowCount * rowLength);

    auto expected = layerNormReference(
        input, weight, bias, rowCount, rowLength, LayerNorm::whisperEpsilon);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-4));
};

// A row with no spread at all: the variance is zero, and epsilon is the only
// thing between rsqrt and a division by it. The answer is the bias, exactly.
auto tLayerNormConstantRowYieldsBias =
    test("Kernels/layerNormConstantRowYieldsBias") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto flatRowLength = 32;

    auto input = sized(flatRowLength);

    for (auto i = 0; i < flatRowLength; ++i)
        input[i] = 2.5f;

    auto weight = spreadValues(flatRowLength, 33u, 2.f);
    auto bias = spreadValues(flatRowLength, 44u, 2.f);

    auto inputBuffer = storageOf(input);
    auto weightBuffer = storageOf(weight);
    auto biasBuffer = storageOf(bias);
    auto output = outputFor(flatRowLength);

    auto kernel = LayerNorm {};
    kernel.input = inputBuffer;
    kernel.weight = weightBuffer;
    kernel.bias = biasBuffer;
    kernel.output = output;
    kernel.rowLength = (unsigned) flatRowLength;

    auto result = runOverRows(kernel, output, 1, flatRowLength);

    for (auto i = 0; i < flatRowLength; ++i)
        check(isClose(result[i], bias[i], 1e-5));
};

// The property the encoder rests on: one compiled pipeline, re-pointed at other
// buffers and told a different row length. A kernel that had baked the width in
// gets the second dispatch wrong.
auto tLayerNormOneProgramTwoWidths =
    test("Kernels/layerNormOneProgramTwoWidths") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto narrowLength = 13;
    constexpr auto wideLength = 64;

    auto narrow = rowsWithDifferentStatistics(rowCount, narrowLength);
    auto wide = rowsWithDifferentStatistics(rowCount, wideLength);
    auto narrowWeight = spreadValues(narrowLength, 55u, 1.25f);
    auto narrowBias = spreadValues(narrowLength, 66u, 0.5f);
    auto wideWeight = spreadValues(wideLength, 77u, 1.25f);
    auto wideBias = spreadValues(wideLength, 88u, 0.5f);

    auto narrowInput = storageOf(narrow);
    auto wideInput = storageOf(wide);
    auto narrowWeightBuffer = storageOf(narrowWeight);
    auto narrowBiasBuffer = storageOf(narrowBias);
    auto wideWeightBuffer = storageOf(wideWeight);
    auto wideBiasBuffer = storageOf(wideBias);
    auto narrowOutput = outputFor(rowCount * narrowLength);
    auto wideOutput = outputFor(rowCount * wideLength);

    auto kernel = LayerNorm {};
    kernel.prepare();

    auto commands = device.makeCommandBuffer();

    {
        auto pass = commands.beginCompute();

        kernel.input = narrowInput;
        kernel.weight = narrowWeightBuffer;
        kernel.bias = narrowBiasBuffer;
        kernel.output = narrowOutput;
        kernel.rowLength = (unsigned) narrowLength;
        pass.dispatch(kernel, rowCount);

        kernel.input = wideInput;
        kernel.weight = wideWeightBuffer;
        kernel.bias = wideBiasBuffer;
        kernel.output = wideOutput;
        kernel.rowLength = (unsigned) wideLength;
        pass.dispatch(kernel, rowCount);
    }

    commands.commit();

    auto narrowResult = readBack(narrowOutput, rowCount * narrowLength);
    auto wideResult = readBack(wideOutput, rowCount * wideLength);

    auto narrowExpected = layerNormReference(narrow,
                                             narrowWeight,
                                             narrowBias,
                                             rowCount,
                                             narrowLength,
                                             LayerNorm::whisperEpsilon);

    auto wideExpected = layerNormReference(
        wide, wideWeight, wideBias, rowCount, wideLength, LayerNorm::whisperEpsilon);

    for (auto i = 0; i < narrowResult.size(); ++i)
        check(isClose(narrowResult[i], narrowExpected[i], 1e-4));

    for (auto i = 0; i < wideResult.size(); ++i)
        check(isClose(wideResult[i], wideExpected[i], 1e-4));
};
