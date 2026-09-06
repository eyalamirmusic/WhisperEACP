#include "Common.h"

#include <algorithm>

using namespace nano;
using namespace WSP;
using namespace eacp::GPU;

namespace
{
Vector<float>
    softmaxReference(const Vector<float>& input, int rowCount, int rowLength)
{
    auto expected = sized(rowCount * rowLength);

    for (auto row = 0; row < rowCount; ++row)
    {
        auto base = row * rowLength;
        auto largest = (double) input[base];

        for (auto i = 1; i < rowLength; ++i)
            largest = std::max(largest, (double) input[base + i]);

        auto total = 0.0;

        for (auto i = 0; i < rowLength; ++i)
            total += std::exp(input[base + i] - largest);

        for (auto i = 0; i < rowLength; ++i)
            expected[base + i] =
                (float) (std::exp(input[base + i] - largest) / total);
    }

    return expected;
}

constexpr auto rowCount = 5;
constexpr auto rowLength = 29;
} // namespace

auto tSoftmaxMatchesCpu = test("Kernels/softmaxMatchesCpu") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto input = spreadValues(rowCount * rowLength, 90210u, 6.f);
    auto inputBuffer = storageOf(input);
    auto output = outputFor(rowCount * rowLength);

    auto kernel = Softmax {};
    kernel.input = inputBuffer;
    kernel.output = output;
    kernel.rowLength = (unsigned) rowLength;

    auto result = runOverRows(kernel, output, rowCount, rowCount * rowLength);
    auto expected = softmaxReference(input, rowCount, rowLength);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));

    for (auto row = 0; row < rowCount; ++row)
    {
        auto total = 0.0;

        for (auto i = 0; i < rowLength; ++i)
            total += result[row * rowLength + i];

        check(isClose((float) total, 1.0, 1e-5));
    }
};

// The row a naive softmax loses: exp overflows float32 somewhere past 88, so
// every one of these logits exponentiates to infinity and the ratio comes back
// as a NaN. Subtracting the row maximum first leaves the same distribution and
// the largest term at exactly one.
auto tSoftmaxSurvivesLargeLogits = test("Kernels/softmaxSurvivesLargeLogits") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto wideRowLength = 16;

    auto input = sized(2 * wideRowLength);

    for (auto i = 0; i < wideRowLength; ++i)
    {
        input[i] = 1000.f + (float) i;
        input[wideRowLength + i] = -1000.f - (float) i;
    }

    auto inputBuffer = storageOf(input);
    auto output = outputFor(2 * wideRowLength);

    auto kernel = Softmax {};
    kernel.input = inputBuffer;
    kernel.output = output;
    kernel.rowLength = (unsigned) wideRowLength;

    auto result = runOverRows(kernel, output, 2, 2 * wideRowLength);
    auto expected = softmaxReference(input, 2, wideRowLength);

    for (auto i = 0; i < result.size(); ++i)
    {
        check(std::isfinite(result[i]));
        check(isClose(result[i], expected[i], 1e-5));
    }
};

// The exponentials live in the output buffer between the summing pass and the
// scaling one, so a read that came back with whatever the buffer held before
// the store would pass unnoticed against a fresh allocation of zeroes. Binding
// an output already full of a large constant is what makes that loud: scaling
// the constant instead of the exponential leaves a row summing to rowLength
// times it, and every element far from what the reference says.
auto tSoftmaxIgnoresPoisonedOutput =
    test("Kernels/softmaxIgnoresPoisonedOutput") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto poison = 1e9f;

    auto input = spreadValues(rowCount * rowLength, 424242u, 4.f);
    auto inputBuffer = storageOf(input);

    auto poisoned = Vector<float> {};
    poisoned.assign(rowCount * rowLength, poison);

    auto output = storageOf(poisoned);

    auto kernel = Softmax {};
    kernel.input = inputBuffer;
    kernel.output = output;
    kernel.rowLength = (unsigned) rowLength;

    auto result = runOverRows(kernel, output, rowCount, rowCount * rowLength);
    auto expected = softmaxReference(input, rowCount, rowLength);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));

    for (auto row = 0; row < rowCount; ++row)
    {
        auto total = 0.0;

        for (auto i = 0; i < rowLength; ++i)
            total += result[row * rowLength + i];

        check(isClose((float) total, 1.0, 1e-5));
    }
};

// A row of equal logits is the one case the answer can be written down without
// computing it, and the one a missing normalisation still gets wrong.
auto tSoftmaxUniformRowIsUniform = test("Kernels/softmaxUniformRowIsUniform") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto flatRowLength = 40;

    auto input = sized(flatRowLength);

    for (auto i = 0; i < flatRowLength; ++i)
        input[i] = -7.25f;

    auto inputBuffer = storageOf(input);
    auto output = outputFor(flatRowLength);

    auto kernel = Softmax {};
    kernel.input = inputBuffer;
    kernel.output = output;
    kernel.rowLength = (unsigned) flatRowLength;

    auto result = runOverRows(kernel, output, 1, flatRowLength);

    for (auto i = 0; i < flatRowLength; ++i)
        check(isClose(result[i], 1.0 / flatRowLength, 1e-6));
};
