#include "Common.h"

using namespace nano;
using namespace WSP;
using namespace eacp::GPU;

namespace
{
// Not a multiple of the 64-wide 1D dispatch group, so a kernel that leaned on
// the grid landing exactly on the buffer fails here.
constexpr auto elementCount = 101;
} // namespace

auto tAddMatchesCpu = test("Kernels/addMatchesCpu") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto a = spreadValues(elementCount, 8080u, 4.f);
    auto b = spreadValues(elementCount, 9090u, 4.f);

    auto aBuffer = storageOf(a);
    auto bBuffer = storageOf(b);
    auto output = outputFor(elementCount);

    auto kernel = Add {};
    kernel.a = aBuffer;
    kernel.b = bBuffer;
    kernel.output = output;

    auto result = runOverRows(kernel, output, elementCount, elementCount);

    for (auto i = 0; i < elementCount; ++i)
        check(isClose(result[i], (double) a[i] + b[i], 1e-6));
};

// The residual connection is what this kernel exists for, and adding a zero
// vector is the one case whose answer is the input exactly — a sum that came
// back merely close would mean the residual path is not the identity it has to
// be when a block contributes nothing.
auto tAddZeroLeavesTheInput = test("Kernels/addZeroLeavesTheInput") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto a = spreadValues(elementCount, 1212u, 3.f);
    auto zeros = sized(elementCount);

    for (auto i = 0; i < elementCount; ++i)
        zeros[i] = 0.f;

    auto aBuffer = storageOf(a);
    auto zeroBuffer = storageOf(zeros);
    auto output = outputFor(elementCount);

    auto kernel = Add {};
    kernel.a = aBuffer;
    kernel.b = zeroBuffer;
    kernel.output = output;

    auto result = runOverRows(kernel, output, elementCount, elementCount);

    for (auto i = 0; i < elementCount; ++i)
        check(result[i] == a[i]);
};

// One program, two lengths, two dispatches — the shape a model that adds a
// residual around blocks of differing widths needs, and the reason the length
// comes from the dispatch rather than from a uniform.
auto tAddOneProgramTwoLengths = test("Kernels/addOneProgramTwoLengths") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto shortLength = 7;

    auto longA = spreadValues(elementCount, 3131u, 2.f);
    auto longB = spreadValues(elementCount, 4141u, 2.f);
    auto shortA = spreadValues(shortLength, 5151u, 2.f);
    auto shortB = spreadValues(shortLength, 6161u, 2.f);

    auto longABuffer = storageOf(longA);
    auto longBBuffer = storageOf(longB);
    auto longOutput = outputFor(elementCount);

    auto shortABuffer = storageOf(shortA);
    auto shortBBuffer = storageOf(shortB);
    auto shortOutput = outputFor(shortLength);

    auto kernel = Add {};
    kernel.prepare();

    auto commands = device.makeCommandBuffer();

    {
        auto pass = commands.beginCompute();

        kernel.a = longABuffer;
        kernel.b = longBBuffer;
        kernel.output = longOutput;
        pass.dispatch(kernel, elementCount);

        kernel.a = shortABuffer;
        kernel.b = shortBBuffer;
        kernel.output = shortOutput;
        pass.dispatch(kernel, shortLength);
    }

    commands.commit();

    auto longResult = readBack(longOutput, elementCount);
    auto shortResult = readBack(shortOutput, shortLength);

    for (auto i = 0; i < elementCount; ++i)
        check(isClose(longResult[i], (double) longA[i] + longB[i], 1e-6));

    for (auto i = 0; i < shortLength; ++i)
        check(isClose(shortResult[i], (double) shortA[i] + shortB[i], 1e-6));
};
