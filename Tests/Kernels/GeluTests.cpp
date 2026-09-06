#include "Common.h"

#include <algorithm>

using namespace nano;
using namespace WSP;
using namespace eacp::GPU;

namespace
{
// The reference is the definition itself, in double precision through
// std::erf. The kernel goes through eacp's erf, which is a float32
// approximation of the same function, so comparing against this is how its
// error gets a number rather than an assurance.
double exactGeluReference(double x)
{
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// The GELU whisper.cpp settles for, kept here only so the kernel's accuracy
// has something to be better than.
double tanhGeluReference(double x)
{
    constexpr auto rootTwoOverPi = 0.7978845608028654;

    return 0.5 * x * (1.0 + std::tanh(rootTwoOverPi * (x + 0.044715 * x * x * x)));
}

// A sweep dense enough to land on the region the two GELUs disagree most
// about, which is |x| between one and three rather than at either tail.
Vector<float> sweptValues(int count, double from, double to)
{
    auto values = sized(count);

    for (auto i = 0; i < count; ++i)
        values[i] = (float) (from + (to - from) * i / (count - 1));

    return values;
}

constexpr auto sweepCount = 513;
} // namespace

auto tGeluMatchesExactCpu = test("Kernels/geluMatchesExactCpu") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto input = sweptValues(sweepCount, -8.0, 8.0);
    auto inputBuffer = storageOf(input);
    auto output = outputFor(sweepCount);

    auto kernel = Gelu {};
    kernel.input = inputBuffer;
    kernel.output = output;

    auto result = runOverRows(kernel, output, sweepCount, sweepCount);

    for (auto i = 0; i < sweepCount; ++i)
        check(isClose(result[i], exactGeluReference(input[i]), 1e-6));
};

// The claim the exact GELU is chosen on: over the range a transformer's
// activations actually occupy it is nearer the true function than the tanh
// form the reference implementations settle for, by about three orders of
// magnitude — eacp's erf being an approximation does not cost that margin.
auto tGeluBeatsTanhApproximation = test("Kernels/geluBeatsTanhApproximation") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto input = sweptValues(sweepCount, -4.0, 4.0);
    auto inputBuffer = storageOf(input);
    auto output = outputFor(sweepCount);

    auto kernel = Gelu {};
    kernel.input = inputBuffer;
    kernel.output = output;

    auto result = runOverRows(kernel, output, sweepCount, sweepCount);

    auto worstKernelError = 0.0;
    auto worstTanhError = 0.0;

    for (auto i = 0; i < sweepCount; ++i)
    {
        auto exact = exactGeluReference(input[i]);

        worstKernelError = std::max(worstKernelError, std::abs(result[i] - exact));
        worstTanhError =
            std::max(worstTanhError, std::abs(tanhGeluReference(input[i]) - exact));
    }

    check(worstTanhError > 1e-4);
    check(worstKernelError < worstTanhError / 100.0);
};

// Both tails, where the exponential inside erf underflows: GELU is the
// identity far to the right and zero far to the left, and neither may arrive
// as a NaN.
auto tGeluTailsStayFinite = test("Kernels/geluTailsStayFinite") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    const float tails[] = {-1000.f, -100.f, -30.f, 0.f, 30.f, 100.f, 1000.f};
    constexpr auto tailCount = (int) (sizeof(tails) / sizeof(tails[0]));

    auto input = sized(tailCount);

    for (auto i = 0; i < tailCount; ++i)
        input[i] = tails[i];

    auto inputBuffer = storageOf(input);
    auto output = outputFor(tailCount);

    auto kernel = Gelu {};
    kernel.input = inputBuffer;
    kernel.output = output;

    auto result = runOverRows(kernel, output, tailCount, tailCount);

    for (auto i = 0; i < tailCount; ++i)
    {
        check(std::isfinite(result[i]));
        check(isClose(result[i], input[i] > 0.f ? (double) input[i] : 0.0, 1e-6));
    }
};
