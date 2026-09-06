#include "ScalarMel.h"

#include <WhisperEACP/Mel/MelShape.h>
#include <WhisperEACP/Mel/Window.h>

#include <NanoTest/NanoTest.h>

#include <cmath>

using namespace nano;

namespace
{
constexpr auto windowTolerance = 1e-6;

double symmetricHann(int index, int length)
{
    return 0.5 * (1.0 - std::cos(scalar::twoPi() * index / (length - 1)));
}
} // namespace

auto tHannMatchesScalarReference = test("Mel/hannMatchesScalarReference") = []
{
    constexpr auto length = 400;

    const auto expected = scalar::periodicHann(length);
    const auto window = WSP::periodicHannWindow(length);

    check(window.size() == length);

    for (auto n = 0; n < length; ++n)
        check(std::abs(window[n] - expected[(std::size_t) n]) < windowTolerance);
};

// The two variants differ by dividing by length rather than by length - 1, and
// that difference is small enough everywhere to survive an eyeball: the last
// point is the one place it is unmistakable, because the symmetric window ends
// at zero and the periodic one does not.
auto tHannIsPeriodicNotSymmetric = test("Mel/hannIsPeriodicNotSymmetric") = []
{
    constexpr auto length = 400;

    const auto window = WSP::periodicHannWindow(length);

    check(std::abs(window[0]) < windowTolerance);
    check(std::abs(window[length / 2] - 1.0) < windowTolerance);

    check(window[length - 1] > 6.1e-5 && window[length - 1] < 6.3e-5);
    check(std::abs(symmetricHann(length - 1, length)) < windowTolerance);

    for (auto n = 1; n < length; ++n)
        check(std::abs(window[n] - window[length - n]) < windowTolerance);

    auto largestGap = 0.0;

    for (auto n = 0; n < length; ++n)
        largestGap =
            std::max(largestGap, std::abs(window[n] - symmetricHann(n, length)));

    check(largestGap > 1e-3);
};

auto tHannValueMatchesTheWindow = test("Mel/hannValueMatchesTheWindow") = []
{
    const auto window = WSP::periodicHannWindow(WSP::fftSize);

    for (auto n = 0; n < WSP::fftSize; ++n)
        check(WSP::periodicHannValue(n, WSP::fftSize) == window[n]);
};
