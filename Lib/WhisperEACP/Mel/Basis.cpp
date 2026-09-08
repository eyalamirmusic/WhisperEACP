#include "Basis.h"

#include <cmath>
#include <numbers>

namespace WSP
{
Vector<float> dftBasis(int fftLength)
{
    const auto binCount = fftLength / 2 + 1;
    auto basis = Vector<float>(2 * binCount * fftLength);

    for (auto bin = 0; bin < binCount; ++bin)
    {
        for (auto tap = 0; tap < fftLength; ++tap)
        {
            const auto turns =
                (double) ((bin * tap) % fftLength) / (double) fftLength;
            const auto angle = 2.0 * std::numbers::pi * turns;

            basis[(2 * bin) * fftLength + tap] = (float) std::cos(angle);
            basis[(2 * bin + 1) * fftLength + tap] = (float) -std::sin(angle);
        }
    }

    return basis;
}
} // namespace WSP
