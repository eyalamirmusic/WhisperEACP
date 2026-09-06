#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

// Whisper's front-end written out the obvious way: scalar, double, and on the
// CPU. This is what every kernel in the module is asserted against, and it
// shares no line with them — the padding is a padded array here rather than an
// index fold, and the transform is a textbook DFT rather than a dispatch.
namespace scalar
{
inline double twoPi()
{
    return 2.0 * std::numbers::pi;
}

inline std::vector<double> periodicHann(int length)
{
    auto window = std::vector<double>((std::size_t) length);

    for (auto n = 0; n < length; ++n)
        window[(std::size_t) n] = 0.5 * (1.0 - std::cos(twoPi() * n / length));

    return window;
}

// numpy's "reflect", which is what center=True pads with: the signal mirrors
// about its first and last sample, and neither of those is repeated.
inline std::vector<double> reflectPadded(const std::vector<float>& samples, int pad)
{
    const auto count = (int) samples.size();
    auto padded = std::vector<double>((std::size_t) (count + 2 * pad));

    for (auto i = 0; i < pad; ++i)
        padded[(std::size_t) i] = samples[(std::size_t) (pad - i)];

    for (auto i = 0; i < count; ++i)
        padded[(std::size_t) (pad + i)] = samples[(std::size_t) i];

    for (auto i = 0; i < pad; ++i)
        padded[(std::size_t) (pad + count + i)] =
            samples[(std::size_t) (count - 2 - i)];

    return padded;
}

inline std::vector<double> stftPower(const std::vector<float>& samples,
                                     const std::vector<double>& window,
                                     int fftLength,
                                     int hopLength,
                                     int frameCount)
{
    const auto binCount = fftLength / 2 + 1;
    const auto padded = reflectPadded(samples, fftLength / 2);
    auto power = std::vector<double>((std::size_t) (frameCount * binCount));

    for (auto frame = 0; frame < frameCount; ++frame)
    {
        for (auto bin = 0; bin < binCount; ++bin)
        {
            auto real = 0.0;
            auto imaginary = 0.0;

            for (auto tap = 0; tap < fftLength; ++tap)
            {
                const auto amplitude =
                    padded[(std::size_t) (frame * hopLength + tap)]
                    * window[(std::size_t) tap];
                const auto angle = twoPi() * bin * tap / fftLength;

                real += amplitude * std::cos(angle);
                imaginary -= amplitude * std::sin(angle);
            }

            power[(std::size_t) (frame * binCount + bin)] =
                real * real + imaginary * imaginary;
        }
    }

    return power;
}

inline std::vector<double> logMel(const std::vector<double>& power,
                                  const std::vector<float>& filters,
                                  int melCount,
                                  int binCount,
                                  int frameCount)
{
    auto result = std::vector<double>((std::size_t) (melCount * frameCount));

    for (auto band = 0; band < melCount; ++band)
    {
        for (auto frame = 0; frame < frameCount; ++frame)
        {
            auto total = 0.0;

            for (auto bin = 0; bin < binCount; ++bin)
                total += filters[(std::size_t) (band * binCount + bin)]
                         * power[(std::size_t) (frame * binCount + bin)];

            result[(std::size_t) (band * frameCount + frame)] =
                std::log10(std::max(total, 1e-10));
        }
    }

    return result;
}

inline std::vector<double> normalise(std::vector<double> values)
{
    const auto peak = *std::max_element(values.begin(), values.end());

    for (auto& value: values)
        value = (std::max(value, peak - 8.0) + 4.0) / 4.0;

    return values;
}

inline std::vector<double> melSpectrogram(const std::vector<float>& samples,
                                          const std::vector<float>& filters,
                                          int fftLength,
                                          int hopLength,
                                          int frameCount,
                                          int melCount)
{
    const auto window = periodicHann(fftLength);
    const auto power = stftPower(samples, window, fftLength, hopLength, frameCount);

    return normalise(
        logMel(power, filters, melCount, fftLength / 2 + 1, frameCount));
}
} // namespace scalar
