#pragma once

#include <WhisperEACP/Mel/Mel.h>

#include <eacp/GPU/GPU.h>

#include <NanoTest/NanoTest.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace melTest
{
inline eacp::GPU::Buffer upload(eacp::GPU::Device& device,
                                const std::vector<float>& values)
{
    return device.makeBuffer(values.data(),
                             (int) (values.size() * sizeof(float)),
                             eacp::GPU::BufferUsage::Storage);
}

inline eacp::GPU::Buffer allocate(eacp::GPU::Device& device, int elementCount)
{
    return device.makeBuffer((int) (elementCount * (int) sizeof(float)));
}

inline std::vector<float> download(const eacp::GPU::Buffer& buffer, int elementCount)
{
    auto values = std::vector<float>((std::size_t) elementCount);
    buffer.read(values.data(), (int) (elementCount * (int) sizeof(float)));
    return values;
}

inline bool close(double actual, double expected, double tolerance)
{
    return std::abs(actual - expected) <= tolerance;
}

// A signal with no exact spectral null anywhere, which is what keeps a log of
// a float power spectrum comparable with a log of a double one: a bin that
// cancels to zero in double has no meaningful number of correct digits in
// float, and would make a tolerance a statement about luck.
inline std::vector<float> toneAndNoise(int sampleCount)
{
    auto samples = std::vector<float>((std::size_t) sampleCount);
    auto state = 12345u;

    for (auto i = 0; i < sampleCount; ++i)
    {
        state = state * 1664525u + 1013904223u;

        const auto noise = (double) (state >> 8) / (double) (1u << 24) - 0.5;
        const auto phase = 2.0 * std::numbers::pi * i;

        samples[(std::size_t) i] =
            (float) (0.6 * std::sin(phase * 440.0 / WSP::sampleRate)
                     + 0.3 * std::sin(phase * 1237.0 / WSP::sampleRate)
                     + 0.1 * noise);
    }

    return samples;
}

// A filterbank shaped like the published one — overlapping triangles across
// the spectrum — without being it. What is under test is the multiply, and the
// real matrix arrives from preprocessor_config.json through a buffer.
inline std::vector<float> triangularFilters(int melCount, int binCount)
{
    auto filters = std::vector<float>((std::size_t) (melCount * binCount));
    const auto span = (double) (binCount - 1) / (double) (melCount + 1);

    for (auto band = 0; band < melCount; ++band)
    {
        const auto centre = span * (band + 1);

        for (auto bin = 0; bin < binCount; ++bin)
        {
            const auto distance = std::abs(bin - centre) / span;
            const auto weight = distance < 1.0 ? 1.0 - distance : 0.0;

            filters[(std::size_t) (band * binCount + bin)] = (float) weight;
        }
    }

    return filters;
}
} // namespace melTest
