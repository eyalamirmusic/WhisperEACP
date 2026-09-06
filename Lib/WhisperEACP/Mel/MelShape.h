#pragma once

#include <WhisperEACP/Audio/Audio.h>

namespace WSP
{
// feature_size in preprocessor_config.json, and the height of the filterbank
// matrix published there. The encoder's first convolution is written against
// it the way every shape downstream is written against windowFrames.
inline constexpr auto melBands = 80;

// The non-redundant half of a real 400-point spectrum, which is what the
// filterbank's 201 columns are.
inline constexpr auto spectrumBins = fftSize / 2 + 1;

// Whisper's shapes by default, and something a test can shrink. The kernels
// take every one of these as a uniform, so a reduced spectrogram is the same
// code over a signal short enough to keep a scalar reference tractable.
struct MelShape
{
    int sampleCount = windowSamples;
    int fftLength = fftSize;
    int hopLength = hopSize;
    int frameCount = windowFrames;
    int melCount = melBands;

    int binCount() const { return fftLength / 2 + 1; }
    int spectrumElementCount() const { return frameCount * binCount(); }
    int filterElementCount() const { return melCount * binCount(); }
    int melElementCount() const { return melCount * frameCount; }
};
} // namespace WSP
