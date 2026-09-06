#pragma once

#include <MakeASound/Devices/DeviceInfo.h>

namespace WSP
{
// The audio Whisper was trained against, and the only shape its front-end
// accepts. None of these is a choice: capture at any other rate is resampled to
// this one before the mel filterbank sees it, and a window of any other length
// lands the encoder's positional embedding on the wrong frames.
inline constexpr auto sampleRate = 16000;
inline constexpr auto channelCount = 1;

// The STFT the filterbank is applied to: a 25 ms window every 10 ms.
inline constexpr auto fftSize = 400;
inline constexpr auto hopSize = 160;

// 30 s, padded with silence or trimmed to exactly this before the encoder runs.
// A shorter utterance costs the same dispatch as a full one.
inline constexpr auto windowSeconds = 30;
inline constexpr auto windowSamples = sampleRate * windowSeconds;
inline constexpr auto windowFrames = windowSamples / hopSize;

// The encoder's second convolution has stride 2, so it halves the frame count
// before the transformer blocks — every attention shape downstream is built
// against this number rather than against windowFrames.
inline constexpr auto encoderPositions = windowFrames / 2;

// Whole hops only. A frame needs fftSize samples behind it, so a run shorter
// than one window yields nothing rather than a partly-filled frame.
constexpr int framesForSamples(int sampleCount)
{
    return sampleCount < fftSize ? 0 : 1 + (sampleCount - fftSize) / hopSize;
}

constexpr int samplesForSeconds(double seconds)
{
    return (int) (seconds * sampleRate);
}

constexpr double secondsForFrames(int frameCount)
{
    return (double) (frameCount * hopSize) / sampleRate;
}

// Whether this device can be opened at Whisper's rate directly, which is the
// difference between a capture that feeds the front-end as it arrives and one
// that has to be resampled first. False for a device with no input at all.
bool canCaptureNatively(const MakeASound::DeviceInfo& device);
} // namespace WSP
