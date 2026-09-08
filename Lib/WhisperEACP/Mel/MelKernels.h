#pragma once

#include <eacp/GPU/GPU.h>

#include <limits>

// The four dispatches Whisper's front-end is. Each takes its shapes as
// uniforms, so the same programs run over a 30 s window and over a signal
// short enough for a scalar reference to be written out by hand.

namespace WSP
{
using eacp::GPU::ComputeProgram;
using eacp::GPU::Float;
using eacp::GPU::InputBuffer;
using eacp::GPU::Int;
using eacp::GPU::OutputBuffer;
using eacp::GPU::UInt;
using eacp::GPU::Uniform;

inline constexpr auto radiansPerTurn = 6.283185307179586f;
inline constexpr auto lowestFloat = std::numeric_limits<float>::lowest();

// Where a padded index lands once folded back inside [0, lastIndex]: the
// reflect the STFT's centre padding is spelled with, about the edge sample
// rather than about the edge itself. abs() twice rather than two branches,
// which is the same fold for a step off either end.
inline Int reflectedIndex(const Int& index, const Int& lastIndex)
{
    return lastIndex - abs(lastIndex - abs(index));
}

// One thread per (tap, frame): the windowed, reflect-padded frame, laid out
// [frameCount, fftLength] so the transform can be one product of every frame
// against the DFT basis (Basis.h) rather than a per-bin loop — which is what
// StftPowerKernel below does, and at 3.4 ms of a 12 ms encoder was the
// largest kernel in it.
//
// The frame is centred on hopLength * frame, which is center=True: the signal
// is reflected by fftLength / 2 at each end rather than the frame starting
// there. Frame 3000 — the one HF computes and then drops — is simply never
// dispatched.
struct StftFramesKernel final : ComputeProgram
{
    StftFramesKernel() { compile(); }

    void define() override
    {
        auto position = threadPosition();
        auto tap = toInt(position.x);
        auto windowStart = toInt(position.y) * hopLength - fftLength / 2;
        auto source = reflectedIndex(windowStart + tap, sampleCount - 1);

        write(frames,
              position.y * toUInt(fftLength) + position.x,
              samples[toUInt(source)] * window[position.x]);
    }

    Uniform<InputBuffer> samples;
    Uniform<InputBuffer> window;
    Uniform<OutputBuffer> frames;
    Uniform<Int> sampleCount;
    Uniform<Int> fftLength;
    Uniform<Int> hopLength;

    EACP_SHADER(samples, window, frames, sampleCount, fftLength, hopLength)
};

// One thread per (bin, frame): the power of a bin out of the real and
// imaginary parts the transform left side by side.
struct SpectrumPowerKernel final : ComputeProgram
{
    SpectrumPowerKernel() { compile(); }

    void define() override
    {
        auto position = threadPosition();
        auto at = position.y * (2u * binCount) + 2u * position.x;
        auto real = spectrum[at];
        auto imaginary = spectrum[at + 1u];

        write(power,
              position.y * binCount + position.x,
              real * real + imaginary * imaginary);
    }

    Uniform<InputBuffer> spectrum;
    Uniform<OutputBuffer> power;
    Uniform<UInt> binCount;

    EACP_SHADER(spectrum, power, binCount)
};

// One thread per (bin, frame) of the power spectrum: a windowed DFT evaluated
// bin by bin, squared. Naive on purpose — the framing and product above are
// the optimisation this is the reference for, and the tests hold the two to
// each other.
//
// The frame is centred on hopLength * frame, which is center=True: the signal
// is reflected by fftLength / 2 at each end rather than the frame starting
// there. Frame 3000 — the one HF computes and then drops — is simply never
// dispatched.
struct StftPowerKernel final : ComputeProgram
{
    StftPowerKernel() { compile(); }

    void define() override
    {
        auto position = threadPosition();
        auto bin = toInt(position.x);
        auto windowStart = toInt(position.y) * hopLength - fftLength / 2;
        auto radiansPerStep = constant(radiansPerTurn) / toFloat(fftLength);

        auto real = var(0.f);
        auto imaginary = var(0.f);
        auto tap = var(0);

        loop(tap.get() < fftLength,
             [&]
             {
                 auto source =
                     reflectedIndex(windowStart + tap.get(), sampleCount - 1);
                 auto amplitude =
                     samples[toUInt(source)] * window[toUInt(tap.get())];
                 auto angle =
                     radiansPerStep * toFloat((bin * tap.get()) % fftLength);

                 real += amplitude * cos(angle);
                 imaginary -= amplitude * sin(angle);
                 tap += 1;
             });

        write(power,
              position.y * binCount + position.x,
              real.get() * real.get() + imaginary.get() * imaginary.get());
    }

    Uniform<InputBuffer> samples;
    Uniform<InputBuffer> window;
    Uniform<OutputBuffer> power;
    Uniform<Int> sampleCount;
    Uniform<Int> fftLength;
    Uniform<Int> hopLength;
    Uniform<UInt> binCount;

    EACP_SHADER(samples, window, power, sampleCount, fftLength, hopLength, binCount)
};

// filters @ power, then log10 of the result floored at 1e-10. One thread per
// (frame, mel band); the filterbank is bound by the caller, never built here.
struct MelProjectKernel final : ComputeProgram
{
    MelProjectKernel() { compile(); }

    void define() override
    {
        auto position = threadPosition();
        auto total = var(0.f);
        auto bin = var(0u);

        loop(bin.get() < binCount,
             [&]
             {
                 total += filters[position.y * binCount + bin.get()]
                          * power[position.x * binCount + bin.get()];
                 bin += 1u;
             });

        write(logMel,
              position.y * frameCount + position.x,
              log10(max(total.get(), 1e-10f)));
    }

    Uniform<InputBuffer> power;
    Uniform<InputBuffer> filters;
    Uniform<OutputBuffer> logMel;
    Uniform<UInt> binCount;
    Uniform<UInt> frameCount;

    EACP_SHADER(power, filters, logMel, binCount, frameCount)
};

// One partial maximum per threadgroup, so the whole spectrogram's maximum is
// reached by dispatching this over its own output until a single value is
// left. Whisper's floor is taken against that global maximum, and the end of a
// dispatch is the only thing that orders one thread's write against another's
// read.
//
// lanes is the group this is dispatched in and what the host divides its
// element count by, so the two cannot drift; it is named here rather than
// taken as ComputeProgram::groupWidth because this kernel's group is its own.
//
// 512 of them, which is the one lane count here that a wide group wins by
// arithmetic rather than by occupancy: the 240000 cells of a 30 s spectrogram
// come down in two rounds at 512 lanes where every count up to 256 takes
// three, and 17 us of hand tree measured down to 10.5. Both backends that ship
// allow 1024 threads to a group, so the ceiling is not near.
struct MaxReduceKernel final : ComputeProgram
{
    static constexpr auto lanes = 512;

    MaxReduceKernel()
        : ComputeProgram({lanes})
    {
        compile();
    }

    void define() override
    {
        auto best = var(lowestFloat);
        auto index = var(threadId());

        loop(index.get() < count,
             [&]
             {
                 best = max(best.get(), values[index.get()]);
                 index += stride;
             });

        auto folded = groupMax(best.get());

        ifThen(localId() == 0u, [&] { write(partials, groupId(), folded); });
    }

    Uniform<InputBuffer> values;
    Uniform<OutputBuffer> partials;
    Uniform<UInt> count;
    Uniform<UInt> stride;

    EACP_SHADER(values, partials, count, stride)
};

// The tail of Whisper's normalisation: floor eight decades below the loudest
// cell of the whole spectrogram, then shift and scale into the range the model
// was trained on. peak holds that single maximum, read from element zero.
struct MelNormaliseKernel final : ComputeProgram
{
    MelNormaliseKernel() { compile(); }

    void define() override
    {
        auto index = threadId();
        auto origin = var(0u);
        auto floorValue = peak[origin.get()] - 8.0f;

        write(normalised, index, (max(values[index], floorValue) + 4.0f) / 4.0f);
    }

    Uniform<InputBuffer> values;
    Uniform<InputBuffer> peak;
    Uniform<OutputBuffer> normalised;

    EACP_SHADER(values, peak, normalised)
};
} // namespace WSP
