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

// The EDSL has log and log2 and no log10, so this is the change of base rather
// than a helper hiding a gap: log10 belongs in eacp beside the two that are
// already there, and this is what it would be.
inline Float decimalLog(const Float& value)
{
    constexpr auto reciprocalLogOfTen = 0.43429448190325176f;
    return log(value) * reciprocalLogOfTen;
}

// Where a padded index lands once folded back inside [0, lastIndex]: the
// reflect the STFT's centre padding is spelled with, about the edge sample
// rather than about the edge itself. abs() twice rather than two branches,
// which is the same fold for a step off either end.
inline Int reflectedIndex(const Int& index, const Int& lastIndex)
{
    return lastIndex - abs(lastIndex - abs(index));
}

// One thread per (bin, frame) of the power spectrum: a windowed DFT evaluated
// bin by bin, squared. Naive on purpose — an FFT is the optimisation this is
// the reference for.
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
              decimalLog(max(total.get(), 1e-10f)));
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
struct MaxReduceKernel final : ComputeProgram
{
    MaxReduceKernel() { compile(); }

    void define() override
    {
        auto lane = localId();
        auto tile = shared<Float>(groupWidth);

        auto best = var(lowestFloat);
        auto index = var(threadId());

        loop(index.get() < count,
             [&]
             {
                 best = max(best.get(), values[index.get()]);
                 index += stride;
             });

        write(tile, lane, best.get());
        barrier();

        for (auto span = (unsigned) groupWidth / 2u; span > 0u; span /= 2u)
        {
            ifThen(lane < span,
                   [&] { write(tile, lane, max(tile[lane], tile[lane + span])); });

            barrier();
        }

        ifThen(lane == 0u, [&] { write(partials, groupId(), tile[0u]); });
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
