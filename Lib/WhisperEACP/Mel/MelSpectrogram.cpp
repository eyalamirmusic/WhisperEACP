#include "MelSpectrogram.h"

#include "Basis.h"
#include "Window.h"

namespace WSP
{
using eacp::GPU::Buffer;
using eacp::GPU::BufferUsage;
using eacp::GPU::ComputePass;
using eacp::GPU::Device;

namespace
{
constexpr int floatBytes(int elementCount)
{
    return elementCount * (int) sizeof(float);
}

constexpr int groupsFor(int elementCount)
{
    constexpr auto width = MaxReduceKernel::lanes;
    return elementCount <= 0 ? 1 : (elementCount + width - 1) / width;
}
} // namespace

MelSpectrogram::MelSpectrogram(const MelShape& shapeToUse)
    : melShape(shapeToUse)
{
}

void MelSpectrogram::prepare(Device& device)
{
    framing.prepare(device);
    transform.prepare(device);
    squaring.prepare(device);
    projection.prepare(device);
    reduction.prepare(device);
    normalisation.prepare(device);

    const auto hann = periodicHannWindow(melShape.fftLength);
    window.emplace(
        device, hann.data(), floatBytes(melShape.fftLength), BufferUsage::Storage);

    const auto complexBins = 2 * melShape.binCount();
    const auto dft = dftBasis(melShape.fftLength);
    basis.emplace(device, dft.data(), floatBytes(dft.size()), BufferUsage::Storage);

    auto zeroes = Vector<float>(complexBins);

    for (auto index = 0; index < complexBins; ++index)
        zeroes[index] = 0.f;

    zeroBias.emplace(
        device, zeroes.data(), floatBytes(complexBins), BufferUsage::Storage);

    frames.emplace(device,
                   nullptr,
                   floatBytes(melShape.frameCount * melShape.fftLength),
                   BufferUsage::Storage);

    spectrum.emplace(device,
                     nullptr,
                     floatBytes(melShape.frameCount * complexBins),
                     BufferUsage::Storage);

    power.emplace(device,
                  nullptr,
                  floatBytes(melShape.spectrumElementCount()),
                  BufferUsage::Storage);

    logMel.emplace(device,
                   nullptr,
                   floatBytes(melShape.melElementCount()),
                   BufferUsage::Storage);

    const auto firstRound = groupsFor(melShape.melElementCount());
    partials.emplace(device, nullptr, floatBytes(firstRound), BufferUsage::Storage);
    partialsOfPartials.emplace(
        device, nullptr, floatBytes(groupsFor(firstRound)), BufferUsage::Storage);
}

void MelSpectrogram::prepare()
{
    prepare(Device::shared());
}

// Frames, transform, power: the windowed frames gathered into rows, one
// tiled product of them against the basis, and each bin squared out of its
// real and imaginary parts. Each stage reads what the one before it wrote, so
// each boundary is a barrier — one that costs nothing in a serial pass and is
// the whole ordering in a concurrent one.
void MelSpectrogram::encodeSpectrum(ComputePass& pass, const Buffer& samples)
{
    framing.samples = samples;
    framing.window = *window;
    framing.frames = *frames;
    framing.sampleCount = melShape.sampleCount;
    framing.fftLength = melShape.fftLength;
    framing.hopLength = melShape.hopLength;

    pass.dispatch(framing, melShape.fftLength, melShape.frameCount);

    pass.barrier();

    transform.a = *frames;
    transform.b = *basis;
    transform.bias = *zeroBias;
    transform.output = *spectrum;

    transform.dispatch(pass,
                       TiledMatMulShape::forLinear(melShape.frameCount,
                                                   melShape.fftLength,
                                                   2 * melShape.binCount()));

    pass.barrier();

    squaring.spectrum = *spectrum;
    squaring.power = *power;
    squaring.binCount = (std::uint32_t) melShape.binCount();

    pass.dispatch(squaring, melShape.binCount(), melShape.frameCount);
}

void MelSpectrogram::encodeProjection(ComputePass& pass, const Buffer& filterBank)
{
    projection.power = *power;
    projection.filters = filterBank;
    projection.logMel = *logMel;
    projection.binCount = (std::uint32_t) melShape.binCount();
    projection.frameCount = (std::uint32_t) melShape.frameCount;

    pass.dispatch(projection, melShape.frameCount, melShape.melCount);
}

// Rounds of a tree reduction, ping-ponging between the two partial buffers
// until one value is left. Each round is its own dispatch: threads of a
// dispatch are ordered against each other by the end of that dispatch and
// nothing else. A round folds what the round before it wrote, so every round
// after the first opens with a barrier; the first is ordered by the caller.
const Buffer& MelSpectrogram::encodePeak(ComputePass& pass)
{
    const auto* source = &logMel.value();
    auto remaining = melShape.melElementCount();
    auto intoPartials = true;
    auto isFirstRound = true;

    while (remaining > 1)
    {
        if (!isFirstRound)
            pass.barrier();

        isFirstRound = false;

        const auto groups = groupsFor(remaining);
        auto& target = intoPartials ? partials.value() : partialsOfPartials.value();

        reduction.values = *source;
        reduction.partials = target;
        reduction.count = (std::uint32_t) remaining;
        reduction.stride = (std::uint32_t) (groups * MaxReduceKernel::lanes);

        pass.dispatch(reduction, groups * MaxReduceKernel::lanes);

        source = &target;
        remaining = groups;
        intoPartials = !intoPartials;
    }

    return *source;
}

void MelSpectrogram::encode(ComputePass& pass,
                            const Buffer& samples,
                            const Buffer& filterBank,
                            const Buffer& output)
{
    encodeSpectrum(pass, samples);

    pass.barrier();

    encodeProjection(pass, filterBank);

    pass.barrier();

    const auto& peak = encodePeak(pass);

    pass.barrier();

    normalisation.values = *logMel;
    normalisation.peak = peak;
    normalisation.normalised = output;

    pass.dispatch(normalisation, melShape.melElementCount());
}
} // namespace WSP
