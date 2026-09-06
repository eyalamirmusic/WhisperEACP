#include "MelSpectrogram.h"

#include "Window.h"

namespace WSP
{
using eacp::GPU::Buffer;
using eacp::GPU::BufferUsage;
using eacp::GPU::CommandBuffer;
using eacp::GPU::Device;

namespace
{
constexpr int floatBytes(int elementCount)
{
    return elementCount * (int) sizeof(float);
}

constexpr int groupsFor(int elementCount)
{
    constexpr auto width = MaxReduceKernel::groupWidth;
    return elementCount <= 0 ? 1 : (elementCount + width - 1) / width;
}
} // namespace

MelSpectrogram::MelSpectrogram(const MelShape& shapeToUse)
    : melShape(shapeToUse)
{
}

void MelSpectrogram::prepare(Device& device)
{
    spectrum.prepare(device);
    projection.prepare(device);
    reduction.prepare(device);
    normalisation.prepare(device);

    const auto hann = periodicHannWindow(melShape.fftLength);
    window.emplace(
        device, hann.data(), floatBytes(melShape.fftLength), BufferUsage::Storage);

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

void MelSpectrogram::encodeSpectrum(CommandBuffer& commands, const Buffer& samples)
{
    spectrum.samples = samples;
    spectrum.window = *window;
    spectrum.power = *power;
    spectrum.sampleCount = melShape.sampleCount;
    spectrum.fftLength = melShape.fftLength;
    spectrum.hopLength = melShape.hopLength;
    spectrum.binCount = (std::uint32_t) melShape.binCount();

    auto pass = commands.beginCompute();
    pass.dispatch(spectrum, melShape.binCount(), melShape.frameCount);
}

void MelSpectrogram::encodeProjection(CommandBuffer& commands,
                                      const Buffer& filterBank)
{
    projection.power = *power;
    projection.filters = filterBank;
    projection.logMel = *logMel;
    projection.binCount = (std::uint32_t) melShape.binCount();
    projection.frameCount = (std::uint32_t) melShape.frameCount;

    auto pass = commands.beginCompute();
    pass.dispatch(projection, melShape.frameCount, melShape.melCount);
}

// Rounds of a tree reduction, ping-ponging between the two partial buffers
// until one value is left. Each round is its own pass: threads of a dispatch
// are ordered against each other by the end of that dispatch and nothing else.
const Buffer& MelSpectrogram::encodePeak(CommandBuffer& commands)
{
    const auto* source = &logMel.value();
    auto remaining = melShape.melElementCount();
    auto intoPartials = true;

    while (remaining > 1)
    {
        const auto groups = groupsFor(remaining);
        auto& target = intoPartials ? partials.value() : partialsOfPartials.value();

        reduction.values = *source;
        reduction.partials = target;
        reduction.count = (std::uint32_t) remaining;
        reduction.stride = (std::uint32_t) (groups * MaxReduceKernel::groupWidth);

        {
            auto pass = commands.beginCompute();
            pass.dispatch(reduction, groups * MaxReduceKernel::groupWidth);
        }

        source = &target;
        remaining = groups;
        intoPartials = !intoPartials;
    }

    return *source;
}

void MelSpectrogram::encode(CommandBuffer& commands,
                            const Buffer& samples,
                            const Buffer& filterBank,
                            const Buffer& output)
{
    encodeSpectrum(commands, samples);
    encodeProjection(commands, filterBank);

    const auto& peak = encodePeak(commands);

    normalisation.values = *logMel;
    normalisation.peak = peak;
    normalisation.normalised = output;

    auto pass = commands.beginCompute();
    pass.dispatch(normalisation, melShape.melElementCount());
}
} // namespace WSP
