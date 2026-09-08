#include "MelTestSupport.h"
#include "ScalarMel.h"

using namespace nano;
using namespace eacp::GPU;

namespace
{
std::vector<float> runSpectrogram(Device& device,
                                  const WSP::MelShape& shape,
                                  const std::vector<float>& samples,
                                  const std::vector<float>& filters)
{
    const auto samplesBuffer = melTest::upload(device, samples);
    const auto filterBuffer = melTest::upload(device, filters);
    const auto output = melTest::allocate(device, shape.melElementCount());

    auto front = WSP::MelSpectrogram {shape};
    front.prepare(device);

    auto commands = device.makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        front.encode(pass, samplesBuffer, filterBuffer, output);
    }

    commands.commit();

    return melTest::download(output, shape.melElementCount());
}

WSP::MelShape reducedShape()
{
    auto shape = WSP::MelShape {};
    shape.sampleCount = 3200;
    shape.fftLength = 64;
    shape.hopLength = 16;
    shape.frameCount = 200;
    shape.melCount = 12;
    return shape;
}

// The log of the loudest cell, recovered from the output the scale was applied
// to. Whisper's floor is eight decades below it, so a cell's expected value
// cannot be written down without it, and the CPU cannot know it at Whisper's
// own shape without computing every cell.
double peakLogFrom(const std::vector<float>& normalised)
{
    const auto largest = *std::max_element(normalised.begin(), normalised.end());
    return (double) largest * 4.0 - 4.0;
}

double referenceLogMelCell(const std::vector<float>& samples,
                           const std::vector<float>& filters,
                           const WSP::MelShape& shape,
                           int band,
                           int frame)
{
    const auto window = scalar::periodicHann(shape.fftLength);
    const auto padded = scalar::reflectPadded(samples, shape.fftLength / 2);
    const auto binCount = shape.binCount();

    auto total = 0.0;

    for (auto bin = 0; bin < binCount; ++bin)
    {
        auto real = 0.0;
        auto imaginary = 0.0;

        for (auto tap = 0; tap < shape.fftLength; ++tap)
        {
            const auto amplitude =
                padded[(std::size_t) (frame * shape.hopLength + tap)]
                * window[(std::size_t) tap];
            const auto angle =
                scalar::twoPi() * bin * tap / (double) shape.fftLength;

            real += amplitude * std::cos(angle);
            imaginary -= amplitude * std::sin(angle);
        }

        total += filters[(std::size_t) (band * binCount + bin)]
                 * (real * real + imaginary * imaginary);
    }

    return std::log10(std::max(total, 1e-10));
}
} // namespace

auto tSpectrogramMatchesScalarReference =
    test("Mel/spectrogramMatchesScalarReference") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    const auto shape = reducedShape();
    const auto samples = melTest::toneAndNoise(shape.sampleCount);
    const auto filters =
        melTest::triangularFilters(shape.melCount, shape.binCount());

    const auto normalised = runSpectrogram(device, shape, samples, filters);
    const auto expected = scalar::melSpectrogram(samples,
                                                 filters,
                                                 shape.fftLength,
                                                 shape.hopLength,
                                                 shape.frameCount,
                                                 shape.melCount);

    check((int) normalised.size() == shape.melElementCount());

    for (auto i = 0; i < shape.melElementCount(); ++i)
        check(melTest::close(
            normalised[(std::size_t) i], expected[(std::size_t) i], 1e-4));
};

// Whisper's own shape. A scalar reference for all 240000 cells is 241 million
// windowed taps, so what is asserted here is the chain's arithmetic on a
// handful of cells computed the slow way, plus the span the normalisation
// gives the whole spectrogram — the first second of the signal is digital
// silence, so the eight-decade floor is certain to be reached somewhere and
// the output is exactly two wide.
auto tSpectrogramAtWhisperShape = test("Mel/spectrogramAtWhisperShape") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    const auto shape = WSP::MelShape {};
    const auto filters =
        melTest::triangularFilters(shape.melCount, shape.binCount());

    auto samples = melTest::toneAndNoise(shape.sampleCount);
    std::fill(samples.begin(), samples.begin() + WSP::sampleRate, 0.0f);

    const auto normalised = runSpectrogram(device, shape, samples, filters);

    check((int) normalised.size() == 80 * 3000);

    const auto largest = *std::max_element(normalised.begin(), normalised.end());
    const auto smallest = *std::min_element(normalised.begin(), normalised.end());

    check(melTest::close(smallest, largest - 2.0, 1e-5));

    const auto peakLog = peakLogFrom(normalised);

    for (const auto band: {0, 37, 79})
    {
        for (const auto frame: {0, 1, 1500, 2999})
        {
            const auto cell =
                referenceLogMelCell(samples, filters, shape, band, frame);
            const auto expected = (std::max(cell, peakLog - 8.0) + 4.0) / 4.0;

            check(melTest::close(
                normalised[(std::size_t) (band * shape.frameCount + frame)],
                expected,
                1e-4));
        }
    }
};
