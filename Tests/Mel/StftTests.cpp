#include "MelTestSupport.h"
#include "ScalarMel.h"

using namespace nano;
using namespace eacp::GPU;

namespace
{
std::vector<float> runStft(Device& device,
                           const std::vector<float>& samples,
                           const std::vector<float>& window,
                           int hopLength,
                           int frameCount)
{
    const auto fftLength = (int) window.size();
    const auto binCount = fftLength / 2 + 1;

    const auto samplesBuffer = melTest::upload(device, samples);
    const auto windowBuffer = melTest::upload(device, window);
    const auto powerBuffer = melTest::allocate(device, frameCount * binCount);

    auto kernel = WSP::StftPowerKernel {};
    kernel.samples = samplesBuffer;
    kernel.window = windowBuffer;
    kernel.power = powerBuffer;
    kernel.sampleCount = (std::int32_t) samples.size();
    kernel.fftLength = (std::int32_t) fftLength;
    kernel.hopLength = (std::int32_t) hopLength;
    kernel.binCount = (std::uint32_t) binCount;
    kernel.prepare(device);

    auto commands = device.makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        pass.dispatch(kernel, binCount, frameCount);
    }

    commands.commit();

    return melTest::download(powerBuffer, frameCount * binCount);
}

std::vector<float> rectangularWindow(int length)
{
    return std::vector<float>((std::size_t) length, 1.0f);
}

std::vector<float> ramp(int count)
{
    auto samples = std::vector<float>((std::size_t) count);

    for (auto i = 0; i < count; ++i)
        samples[(std::size_t) i] = (float) i;

    return samples;
}
} // namespace

// Bin zero of a rectangular window is the plain sum of the samples the frame
// covers, so it reads the padding straight out. Frame 0 of an eight-point
// window over 0, 1, 2, ... covers padded taps 4 3 2 1 0 1 2 3 — the mirror
// about sample 0, with sample 0 itself appearing once — which sums to 16.
auto tStftReflectsAtBothEdges = test("Mel/stftReflectsAtBothEdges") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto fftLength = 8;
    constexpr auto hopLength = 4;
    constexpr auto sampleCount = 16;
    constexpr auto frameCount = 5;
    constexpr auto binCount = fftLength / 2 + 1;

    const auto samples = ramp(sampleCount);
    const auto power = runStft(
        device, samples, rectangularWindow(fftLength), hopLength, frameCount);

    check(melTest::close(power[0], 16.0 * 16.0, 1e-2));

    const auto padded = scalar::reflectPadded(samples, fftLength / 2);

    for (auto frame = 0; frame < frameCount; ++frame)
    {
        auto total = 0.0;

        for (auto tap = 0; tap < fftLength; ++tap)
            total += padded[(std::size_t) (frame * hopLength + tap)];

        check(melTest::close(
            power[(std::size_t) (frame * binCount)], total * total, 1e-2));
    }

    // The last frame reaches four samples past the end, so it is the mirror
    // about sample 15 that decides it: 12 13 14 15 14 13 12 11.
    check(melTest::close(power[(std::size_t) (4 * binCount)], 104.0 * 104.0, 1e-1));
};

// A sinusoid whose period divides the window lands entirely in one bin under a
// rectangular window, with every other bin exactly zero — the check that the
// transform is a DFT of the taps it says it is, and not off by a sample.
auto tStftPlacesABinCentredTone = test("Mel/stftPlacesABinCentredTone") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto fftLength = 16;
    constexpr auto hopLength = 8;
    constexpr auto sampleCount = 64;
    constexpr auto frameCount = 6;
    constexpr auto binCount = fftLength / 2 + 1;
    constexpr auto tone = 3;

    auto samples = std::vector<float>((std::size_t) sampleCount);

    for (auto i = 0; i < sampleCount; ++i)
        samples[(std::size_t) i] =
            (float) std::cos(scalar::twoPi() * tone * i / fftLength);

    const auto power = runStft(
        device, samples, rectangularWindow(fftLength), hopLength, frameCount);

    // Frame 2 starts at sample 8, so it is clear of the padding at either end.
    const auto frame = 2;

    for (auto bin = 0; bin < binCount; ++bin)
    {
        const auto value = power[(std::size_t) (frame * binCount + bin)];
        const auto expected =
            bin == tone ? (double) (fftLength * fftLength) / 4.0 : 0.0;

        check(melTest::close(value, expected, 1e-3));
    }
};

auto tStftMatchesScalarReference = test("Mel/stftMatchesScalarReference") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto fftLength = 32;
    constexpr auto hopLength = 8;
    constexpr auto sampleCount = 300;
    constexpr auto frameCount = 30;
    constexpr auto binCount = fftLength / 2 + 1;

    const auto samples = melTest::toneAndNoise(sampleCount);
    const auto hann = WSP::periodicHannWindow(fftLength);
    const auto window = std::vector<float>(hann.data(), hann.data() + hann.size());

    const auto power = runStft(device, samples, window, hopLength, frameCount);
    const auto expected = scalar::stftPower(
        samples, scalar::periodicHann(fftLength), fftLength, hopLength, frameCount);

    auto largest = 0.0;

    for (auto i = 0; i < frameCount * binCount; ++i)
        largest = std::max(largest, expected[(std::size_t) i]);

    for (auto i = 0; i < frameCount * binCount; ++i)
        check(melTest::close(power[(std::size_t) i],
                             expected[(std::size_t) i],
                             largest * 1e-5 + expected[(std::size_t) i] * 1e-4));
};
