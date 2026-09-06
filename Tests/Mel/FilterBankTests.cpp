#include "MelTestSupport.h"
#include "ScalarMel.h"

using namespace nano;
using namespace eacp::GPU;

namespace
{
// The EDSL has no log10, so the kernel takes a natural log and changes base.
// Both backends leave a shader's log a few units in the last place loose, and
// that is a few parts in a million of a log around ten rather than of the
// value it came from — hence a tolerance that grows with the magnitude of the
// log rather than a fixed one.
constexpr auto logTolerance = 1e-5;
constexpr auto floorTolerance = 1e-4;

std::vector<float> runProjection(Device& device,
                                 const std::vector<float>& power,
                                 const std::vector<float>& filters,
                                 int binCount,
                                 int frameCount,
                                 int melCount)
{
    const auto powerBuffer = melTest::upload(device, power);
    const auto filterBuffer = melTest::upload(device, filters);
    const auto logMelBuffer = melTest::allocate(device, melCount * frameCount);

    auto kernel = WSP::MelProjectKernel {};
    kernel.power = powerBuffer;
    kernel.filters = filterBuffer;
    kernel.logMel = logMelBuffer;
    kernel.binCount = (std::uint32_t) binCount;
    kernel.frameCount = (std::uint32_t) frameCount;
    kernel.prepare(device);

    auto commands = device.makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        pass.dispatch(kernel, frameCount, melCount);
    }

    commands.commit();

    return melTest::download(logMelBuffer, melCount * frameCount);
}
} // namespace

// A matrix small enough to multiply by hand: the power spectrum is frame major
// and the mel output is band major, and getting either transposed is the
// failure this catches.
auto tFilterBankMultipliesByHand = test("Mel/filterBankMultipliesByHand") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto binCount = 3;
    constexpr auto frameCount = 2;
    constexpr auto melCount = 2;

    const auto power = std::vector<float> {1.0f, 10.0f, 100.0f, 2.0f, 20.0f, 200.0f};
    const auto filters = std::vector<float> {1.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.25f};

    const auto logMel =
        runProjection(device, power, filters, binCount, frameCount, melCount);

    check(melTest::close(logMel[0], std::log10(1.0), logTolerance));
    check(melTest::close(logMel[1], std::log10(2.0), logTolerance));
    check(melTest::close(
        logMel[2], std::log10(0.5 * 10.0 + 0.25 * 100.0), logTolerance));
    check(melTest::close(
        logMel[3], std::log10(0.5 * 20.0 + 0.25 * 200.0), logTolerance));
};

// The floor is 1e-10 before the log, so a band that sums to nothing reads as
// exactly minus ten rather than as minus infinity.
auto tFilterBankFloorsBeforeTheLog = test("Mel/filterBankFloorsBeforeTheLog") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto binCount = 4;
    constexpr auto frameCount = 3;
    constexpr auto melCount = 2;

    auto power = std::vector<float>((std::size_t) (frameCount * binCount), 0.0f);
    power[0] = 1.0f;

    auto filters = std::vector<float>((std::size_t) (melCount * binCount), 0.0f);
    filters[0] = 1.0f;

    const auto logMel =
        runProjection(device, power, filters, binCount, frameCount, melCount);

    check(melTest::close(logMel[0], 0.0, logTolerance));

    for (auto frame = 1; frame < frameCount; ++frame)
        check(melTest::close(logMel[(std::size_t) frame], -10.0, floorTolerance));

    for (auto frame = 0; frame < frameCount; ++frame)
        check(melTest::close(
            logMel[(std::size_t) (frameCount + frame)], -10.0, floorTolerance));
};

auto tFilterBankMatchesScalarReference =
    test("Mel/filterBankMatchesScalarReference") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto binCount = 33;
    constexpr auto frameCount = 40;
    constexpr auto melCount = 12;

    auto power = std::vector<float>((std::size_t) (frameCount * binCount));
    auto state = 987u;

    for (auto& value: power)
    {
        state = state * 1103515245u + 12345u;
        value = (float) ((state >> 8) % 100000u) * 1e-3f;
    }

    const auto filters = melTest::triangularFilters(melCount, binCount);
    const auto logMel =
        runProjection(device, power, filters, binCount, frameCount, melCount);

    const auto reference = std::vector<double>(power.begin(), power.end());
    const auto expected =
        scalar::logMel(reference, filters, melCount, binCount, frameCount);

    for (auto i = 0; i < melCount * frameCount; ++i)
        check(melTest::close(
            logMel[(std::size_t) i], expected[(std::size_t) i], logTolerance));
};
