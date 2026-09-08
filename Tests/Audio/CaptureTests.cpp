#include <WhisperEACP/Audio/Audio.h>

#include <NanoTest/NanoTest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <thread>

// The arithmetic the device callback runs is free functions on purpose: every
// check below runs on a machine with no microphone, and the one test that needs
// a real device is gated on an environment variable because a CI runner has no
// microphone and a terminal may not have been granted one.

using namespace nano;
using namespace WSP;

namespace
{
constexpr auto tolerance = 1.0e-4f;

bool near(float value, float expected)
{
    return std::abs(value - expected) < tolerance;
}

MakeASound::Buffer planarBlock(Span<float> data, int channels)
{
    return MakeASound::Buffer {data.data(), channels, data.size() / channels};
}

bool liveCaptureRequested()
{
    return std::getenv("WHISPER_EACP_CAPTURE_TEST") != nullptr;
}
} // namespace

auto tMixToMonoAverages = test("Capture/mixToMonoAverages") = []
{
    auto planar = std::array<float, 8> {1.f, 2.f, 3.f, 4.f, 3.f, 6.f, 9.f, 12.f};
    auto mono = std::array<float, 4> {};

    mixToMono(planarBlock(planar, 2), mono);

    check(near(mono[0], 2.f));
    check(near(mono[1], 4.f));
    check(near(mono[2], 6.f));
    check(near(mono[3], 8.f));
};

auto tMixToMonoCopiesOneChannel = test("Capture/mixToMonoCopiesOneChannel") = []
{
    auto planar = std::array<float, 4> {-1.f, -0.5f, 0.25f, 1.f};
    auto mono = std::array<float, 4> {};

    mixToMono(planarBlock(planar, 1), mono);

    for (auto i = 0; i < 4; ++i)
        check(near(mono[i], planar[(std::size_t) i]));
};

auto tMixToMonoZeroesAChannellessBlock =
    test("Capture/mixToMonoZeroesAChannellessBlock") = []
{
    auto mono = std::array<float, 3> {1.f, 1.f, 1.f};

    mixToMono(MakeASound::Buffer {}, mono);

    check(mono[0] == 0.f && mono[1] == 0.f && mono[2] == 0.f);
};

auto tMeasureLevelOfSilence = test("Capture/measureLevelOfSilence") = []
{
    auto silence = std::array<float, 64> {};
    auto level = measureLevel(silence);

    check(level.peak == 0.f);
    check(level.rms == 0.f);

    auto nothing = Span<const float> {};

    check(measureLevel(nothing).peak == 0.f);
    check(measureLevel(nothing).rms == 0.f);
};

auto tMeasureLevelOfFullScaleSquare =
    test("Capture/measureLevelOfFullScaleSquare") = []
{
    auto square = std::array<float, 64> {};

    for (auto i = 0; i < 64; ++i)
        square[(std::size_t) i] = i % 2 == 0 ? 1.f : -1.f;

    auto level = measureLevel(square);

    check(near(level.peak, 1.f));
    check(near(level.rms, 1.f));
};

// Whole periods only: the RMS of a sine is its amplitude over root two exactly
// when the window holds an integer number of them.
auto tMeasureLevelOfSine = test("Capture/measureLevelOfSine") = []
{
    constexpr auto samplesPerPeriod = 100;
    constexpr auto count = samplesPerPeriod * 10;

    auto sine = std::array<float, count> {};

    for (auto i = 0; i < count; ++i)
        sine[(std::size_t) i] = std::sin(2.f * std::numbers::pi_v<float>
                                         * (float) i / (float) samplesPerPeriod);

    auto level = measureLevel(sine);

    check(near(level.peak, 1.f));
    check(near(level.rms, 1.f / std::numbers::sqrt2_v<float>));
};

auto tCaptureStartsIdle = test("Capture/startsIdle") = []
{
    auto capture = Capture {};
    auto samples = Vector<float> {};

    check(!capture.isRunning());
    check(capture.drain(samples) == 0);
    check(samples.empty());

    auto state = capture.status();

    check(!state.running);
    check(state.blocks == 0);
    check(state.dropped == 0);
    check(state.lastError == MakeASound::Error::NoError);

    auto expectedId = capture.deviceManager().getDefaultInputDevice();

    check(capture.deviceId() == (expectedId.hasChannels(true) ? expectedId.id : -1));
    check(capture.config().sampleRate == sampleRate);
};

auto tUnknownDeviceChangesNothing = test("Capture/unknownDeviceChangesNothing") = []
{
    auto capture = Capture {};

    auto before = capture.deviceId();
    auto beforeFirst = capture.firstChannel();
    auto beforeCount = capture.channelCount();

    check(capture.setDevice(-12345) == MakeASound::Error::INVALID_DEVICE);

    check(capture.deviceId() == before);
    check(capture.firstChannel() == beforeFirst);
    check(capture.channelCount() == beforeCount);
    check(capture.config().sampleRate == sampleRate);
    check(!capture.isRunning());
};

auto tInputDevicesOnlyHaveInputs = test("Capture/inputDevicesOnlyHaveInputs") = []
{
    auto capture = Capture {};
    auto inputs = capture.inputDevices();
    auto expected = 0;

    for (const auto& device: capture.deviceManager().getDevices())
        if (device.hasChannels(true))
            ++expected;

    check(inputs.size() == expected);

    for (const auto& device: inputs)
        check(device.hasChannels(true));
};

// Needs a microphone and the permission to open it, so it runs only when it is
// asked for by name.
auto tLiveCapture = test("Capture/liveCapture") = []
{
    if (!liveCaptureRequested())
        return;

    auto capture = Capture {};

    if (capture.inputDevices().empty() || capture.deviceId() < 0)
    {
        std::printf("  no input device - live capture skipped\n");
        return;
    }

    auto started = capture.start();

    if (started != MakeASound::Error::NoError)
    {
        std::printf("  start failed: %s\n",
                    MakeASound::getErrorMessage(started).c_str());
        check(started == MakeASound::Error::NoError);
        return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(400));

    auto samples = Vector<float> {};
    auto drained = capture.drain(samples);
    auto state = capture.status();
    auto meter = capture.level();

    std::printf("  %d Hz, block %d, %d ch, %lld blocks, %d samples, %s,"
                " peak %.4f rms %.4f, %d overflows, %d dropped\n",
                state.sampleRate,
                state.blockSize,
                state.streamChannels,
                state.blocks,
                drained,
                state.native ? "native" : "resampled",
                (double) meter.peak,
                (double) meter.rms,
                state.overflows,
                state.dropped);

    check(state.running);
    check(state.blocks > 0);
    check(state.sampleRate == sampleRate);
    check(state.streamChannels > 0);
    check(drained > 0);
    check(drained == samples.size());

    capture.stop();

    check(!capture.isRunning());
    check(!capture.status().running);
};
