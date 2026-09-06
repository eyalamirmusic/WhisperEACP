#include <WhisperEACP/Audio/Audio.h>
#include <NanoTest/NanoTest.h>

using namespace nano;
using namespace WSP;

// These constants are the model's, not ours. A change to any of them silently
// misfeeds the encoder rather than failing to build, so they are pinned here by
// value the way a wire format would be.

auto tCaptureFormat = test("Format/captureFormat") = []
{
    check(sampleRate == 16000);
    check(channelCount == 1);
    check(fftSize == 400);
    check(hopSize == 160);
};

auto tEncoderWindow = test("Format/encoderWindow") = []
{
    check(windowSamples == 480000);
    check(windowFrames == 3000);
    check(encoderPositions == 1500);
};

// A frame needs a full window behind it, so the count is not just a division.
auto tFramesForSamples = test("Format/framesForSamples") = []
{
    check(framesForSamples(0) == 0);
    check(framesForSamples(fftSize - 1) == 0);
    check(framesForSamples(fftSize) == 1);
    check(framesForSamples(fftSize + hopSize) == 2);
};

auto tSampleAndFrameConversions = test("Format/sampleAndFrameConversions") = []
{
    check(samplesForSeconds(1.0) == sampleRate);
    check(samplesForSeconds(windowSeconds) == windowSamples);
    check(secondsForFrames(100) == 1.0);
};

// hasChannels(true) first: a machine with speakers and no microphone is an
// ordinary desktop, and it is not a device we can capture from at any rate.
auto tOutputOnlyDeviceCannotCapture =
    test("Format/outputOnlyDeviceCannotCapture") = []
{
    auto device = MakeASound::DeviceInfo {};
    device.outputChannels = 2;
    device.sampleRates = {sampleRate};

    check(!canCaptureNatively(device));
};

auto tNativeCaptureNeedsTheRate = test("Format/nativeCaptureNeedsTheRate") = []
{
    auto device = MakeASound::DeviceInfo {};
    device.inputChannels = 1;

    device.sampleRates = {44100, 48000};
    check(!canCaptureNatively(device));

    device.sampleRates = {sampleRate, 48000};
    check(canCaptureNatively(device));
};
