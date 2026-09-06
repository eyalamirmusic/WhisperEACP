#include <WhisperEACP/WhisperEACP.h>

#include <eacp/Core/App/App.h>
#include <eacp/GPU/GPU.h>

#include <MakeASound/Devices/DeviceManager.h>

#include <cstdio>

// Bring-up check: the two halves this project stands on — a device to dispatch
// compute on, and a microphone to feed it — reporting what each one gives us to
// plan against. Worth a real binary rather than knowledge, because every number
// below is a property of the eacp and MakeASound versions that were fetched,
// not of this project.

using namespace eacp;
using namespace eacp::GPU;

namespace
{
void printGpuInfo()
{
    auto& device = Device::shared();

    std::printf("GPU\n");

    if (!device.isValid())
    {
        std::printf("  no device available - nothing here can run\n");
        return;
    }

    std::printf("  1D thread group width     %d\n", ComputePass::threadGroupWidth);
    std::printf("  2D thread group size      %d x %d\n",
                ComputePass::threadGroupSize2D,
                ComputePass::threadGroupSize2D);
    std::printf("  storage buffer slots      %d\n", ComputePass::uniformBase);
    std::printf("  first texture register    %d\n",
                ComputePass::textureRegisterBase);
}

void printAudioInfo()
{
    auto manager = MakeASound::DeviceManager {};

    std::printf("\nAudio input\n");

    for (const auto& device: manager.getDevices())
    {
        if (!device.hasChannels(true))
            continue;

        std::printf("  %-38s %2d in  %5d Hz  %s\n",
                    device.name.c_str(),
                    device.inputChannels,
                    device.preferredSampleRate,
                    WSP::canCaptureNatively(device) ? "16k native"
                                                    : "needs resampling");
    }
}

void printFormat()
{
    std::printf("\nWhisper front-end\n");
    std::printf("  capture                   %d Hz, %d ch\n",
                WSP::sampleRate,
                WSP::channelCount);
    std::printf("  STFT                      %d-point, hop %d\n",
                WSP::fftSize,
                WSP::hopSize);
    std::printf("  encoder window            %d s = %d frames = %d positions\n",
                WSP::windowSeconds,
                WSP::windowFrames,
                WSP::encoderPositions);
}

void printDeviceInfo()
{
    std::printf("WhisperEACP %s\n\n",
                WSP::toString(WSP::getLibraryVersion()).c_str());

    printGpuInfo();
    printAudioInfo();
    printFormat();
}
} // namespace

int main()
{
    return Apps::run(printDeviceInfo);
}
