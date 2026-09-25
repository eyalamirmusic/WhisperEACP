#include <WhisperEACP/WhisperEACP.h>

#include <eacp/Core/App/App.h>
#include <eacp/GPU/GPU.h>

#include <MakeASound/Devices/DeviceManager.h>

#include <chrono>
#include <cstdio>
#include <exception>
#include <string_view>

// Bring-up check: the two halves this project stands on — a device to dispatch
// compute on, and a microphone to feed it — reporting what each one gives us to
// plan against, and whether Core ML can take the encoder. Worth a real binary
// rather than knowledge, because every number below is a property of the eacp
// and MakeASound versions that were fetched, not of this project.

using namespace eacp;
using namespace eacp::GPU;

namespace
{
constexpr auto usage =
    "usage: DeviceInfo [--plan]\n"
    "\n"
    "  --plan  load the bundled model with its encoder on Core ML and print\n"
    "          where Core ML placed the encoder's ops. Reading the plan costs\n"
    "          the Neural Engine compile again, about 14 s, and a first load\n"
    "          on a machine another 14 s.\n";

bool readsThePlan = false;
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

#if EACP_HAS_COREML
const char* yesOrNo(bool answer)
{
    return answer ? "yes" : "no";
}

double secondsSince(std::chrono::steady_clock::time_point start)
{
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double>(elapsed).count();
}

void printEncoderPlan()
{
    if (!WSP::Whisper::hasBundledModel())
    {
        std::printf("  plan                      no bundled model: configure "
                    "with -DWHISPER_EACP_FETCH_MODEL=ON\n");
        return;
    }

    if (!Device::shared().isValid()
        || !WSP::Whisper::supportsEncoderBackend(WSP::EncoderBackend::coreML))
    {
        std::printf("  plan                      the encoder cannot run on Core "
                    "ML here\n");
        return;
    }

    try
    {
        auto whisper = WSP::Whisper {};
        whisper.loadBundled();
        whisper.setEncoderBackend(WSP::EncoderBackend::coreML);
        whisper.prepare();

        std::printf("  encoder load              %.2f s, %s\n",
                    whisper.encoderLoadSeconds(),
                    whisper.encoderWasCacheHit() ? "cache hit" : "compiled");

        const auto start = std::chrono::steady_clock::now();
        const auto& plan = whisper.encoderComputePlan();

        std::printf("  plan, cpuAndNeuralEngine  read in %.1f s\n%s",
                    secondsSince(start),
                    WSP::describePlacement(plan, "    ").c_str());
    }
    catch (const std::exception& failure)
    {
        std::printf("  plan                      %s\n", failure.what());
    }
}
#endif

void printCoreMLInfo()
{
    std::printf("\nCore ML\n");

#if EACP_HAS_COREML
    std::printf("  Neural Engine             %s\n",
                yesOrNo(eacp::ML::hasNeuralEngine()));
    std::printf("  Core ML runner            %s\n",
                yesOrNo(eacp::ML::isSupported()));
    std::printf("  fused attention (spec 9)  %s\n",
                yesOrNo(eacp::ML::supportsSpecification(9)));
    std::printf(
        "  encoder on Core ML        %s\n",
        yesOrNo(WSP::Whisper::supportsEncoderBackend(WSP::EncoderBackend::coreML)));

    if (readsThePlan)
        printEncoderPlan();
    else
        std::printf("  plan                      --plan reads it, about 14 s\n");
#else
    std::printf("  not built with Core ML\n");
#endif
}

void printDeviceInfo()
{
    std::printf("WhisperEACP %s\n\n",
                WSP::toString(WSP::getLibraryVersion()).c_str());

    printGpuInfo();
    printAudioInfo();
    printFormat();
    printCoreMLInfo();
}
} // namespace

int main(int argc, char* argv[])
{
    for (auto index = 1; index < argc; ++index)
    {
        if (std::string_view {argv[index]} != "--plan")
        {
            std::printf("%s", usage);
            return 2;
        }

        readsThePlan = true;
    }

    Apps::setCommandLineArgs(argc, argv);
    return Apps::run(printDeviceInfo);
}
