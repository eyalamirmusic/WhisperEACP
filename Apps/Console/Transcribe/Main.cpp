#include <WhisperEACP/WhisperEACP.h>

#include <eacp/Core/App/App.h>
#include <eacp/GPU/GPU.h>

#include <chrono>
#include <cstdio>
#include <exception>
#include <string>

// The whole runtime as a binary: a model directory and a WAV file in, the
// transcript and what each half of the run cost out.
//
// Inside eacp::Apps::run for the reason every GPU-touching thing here is — the
// Metal backend is written against the run loop and autorelease pool that owns,
// and DeviceInfo next door is the same shape.

using namespace eacp;

namespace
{
constexpr auto usage =
    "usage: Transcribe <model directory> <wav file>\n"
    "\n"
    "  model directory  a HuggingFace Whisper repo: config.json,\n"
    "                   preprocessor_config.json, model.safetensors and\n"
    "                   tokenizer.json\n"
    "  wav file         16 kHz mono, at most 30 seconds\n";

double secondsSince(std::chrono::steady_clock::time_point start)
{
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double>(elapsed).count();
}

void report(const WSP::Whisper& whisper, int tokenCount, double loading)
{
    const auto steps = whisper.lastStepCount();
    const auto perStep = steps > 0 ? whisper.lastDecodeSeconds() / steps : 0.0;

    std::printf("  load and upload       %8.3f s\n", loading);
    std::printf("  mel + encoder         %8.3f s\n", whisper.lastEncodeSeconds());
    std::printf("  decode, %3d tokens    %8.3f s   %5.1f ms per step over %d\n",
                tokenCount,
                whisper.lastDecodeSeconds(),
                1000.0 * perStep,
                steps);
    std::printf("  mel to transcript     %8.3f s\n",
                whisper.lastEncodeSeconds() + whisper.lastDecodeSeconds());
}

void transcribe()
{
    const auto& arguments = Apps::getAppEnvironment().commandLineArgs;

    if (arguments.size() != 3)
    {
        std::printf("%s", usage);
        Apps::setReturnValue(2);
        return;
    }

    if (!GPU::Device::shared().isValid())
    {
        std::printf("no GPU device available - nothing here can run\n");
        Apps::setReturnValue(1);
        return;
    }

    try
    {
        const auto start = std::chrono::steady_clock::now();

        auto whisper = WSP::Whisper {};
        whisper.load(arguments[1]);
        whisper.prepare();

        const auto samples = WSP::readWavFile(arguments[2]);
        const auto loading = secondsSince(start);

        const auto tokens = whisper.transcribe(samples);

        std::printf("%s\n\n", whisper.textForTokens(tokens).c_str());
        report(whisper, tokens.size(), loading);
    }
    catch (const std::exception& failure)
    {
        std::printf("%s\n", failure.what());
        Apps::setReturnValue(1);
    }
}
} // namespace

int main(int argc, char* argv[])
{
    Apps::setCommandLineArgs(argc, argv);
    return Apps::run(transcribe);
}
