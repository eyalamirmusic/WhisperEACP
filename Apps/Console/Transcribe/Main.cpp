#include <WhisperEACP/WhisperEACP.h>

#include <eacp/Core/App/App.h>
#include <eacp/GPU/GPU.h>

#include <ResEmbed/ResEmbed.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <string_view>

// The whole runtime as a binary, in three forms: with no arguments it runs the
// model the build copied beside it on the recording it carries inside itself,
// with one it runs that model on a WAV of yours, and with two it takes a
// HuggingFace directory as well. Either way it prints the transcript and what
// each half of the run cost.
//
// Inside eacp::Apps::run for the reason every GPU-touching thing here is — the
// Metal backend is written against the run loop and autorelease pool that owns,
// and DeviceInfo next door is the same shape.

using namespace eacp;

namespace
{
constexpr auto usage =
    "usage: Transcribe                                  bundled model, "
    "built-in sample\n"
    "       Transcribe <wav file>                       bundled model\n"
    "       Transcribe <model directory> <wav file>\n"
    "\n"
    "  model directory  a HuggingFace Whisper repo: config.json,\n"
    "                   preprocessor_config.json, model.safetensors and\n"
    "                   tokenizer.json\n"
    "  wav file         16 kHz mono, at most 30 seconds\n"
    "  --audio-ctx=N    encode N of the encoder's 1500 positions rather than\n"
    "                   the whole 30 s window, or 'audio' for the count the\n"
    "                   recording itself fills\n"
    "\n"
    "The built-in sample is 11 s of Kennedy's inaugural address. The bundled\n"
    "model is whisper-tiny.en, copied beside this binary by a build configured\n"
    "with -DWHISPER_EACP_FETCH_MODEL=ON.\n";

constexpr auto missingBundledModel =
    "this build copied no model beside the binary, so there is nothing to run\n"
    "without a model directory. Configure with -DWHISPER_EACP_FETCH_MODEL=ON to\n"
    "get one, or name a HuggingFace Whisper repo on the command line.\n\n";

constexpr auto embeddedSample = "jfk.wav";
constexpr auto embeddedSampleCategory = "TranscribeSamples";

double secondsSince(std::chrono::steady_clock::time_point start)
{
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double>(elapsed).count();
}

WSP::Vector<float> readEmbeddedSample()
{
    const auto wav = ResEmbed::get(embeddedSample, embeddedSampleCategory);
    const auto bytes = WSP::Span<const std::uint8_t> {wav.data(), wav.getSize()};

    return WSP::readWavBytes(bytes, embeddedSample);
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

// Which of the three forms the command line asked for, resolved before the
// device is opened so a build with nothing embedded says so rather than
// reporting a missing GPU first.
struct Request
{
    std::string modelDirectory;
    std::string wavFile;

    // 0 is the whole window and -1 is the count the samples fill; anything
    // else is that many encoder positions.
    int audioContext = 0;

    static constexpr int fromTheAudio = -1;

    bool usesBundledModel() const { return modelDirectory.empty(); }
    bool usesEmbeddedSample() const { return wavFile.empty(); }

    std::string modelPath() const
    {
        return usesBundledModel() ? WSP::Whisper::bundledModelDirectory().string()
                                  : modelDirectory;
    }

    void announce() const
    {
        std::printf("model: %s\n", modelPath().c_str());
        std::printf("audio: %s\n",
                    usesEmbeddedSample() ? "built-in jfk.wav" : wavFile.c_str());

        if (audioContext == fromTheAudio)
            std::printf("context: the positions the recording fills\n");
        else if (audioContext > 0)
            std::printf("context: %d of 1500 encoder positions\n", audioContext);

        std::printf("\n");
    }
};

// Each argument is the audio context if it is that flag, and a path otherwise,
// in the order the two forms above take them.
bool parse(const WSP::Vector<std::string>& arguments, Request& request)
{
    constexpr auto flag = std::string_view {"--audio-ctx="};
    auto paths = WSP::Vector<std::string> {};

    for (auto index = 1; index < arguments.size(); ++index)
    {
        const auto& argument = arguments[index];

        if (!std::string_view {argument}.starts_with(flag))
        {
            paths.add(argument);
            continue;
        }

        const auto value = argument.substr(flag.size());

        if (value == "audio")
        {
            request.audioContext = Request::fromTheAudio;
            continue;
        }

        request.audioContext = std::atoi(value.c_str());

        if (request.audioContext < 1 || request.audioContext > WSP::encoderPositions)
            return false;
    }

    if (paths.size() > 2)
        return false;

    if (paths.size() == 2)
    {
        request.modelDirectory = paths[0];
        request.wavFile = paths[1];
    }
    else if (paths.size() == 1)
    {
        request.wavFile = paths[0];
    }

    return true;
}

void run(const Request& request)
{
    const auto start = std::chrono::steady_clock::now();

    auto whisper = WSP::Whisper {};

    if (request.usesBundledModel())
        whisper.loadBundled();
    else
        whisper.load(request.modelDirectory);

    whisper.prepare();

    const auto samples = request.usesEmbeddedSample()
                             ? readEmbeddedSample()
                             : WSP::readWavFile(request.wavFile);

    if (request.audioContext == Request::fromTheAudio)
        whisper.setAudioContext(
            WSP::Whisper::audioContextForSamples(samples.size()));
    else
        whisper.setAudioContext(request.audioContext);

    const auto loading = secondsSince(start);
    const auto tokens = whisper.transcribe(samples);

    std::printf("%s\n\n", whisper.textForTokens(tokens).c_str());
    report(whisper, tokens.size(), loading);
}

void transcribe()
{
    const auto& arguments = Apps::getAppEnvironment().commandLineArgs;
    auto request = Request {};

    if (!parse(arguments, request))
    {
        std::printf("%s", usage);
        Apps::setReturnValue(2);
        return;
    }

    if (request.usesBundledModel() && !WSP::Whisper::hasBundledModel())
    {
        std::printf("%s%s", missingBundledModel, usage);
        Apps::setReturnValue(2);
        return;
    }

    if (!GPU::Device::shared().isValid())
    {
        std::printf("no GPU device available - nothing here can run\n");
        Apps::setReturnValue(1);
        return;
    }

    request.announce();

    try
    {
        run(request);
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
