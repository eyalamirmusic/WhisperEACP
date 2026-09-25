#include <WhisperEACP/WhisperEACP.h>

#include <eacp/Core/App/App.h>
#include <eacp/GPU/GPU.h>

#include <ggml-backend.h>
#include <whisper.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// Our runtime against whisper.cpp, in one process, on the same samples: the
// whole transcription timed by wall clock on every side, each side's own stage
// breakdown under it, over one warm-up and a set of timed runs of which the
// median is what gets compared.
//
// whisper.cpp is built the way a user of it would build it on this machine —
// Metal, Accelerate and BLAS on, on a Mac; the root CMakeLists makes that
// choice whenever this target is in the tree — and runs twice, once on its GPU
// backend and once without it. Both sides load the tiny.en weights: ours from
// the HuggingFace safetensors the build copied beside this binary, F32 on disk
// and its projections packed back to the fp16 values that container holds;
// theirs from the GGML conversion of the same, F16.
//
// `--live` is the other thing this measures, and it has no second side: the
// recording streamed through `LiveTranscriber` at the pace a microphone
// delivers it, so the number that comes out is what the runtime costs a machine
// that is listening rather than what one transcription costs. The live section
// of README.md says why the two are not comparable.
//
// Inside eacp::Apps::run for the reason every GPU-touching thing here is — the
// Metal backend is written against the run loop and autorelease pool that
// owns, and Transcribe is the same shape.

using namespace eacp;

namespace
{
constexpr auto buildType = WHISPER_EACP_BUILD_TYPE;

constexpr auto usage =
    "usage: Benchmark [runs] [wav file] [--audio-ctx=positions|audio]\n"
    "                 [--units=all|cpuAndNeuralEngine|cpuAndGPU|cpu] [--plan]\n"
    "       Benchmark --live [seconds] [gap seconds] [wav file]\n"
    "                 [--audio-ctx=audio] [--units=...]\n"
    "\n"
    "  runs        timed runs per contestant after one warm-up, 10 by default\n"
    "  wav file    16 kHz mono, at most 30 seconds; Samples/jfk.wav by default\n"
    "  --audio-ctx encoder positions both sides run over, out of 1500. Ours is\n"
    "              Whisper::setAudioContext and theirs is whisper_full_params\n"
    "              audio_ctx, which are the same knob; the whole window by\n"
    "              default. \"audio\" sizes it to the recording the way the\n"
    "              live loop does, and in --live turns that option on\n"
    "  --units     where Core ML may place the encoder for the WhisperEACP ANE\n"
    "              column, which runs wherever this build and OS have Core ML:\n"
    "              cpuAndNeuralEngine by default, the only setting that reaches\n"
    "              the engine; under all Core ML puts the encoder on the GPU\n"
    "  --plan      print where Core ML placed the encoder's ops. Reading the\n"
    "              plan costs the Neural Engine compile again, about 14 s\n"
    "\n"
    "  --live      our runtime alone, through LiveTranscriber, streamed at real\n"
    "              time: the recording on repeat with a gap of silence between\n"
    "              passes, once with the encoder on the kernels and once on\n"
    "              Core ML. The numbers, in order, are how long to stream for -\n"
    "              30 s by default - and the gap, 1.5 s. The recording may be\n"
    "              any length here, since the segments are the policy's\n";

constexpr auto missingBundledModel =
    "this build copied no model beside the binary. Configure with\n"
    "-DWHISPER_EACP_FETCH_MODEL=ON to get one.\n";

constexpr auto defaultRuns = 10;
constexpr auto warmUpRuns = 1;
constexpr auto defaultSample = WHISPER_EACP_SAMPLE_DIR "/jfk.wav";
constexpr auto ggmlModel = WHISPER_EACP_GGML_MODEL;

// The stream the live mode plays, and the timer it plays it from: 30 Hz is
// what Apps/Demo/LiveTranscribe ticks at, and every clock in the measurement
// hangs off that tick rather than off how fast the machine could go.
constexpr auto defaultLiveSeconds = 30.0;
constexpr auto defaultLiveGapSeconds = 1.5;
constexpr auto liveTickMilliseconds = 33;

using Clock = std::chrono::steady_clock;

double secondsSince(Clock::time_point start)
{
    const auto elapsed = Clock::now() - start;
    return std::chrono::duration<double>(elapsed).count();
}

// One transcription, as the side that ran it reports it.
//
// `encodeSeconds` is what each side calls its encoder: the mel front-end
// included on ours, since the upload, the mel and the encoder are one command
// buffer, and excluded on whisper.cpp's, whose timings do not expose its mel.
// `decodeSeconds` is the prompt pass and every token after it, and `steps`
// counts them the same way on both sides — one for the prompt, one per sampled
// token, the last of which is <|endoftext|>.
//
// With the encoder on Core ML, `encodeSeconds` is the mel's own command buffer,
// the prediction, and the seam either side of it, and the first two are also
// reported apart: `hasEncoderSplit` says they were.
struct Run
{
    double wallSeconds = 0.0;
    double encodeSeconds = 0.0;
    double decodeSeconds = 0.0;
    double melSeconds = 0.0;
    double predictSeconds = 0.0;
    bool hasEncoderSplit = false;
    int steps = 0;
    WSP::Vector<int> tokens;
};

class Contestant
{
public:
    virtual ~Contestant() = default;

    virtual std::string name() const = 0;
    virtual std::string column() const = 0;
    virtual Run transcribe(WSP::Span<const float> samples) = 0;
    virtual std::string text(const WSP::Vector<int>& tokens) const = 0;
};

const char* nameOf(WSP::EncoderComputeUnits units)
{
    switch (units)
    {
        case WSP::EncoderComputeUnits::all:
            return "all";
        case WSP::EncoderComputeUnits::cpuAndNeuralEngine:
            return "cpuAndNeuralEngine";
        case WSP::EncoderComputeUnits::cpuAndGPU:
            return "cpuAndGPU";
        case WSP::EncoderComputeUnits::cpu:
            return "cpu";
    }

    return "?";
}

bool parseUnits(std::string_view text, WSP::EncoderComputeUnits& units)
{
    constexpr auto every = std::array {WSP::EncoderComputeUnits::all,
                                       WSP::EncoderComputeUnits::cpuAndNeuralEngine,
                                       WSP::EncoderComputeUnits::cpuAndGPU,
                                       WSP::EncoderComputeUnits::cpu};

    for (auto candidate: every)
    {
        if (text == nameOf(candidate))
        {
            units = candidate;
            return true;
        }
    }

    return false;
}

bool canRunTheEncoderOnCoreML()
{
    return WSP::Whisper::supportsEncoderBackend(WSP::EncoderBackend::coreML);
}

// The kernel column prepares the encoder on the GPU; the ANE column records,
// compiles (or finds in the cache) and loads the Core ML one, with the decoder
// on the GPU either way.
class OurRuntime final : public Contestant
{
public:
    OurRuntime(int audioContext,
               WSP::EncoderBackend backendToUse = WSP::EncoderBackend::kernels,
               WSP::EncoderComputeUnits unitsToUse =
                   WSP::EncoderComputeUnits::cpuAndNeuralEngine)
    {
        whisper.loadBundled();
        whisper.setEncoderBackend(backendToUse);
        whisper.setEncoderComputeUnits(unitsToUse);

        const auto start = Clock::now();
        whisper.prepare();
        prepareSeconds = secondsSince(start);

        whisper.setAudioContext(audioContext);
    }

    std::string name() const override
    {
        const auto gpu = GPU::Device::shared().name();

        if (!usesCoreML())
            return "WhisperEACP on " + gpu;

        return std::string {"WhisperEACP, encoder on Core ML ("}
               + nameOf(whisper.encoderComputeUnits()) + "), decoder on " + gpu;
    }

    std::string column() const override
    {
        return usesCoreML() ? "WhisperEACP ANE" : "WhisperEACP";
    }

    Run transcribe(WSP::Span<const float> samples) override
    {
        const auto start = Clock::now();

        auto run = Run {};
        run.tokens = whisper.transcribe(samples);
        run.wallSeconds = secondsSince(start);
        run.encodeSeconds = whisper.lastEncodeSeconds();
        run.decodeSeconds = whisper.lastDecodeSeconds();
        run.melSeconds = whisper.lastEncoderMelSeconds();
        run.predictSeconds = whisper.lastEncoderPredictSeconds();
        run.hasEncoderSplit = usesCoreML();
        run.steps = whisper.lastStepCount();

        return run;
    }

    std::string text(const WSP::Vector<int>& tokens) const override
    {
        return whisper.textForTokens(tokens);
    }

    bool packsWeights() const { return whisper.packsWeights(); }
    bool usesCoreML() const
    {
        return whisper.encoderBackend() == WSP::EncoderBackend::coreML;
    }

    WSP::Whisper& runtime() { return whisper; }
    double lastPrepareSeconds() const { return prepareSeconds; }

private:
    WSP::Whisper whisper;
    double prepareSeconds = 0.0;
};

// What the storage line says after "F32": the file on disk is F32 either way,
// and this is whether the projections reached the device narrowed to the fp16
// values that container holds. Read off the runtime that ran rather than
// assumed.
const char* packedProjections(bool packed)
{
    return packed ? ", projections packed fp16" : "";
}

struct ContextDeleter
{
    void operator()(whisper_context* context) const { whisper_free(context); }
};

using Context = std::unique_ptr<whisper_context, ContextDeleter>;

// Greedy, one segment, no timestamps, temperature 0 with no fallback, and the
// non-speech suppression whisper.cpp leaves off by default turned on: the
// policy our search runs, spelled the way Tests/Oracle spells it, so both
// sides do the same work and the transcripts compare token for token.
whisper_full_params greedyParameters(int threads, int audioContext)
{
    auto parameters = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    parameters.n_threads = threads;
    parameters.audio_ctx = audioContext;
    parameters.language = "en";
    parameters.translate = false;
    parameters.no_timestamps = true;
    parameters.single_segment = true;
    parameters.suppress_blank = true;
    parameters.suppress_nst = true;
    parameters.temperature = 0.0f;
    parameters.temperature_inc = 0.0f;
    parameters.print_progress = false;
    parameters.print_realtime = false;
    parameters.print_special = false;
    parameters.print_timestamps = false;

    return parameters;
}

int whisperCppDefaultThreads()
{
    return whisper_full_default_params(WHISPER_SAMPLING_GREEDY).n_threads;
}

std::string describe(ggml_backend_dev_t device)
{
    return std::string {ggml_backend_reg_name(ggml_backend_dev_backend_reg(device))}
           + ", " + ggml_backend_dev_description(device);
}

class WhisperCpp final : public Contestant
{
public:
    WhisperCpp(bool useGpu, std::string backend, int positions)
        : backendName(std::move(backend))
        , columnName(useGpu ? "whisper.cpp GPU" : "whisper.cpp CPU")
        , threads(whisperCppDefaultThreads())
        , audioContext(positions)
    {
        auto parameters = whisper_context_default_params();
        parameters.use_gpu = useGpu;

        context.reset(whisper_init_from_file_with_params(ggmlModel, parameters));

        if (context == nullptr)
            throw std::runtime_error {std::string {"whisper.cpp could not load "}
                                      + ggmlModel};
    }

    std::string name() const override
    {
        return "whisper.cpp on " + backendName + ", " + std::to_string(threads)
               + " threads";
    }

    std::string column() const override { return columnName; }

    Run transcribe(WSP::Span<const float> samples) override
    {
        whisper_reset_timings(context.get());

        const auto start = Clock::now();

        if (whisper_full(context.get(),
                         greedyParameters(threads, audioContext),
                         samples.data(),
                         samples.size())
            != 0)
            throw std::runtime_error {"whisper_full failed"};

        auto run = Run {};
        run.wallSeconds = secondsSince(start);
        run.tokens = transcriptTokens();

        // whisper_get_timings averages each counter over its calls without
        // saying how many there were, so the totals are put back from the
        // shape of a temperature-0 greedy run, which is exact: one encode, one
        // prompt pass, then one single-token decode per text token, the last
        // of them the one that sampled <|endoftext|>.
        const auto timings =
            std::unique_ptr<whisper_timings> {whisper_get_timings(context.get())};
        const auto decodes = run.tokens.size();

        run.encodeSeconds = timings->encode_ms / 1000.0;
        run.decodeSeconds =
            (timings->prompt_ms + timings->decode_ms * decodes) / 1000.0;
        run.steps = decodes + 1;

        return run;
    }

    std::string text(const WSP::Vector<int>& tokens) const override
    {
        auto joined = std::string {};

        for (auto token: tokens)
            joined += whisper_token_to_str(context.get(), token);

        return joined;
    }

private:
    // Every text token of every segment, in order, the special tokens
    // dropped: whisper.cpp keeps <|endoftext|> as a segment's last token where
    // our loop stops on it, and that is a difference about where the marker
    // is stored rather than about what was decoded.
    WSP::Vector<int> transcriptTokens() const
    {
        const auto endOfText = whisper_token_eot(context.get());
        auto tokens = WSP::Vector<int> {};

        for (auto segment = 0; segment < whisper_full_n_segments(context.get());
             ++segment)
            for (auto index = 0;
                 index < whisper_full_n_tokens(context.get(), segment);
                 ++index)
            {
                const auto token =
                    whisper_full_get_token_id(context.get(), segment, index);

                if (token < endOfText)
                    tokens.add(token);
            }

        return tokens;
    }

    Context context;
    std::string backendName;
    std::string columnName;
    int threads = 0;
    int audioContext = 0;
};

// The medians of every timed run of one contestant, with the transcript the
// warm-up produced so the sides can be checked against each other.
struct Summary
{
    std::string name;
    std::string column;
    double wallMedian = 0.0;
    double wallBest = 0.0;
    double encodeMedian = 0.0;
    double decodeMedian = 0.0;
    double melMedian = 0.0;
    double predictMedian = 0.0;
    double seamMedian = 0.0;
    bool hasEncoderSplit = false;
    int steps = 0;
    WSP::Vector<int> tokens;
    std::string text;
};

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());

    const auto middle = values.size() / 2;

    if (values.size() % 2 == 1)
        return values[middle];

    return 0.5 * (values[middle - 1] + values[middle]);
}

Summary benchmark(Contestant& contestant, WSP::Span<const float> samples, int runs)
{
    auto summary = Summary {};
    summary.name = contestant.name();
    summary.column = contestant.column();

    for (auto index = 0; index < warmUpRuns; ++index)
    {
        const auto warmUp = contestant.transcribe(samples);
        summary.tokens = warmUp.tokens;
        summary.text = contestant.text(warmUp.tokens);
    }

    auto wall = std::vector<double> {};
    auto encode = std::vector<double> {};
    auto decode = std::vector<double> {};
    auto mel = std::vector<double> {};
    auto predict = std::vector<double> {};
    auto seam = std::vector<double> {};

    for (auto index = 0; index < runs; ++index)
    {
        const auto run = contestant.transcribe(samples);
        wall.push_back(run.wallSeconds);
        encode.push_back(run.encodeSeconds);
        decode.push_back(run.decodeSeconds);
        mel.push_back(run.melSeconds);
        predict.push_back(run.predictSeconds);
        seam.push_back(run.encodeSeconds - run.melSeconds - run.predictSeconds);
        summary.steps = run.steps;
        summary.hasEncoderSplit = run.hasEncoderSplit;
    }

    summary.wallMedian = median(wall);
    summary.wallBest = *std::min_element(wall.begin(), wall.end());
    summary.encodeMedian = median(encode);
    summary.decodeMedian = median(decode);
    summary.melMedian = median(mel);
    summary.predictMedian = median(predict);
    summary.seamMedian = median(seam);

    return summary;
}

bool sameTokens(const WSP::Vector<int>& a, const WSP::Vector<int>& b)
{
    if (a.size() != b.size())
        return false;

    for (auto index = 0; index < a.size(); ++index)
        if (a[index] != b[index])
            return false;

    return true;
}

// The table: one column per contestant, the rows below. Every cell is
// formatted into a string first so a row is one printf whatever it holds.
constexpr auto labelWidth = 28;
constexpr auto columnWidth = 19;

std::string cell(double value, const char* unit, int decimals)
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.*f", decimals, value);

    auto text = std::string {buffer};

    if (*unit != '\0')
        text += std::string {" "} + unit;

    return text;
}

// A change worth keeping is tens of microseconds on a step and a few hundred
// on an encode, so the times print at a resolution that can show one.
std::string milliseconds(double seconds)
{
    return cell(1000.0 * seconds, "ms", 2);
}

std::string microseconds(double seconds)
{
    return cell(1e6 * seconds, "us", 0);
}

void printRow(const std::string& label, const std::vector<std::string>& cells)
{
    std::printf("  %-*s", labelWidth, label.c_str());

    for (const auto& value: cells)
        std::printf("%*s", columnWidth, value.c_str());

    std::printf("\n");
}

template <typename Cell>
void printRow(const std::string& label,
              const std::vector<Summary>& summaries,
              Cell cellFor)
{
    auto cells = std::vector<std::string> {};

    for (const auto& summary: summaries)
        cells.push_back(cellFor(summary));

    printRow(label, cells);
}

void printTable(const std::vector<Summary>& summaries)
{
    const auto& reference = summaries.front();

    printRow("", summaries, [](const Summary& s) { return s.column; });

    printRow("transcribe, median",
             summaries,
             [](const Summary& s) { return milliseconds(s.wallMedian); });

    printRow("transcribe, best",
             summaries,
             [](const Summary& s) { return milliseconds(s.wallBest); });

    printRow("x real time, 30 s window",
             summaries,
             [](const Summary& s)
             { return cell(WSP::windowSeconds / s.wallMedian, "", 1); });

    printRow("encode, median",
             summaries,
             [](const Summary& s) { return milliseconds(s.encodeMedian); });

    const auto splitCell = [](double Summary::* field)
    {
        return [field](const Summary& s)
        { return s.hasEncoderSplit ? milliseconds(s.*field) : std::string {"-"}; };
    };

    printRow("  mel on the GPU, median", summaries, splitCell(&Summary::melMedian));
    printRow("  predict, median", summaries, splitCell(&Summary::predictMedian));
    printRow("  seam copies, median", summaries, splitCell(&Summary::seamMedian));

    printRow("decode, median",
             summaries,
             [](const Summary& s) { return milliseconds(s.decodeMedian); });

    printRow("decode per step, median",
             summaries,
             [](const Summary& s)
             { return microseconds(s.decodeMedian / std::max(1, s.steps)); });

    printRow("steps",
             summaries,
             [](const Summary& s) { return std::to_string(s.steps); });

    printRow("tokens",
             summaries,
             [](const Summary& s) { return std::to_string(s.tokens.size()); });

    printRow("transcript",
             summaries,
             [&reference](const Summary& s)
             {
                 if (&s == &reference)
                     return std::string {"reference"};

                 return std::string {
                     sameTokens(s.tokens, reference.tokens) ? "same" : "DIFFERS"};
             });

    std::printf("\n  encode is the mel and the encoder for WhisperEACP, one "
                "command buffer, and the\n  encoder alone for whisper.cpp, "
                "whose timings do not expose its mel.\n");
    std::printf("  For WhisperEACP ANE it is the mel's own command buffer, "
                "committed and waited on,\n  the mel read back and narrowed to "
                "fp16, the Core ML prediction, and the rows\n  widened back "
                "into the decoder's buffer. The three rows under it split it: "
                "the mel\n  is the only GPU work in the encode, so the GPU is "
                "idle through the predict\n  wherever the plan (--plan) puts "
                "the encoder on the engine; the seam copies\n  are the encode "
                "less the other two, per run.\n");
}

void printBackends()
{
    std::printf("  whisper.cpp %s: flash attention %s, %d threads, backends",
                whisper_version(),
                whisper_context_default_params().flash_attn ? "on" : "off",
                whisperCppDefaultThreads());

    for (auto index = std::size_t {}; index < ggml_backend_dev_count(); ++index)
        std::printf("%s %s",
                    index == 0 ? "" : ",",
                    describe(ggml_backend_dev_get(index)).c_str());

    std::printf("\n");
}

// The configuration this binary was built in, next to the numbers it prints:
// the tree's usual configure is Debug, and a Debug benchmark should say so
// rather than pass for a measurement.
void printBuildType()
{
    const auto configuration = std::string_view {buildType};
    const auto optimised =
        configuration.starts_with("Rel") || configuration == "MinSizeRel";

    std::printf("  build %s%s\n",
                buildType,
                optimised ? ""
                          : " - a Debug tree's numbers; configure a Release "
                            "tree for the real ones");
}

void printPlan(OurRuntime& contestant)
{
#if EACP_HAS_COREML
    const auto start = Clock::now();
    const auto& plan = contestant.runtime().encoderComputePlan();

    std::printf("  plan, read in %.1f s\n%s",
                secondsSince(start),
                WSP::describePlacement(plan, "    ").c_str());
#else
    (void) contestant;
#endif
}

struct Request
{
    bool live = false;
    int runs = defaultRuns;
    double liveSeconds = defaultLiveSeconds;
    double gapSeconds = defaultLiveGapSeconds;
    std::string wavFile = defaultSample;

    // Encoder positions on both sides; 0 is the whole window, and
    // fromTheAudio is Whisper::audioContextForSamples over the recording.
    static constexpr auto fromTheAudio = -1;
    int audioContext = 0;

    // The Core ML contestant's compute units, and whether to read its plan.
    WSP::EncoderComputeUnits units = WSP::EncoderComputeUnits::cpuAndNeuralEngine;
    bool readsPlan = false;
};

// What the ANE column ran on: the units it was compiled for, whether the load
// found the compile in the cache, and how long the load and the whole prepare
// took; and the plan only when asked for, since reading it is another engine
// compile.
void printCoreMLContestant(OurRuntime* contestant, const Request& request)
{
    if (contestant == nullptr)
    {
        std::printf("  WhisperEACP ANE not run: this build or this OS cannot run "
                    "the encoder on Core ML\n");
        return;
    }

    auto& whisper = contestant->runtime();

    std::printf("  WhisperEACP ANE: Core ML %s, encoder load %.2f s (%s), "
                "whole prepare %.2f s\n",
                nameOf(whisper.encoderComputeUnits()),
                whisper.encoderLoadSeconds(),
                whisper.encoderWasCacheHit() ? "cache hit" : "compiled",
                contestant->lastPrepareSeconds());

    if (request.readsPlan)
        printPlan(*contestant);
    else
        std::printf("  plan not read: --plan reads it, about 14 s on the "
                    "engine\n");
}

bool parseNumber(const std::string& text, double& value)
{
    auto* end = (char*) nullptr;
    value = std::strtod(text.c_str(), &end);

    return end == text.c_str() + text.size() && !text.empty()
           && std::isfinite(value);
}

// An argument is a number or the WAV file, and which is which is a question the
// argument answers about itself, so the two read naturally in either order. The
// numbers are what the mode is about: the timed run count for the comparison,
// and the length of the stream and then the gap for --live.
bool parse(const WSP::Vector<std::string>& arguments, Request& request)
{
    auto numbers = std::vector<double> {};

    constexpr auto audioContextFlag = std::string_view {"--audio-ctx="};
    constexpr auto unitsFlag = std::string_view {"--units="};

    for (auto index = 1; index < arguments.size(); ++index)
    {
        const auto& argument = arguments[index];
        auto value = 0.0;

        if (std::string_view {argument}.starts_with(unitsFlag))
        {
            if (!parseUnits(std::string_view {argument}.substr(unitsFlag.size()),
                            request.units))
                return false;

            continue;
        }

        if (argument == "--plan")
        {
            request.readsPlan = true;
            continue;
        }

        if (std::string_view {argument}.starts_with(audioContextFlag))
        {
            const auto positions = argument.substr(audioContextFlag.size());
            auto context = 0;

            if (positions == "audio")
            {
                request.audioContext = Request::fromTheAudio;
                continue;
            }

            if (std::from_chars(
                    positions.data(), positions.data() + positions.size(), context)
                        .ec
                    != std::errc {}
                || context < 1 || context > WSP::encoderPositions)
                return false;

            request.audioContext = context;
            continue;
        }

        if (argument == "--live")
            request.live = true;
        else if (parseNumber(argument, value))
            numbers.push_back(value);
        else
            request.wavFile = argument;
    }

    if (numbers.size() > (request.live ? 2u : 1u))
        return false;

    for (auto number: numbers)
        if (number < 0.0
            || (number == 0.0 && (!request.live || number == numbers[0])))
            return false;

    if (!request.live)
    {
        if (!numbers.empty())
            request.runs = (int) numbers[0];

        return numbers.empty() || numbers[0] == (double) request.runs;
    }

    if (!numbers.empty())
        request.liveSeconds = numbers[0];

    if (numbers.size() > 1)
        request.gapSeconds = numbers[1];

    return true;
}

// The recording, zero-filled to the 30 s window both sides encode. That is
// what our runtime does with a short utterance anyway, and handing whisper.cpp
// the same 480000 samples keeps the two front-ends' tail padding like for like
// — whisper.cpp pads the tail with zeros where HF reflects, so an unpadded
// input would reach the two encoders as two different mels.
WSP::Vector<float> readWindow(const std::string& path, double& audioSeconds)
{
    auto samples = WSP::readWavFile(path);

    if (samples.size() > WSP::windowSamples)
        throw std::runtime_error {path + " is longer than the 30 s window"};

    audioSeconds = (double) samples.size() / WSP::sampleRate;
    samples.resize(WSP::windowSamples, 0.0f);

    return samples;
}

void silenceWhisperCpp()
{
    whisper_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
}

// The live mode: the recording on repeat, at the pace a microphone delivers
// it, through the policy the demo app runs.
//
// What comes out is a duty cycle — model seconds per wall second — rather than
// the cost of one transcription, because that is the number a machine that is
// listening pays. Nothing else can report it: macOS's own GPU utilisation
// counter is a short-window snapshot, so a 26 ms burst every 500 ms reads
// there as most of a core when the true share is a twentieth of one.
struct LiveSummary
{
    std::string column;
    double wallSeconds = 0.0;
    int runs = 0;
    int runsThatChangedTheText = 0;
    double modelSeconds = 0.0;
    double longestRunSeconds = 0.0;
    double encodeSeconds = 0.0;
    double decodeSeconds = 0.0;
    int steps = 0;
    int segments = 0;
    std::string firstLine;
};

// The recording followed by the gap, which the tick walks in a circle: a
// microphone left open over somebody who says the same thing again after a
// pause, which is the shape that exercises every branch of the policy — the
// step, the silence hold that closes a segment, and the silence before the
// next one that must reach the model no times at all.
WSP::Vector<float> withGap(const WSP::Vector<float>& recording, double gapSeconds)
{
    auto stream = recording;

    for (auto index = 0; index < WSP::samplesForSeconds(gapSeconds); ++index)
        stream.add(0.0f);

    return stream;
}

// Gives the event loop the thread until the deadline, which is where a run on
// Core ML comes back; on the kernels there is nothing in it to run.
void pumpUntil(Clock::time_point deadline)
{
    while (Clock::now() < deadline)
    {
        const auto remaining =
            std::chrono::ceil<std::chrono::milliseconds>(deadline - Clock::now());

        Threads::runEventLoopFor(
            Time::MS {std::max<std::int64_t>(1, remaining.count())});
    }
}

LiveSummary streamThroughTheTranscriber(WSP::Whisper& whisper,
                                        const WSP::LiveOptions& options,
                                        const WSP::Vector<float>& stream,
                                        double seconds)
{
    auto live = WSP::LiveTranscriber {whisper, options};
    auto summary = LiveSummary {};
    auto chunk = WSP::Vector<float> {};
    auto cursor = 0;
    auto pushed = (long long) 0;
    auto runsSeen = 0;

    // A run is counted when its result is in, which on the kernels is inside
    // update() and on Core ML inside a pump, and before the next update()
    // starts another run over the runtime's clocks.
    const auto countFinishedRun = [&live, &whisper, &summary, &runsSeen]
    {
        if (live.stats().runs == runsSeen)
            return;

        runsSeen = live.stats().runs;

        const auto runSeconds = live.stats().lastRunSeconds;

        ++summary.runs;
        summary.modelSeconds += runSeconds;
        summary.longestRunSeconds = std::max(summary.longestRunSeconds, runSeconds);
        summary.encodeSeconds += whisper.lastEncodeSeconds();
        summary.decodeSeconds += whisper.lastDecodeSeconds();
        summary.steps += whisper.lastStepCount();
    };

    const auto start = Clock::now();

    for (auto tick = 1; secondsSince(start) < seconds; ++tick)
    {
        pumpUntil(start + std::chrono::milliseconds(liveTickMilliseconds * tick));
        countFinishedRun();

        // Whatever the elapsed audio time asks for, which is a run that
        // overran its tick handing the next one a larger block — exactly what
        // a capture queue does while the model has the thread.
        const auto due = (long long) (secondsSince(start) * WSP::sampleRate);

        chunk.clear();

        while (pushed < due)
        {
            chunk.add(stream[cursor]);
            cursor = (cursor + 1) % stream.size();
            ++pushed;
        }

        live.push(chunk);
        live.update();
        countFinishedRun();
    }

    summary.wallSeconds = secondsSince(start);

    const auto isIdle = [&live] { return !live.isRunning(); };
    Threads::runEventLoopUntil(isIdle, Time::MS {10000}, Time::MS {1});
    countFinishedRun();

    summary.runsThatChangedTheText = live.stats().runsThatChangedTheText;
    summary.segments = live.committed().size();

    if (summary.segments > 0)
        summary.firstLine = live.committed()[0];

    return summary;
}

void printLiveTable(const std::vector<LiveSummary>& summaries)
{
    const auto row = [&summaries](const std::string& label, auto cellFor)
    {
        auto cells = std::vector<std::string> {};

        for (const auto& summary: summaries)
            cells.push_back(cellFor(summary));

        printRow(label, cells);
    };

    const auto perRun = [](const LiveSummary& s, double total)
    { return cell(1000.0 * total / std::max(1, s.runs), "ms", 1); };

    const auto column = [](const LiveSummary& s) { return s.column; };
    const auto wall = [](const LiveSummary& s)
    { return cell(s.wallSeconds, "s", 1); };
    const auto runs = [](const LiveSummary& s) { return std::to_string(s.runs); };
    const auto changed = [](const LiveSummary& s)
    { return std::to_string(s.runsThatChangedTheText); };
    const auto model = [](const LiveSummary& s)
    { return cell(s.modelSeconds, "s", 3); };
    const auto duty = [](const LiveSummary& s)
    { return cell(100.0 * s.modelSeconds / s.wallSeconds, "%", 1); };
    const auto meanRun = [&perRun](const LiveSummary& s)
    { return perRun(s, s.modelSeconds); };
    const auto longestRun = [](const LiveSummary& s)
    { return cell(1000.0 * s.longestRunSeconds, "ms", 1); };
    const auto meanEncode = [&perRun](const LiveSummary& s)
    { return perRun(s, s.encodeSeconds); };
    const auto meanDecode = [&perRun](const LiveSummary& s)
    { return perRun(s, s.decodeSeconds); };
    const auto steps = [](const LiveSummary& s)
    { return cell((double) s.steps / std::max(1, s.runs), "", 1); };
    const auto segments = [](const LiveSummary& s)
    { return std::to_string(s.segments); };

    row("", column);
    row("stream, wall clock", wall);
    row("runs", runs);
    row("runs that changed the text", changed);
    row("model", model);
    row("duty", duty);
    row("per run, mean", meanRun);
    row("per run, longest", longestRun);
    row("encode, mean", meanEncode);
    row("decode, mean", meanDecode);
    row("steps per run", steps);
    row("segments committed", segments);

    std::printf("\n  a run at this cadence costs about twice what the "
                "comparison above measures:\n  the GPU clocks down between "
                "bursts half a second apart.\n");
    std::printf("  On Core ML a run is its start to its result, the loop "
                "turns the main thread\n  spent elsewhere while the engine "
                "worked included, and its encode ends when the\n  prediction "
                "comes back to the loop; so model and duty there are the "
                "run's\n  latency, not the main thread's or the GPU's "
                "share.\n");
}

void printLivePolicy(const WSP::LiveOptions& options)
{
    std::printf("  policy step %.2f s, new speech %.2f s, silence hold %.2f s, "
                "segment cut %.1f s\n",
                options.stepSeconds,
                options.minNewSpeechSeconds,
                options.silenceHoldSeconds,
                options.maxSegmentSeconds);
    std::printf("  audio context %s\n",
                options.encodeOnlyTheAudioThereIs
                    ? "the segment's audio plus the margin"
                    : "the whole window, 1500 positions");
}

void runLive(const Request& request)
{
    auto ours = OurRuntime {0};
    auto onCoreML = std::unique_ptr<OurRuntime> {};

    if (canRunTheEncoderOnCoreML())
        onCoreML = std::make_unique<OurRuntime>(
            0, WSP::EncoderBackend::coreML, request.units);

    auto& whisper = ours.runtime();

    const auto recording = WSP::readWavFile(request.wavFile);
    const auto stream = withGap(recording, request.gapSeconds);

    std::printf("WhisperEACP live benchmark\n");
    std::printf("  audio %s: %.1f s, on repeat with a %.1f s silence gap\n",
                request.wavFile.c_str(),
                (double) recording.size() / WSP::sampleRate,
                request.gapSeconds);
    std::printf("  model tiny.en: %s, F32%s\n",
                WSP::Whisper::bundledModelDirectory().string().c_str(),
                packedProjections(whisper.packsWeights()));
    std::printf("  stream %.0f s at %d ms ticks, on %s\n",
                request.liveSeconds,
                liveTickMilliseconds,
                GPU::Device::shared().name().c_str());
    auto options = WSP::LiveOptions {};
    options.encodeOnlyTheAudioThereIs =
        request.audioContext == Request::fromTheAudio;

    printLivePolicy(options);
    printBuildType();
    printCoreMLContestant(onCoreML.get(), request);
    std::printf("\n");

    auto summaries = std::vector<LiveSummary> {};

    summaries.push_back(
        streamThroughTheTranscriber(whisper, options, stream, request.liveSeconds));
    summaries.back().column = ours.column();

    if (onCoreML != nullptr)
    {
        summaries.push_back(streamThroughTheTranscriber(
            onCoreML->runtime(), options, stream, request.liveSeconds));
        summaries.back().column = onCoreML->column();
    }

    printLiveTable(summaries);

    for (const auto& summary: summaries)
    {
        if (summary.segments > 0)
            std::printf("\n  %s, first committed line\n   %s\n",
                        summary.column.c_str(),
                        summary.firstLine.c_str());
        else
            std::printf("\n  %s committed nothing: the stream ended before a "
                        "segment closed\n",
                        summary.column.c_str());
    }
}

void run(const Request& request)
{
    auto audioSeconds = 0.0;
    const auto samples = readWindow(request.wavFile, audioSeconds);
    const auto audioContext =
        request.audioContext == Request::fromTheAudio
            ? WSP::Whisper::audioContextForSamples(
                  (int) std::lround(audioSeconds * WSP::sampleRate))
            : request.audioContext;

    // Loaded before the header rather than with the other contestants, since
    // what it packed its weights as is part of what the header says ran, and
    // what the Core ML one compiled for and how long it took to load.
    auto ours = std::make_unique<OurRuntime>(audioContext);
    auto onCoreML = std::unique_ptr<OurRuntime> {};

    if (canRunTheEncoderOnCoreML())
        onCoreML = std::make_unique<OurRuntime>(
            audioContext, WSP::EncoderBackend::coreML, request.units);

    std::printf("WhisperEACP benchmark\n");
    std::printf("  audio %s: %.1f s, zero-filled to the %d s window on both "
                "sides\n",
                request.wavFile.c_str(),
                audioSeconds,
                WSP::windowSeconds);
    std::printf("  model tiny.en: %s for WhisperEACP, F32%s; %s for "
                "whisper.cpp, F16\n",
                WSP::Whisper::bundledModelDirectory().string().c_str(),
                packedProjections(ours->packsWeights()),
                ggmlModel);
    std::printf("  runs  %d timed after %d warm-up, per contestant\n",
                request.runs,
                warmUpRuns);
    std::printf("  audio context %s\n",
                audioContext == 0 ? "the whole window, 1500 positions, on both sides"
                                  : (std::to_string(audioContext)
                                     + " encoder positions on both sides")
                                        .c_str());
    printBuildType();
    printBackends();
    printCoreMLContestant(onCoreML.get(), request);
    std::printf("\n");

    auto contestants = std::vector<std::unique_ptr<Contestant>> {};
    contestants.push_back(std::move(ours));

    if (onCoreML != nullptr)
        contestants.push_back(std::move(onCoreML));

    if (const auto gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU))
        contestants.push_back(
            std::make_unique<WhisperCpp>(true, describe(gpu), audioContext));
    else
        std::printf("  whisper.cpp has no GPU backend in this build, so it "
                    "runs without one only\n\n");

    contestants.push_back(std::make_unique<WhisperCpp>(false, "CPU", audioContext));

    auto summaries = std::vector<Summary> {};

    for (auto& contestant: contestants)
    {
        summaries.push_back(benchmark(*contestant, samples, request.runs));
        std::printf("  %s\n   %s\n\n",
                    summaries.back().name.c_str(),
                    summaries.back().text.c_str());
    }

    printTable(summaries);
}

void benchmarkMain()
{
    auto request = Request {};

    if (!parse(Apps::getAppEnvironment().commandLineArgs, request))
    {
        std::printf("%s", usage);
        Apps::setReturnValue(2);
        return;
    }

    if (!WSP::Whisper::hasBundledModel())
    {
        std::printf("%s", missingBundledModel);
        Apps::setReturnValue(2);
        return;
    }

    if (!request.live && !std::filesystem::is_regular_file(ggmlModel))
    {
        std::printf("the GGML model the configure fetched is not at %s\n",
                    ggmlModel);
        Apps::setReturnValue(2);
        return;
    }

    if (!GPU::Device::shared().isValid())
    {
        std::printf("no GPU device available - nothing here can run\n");
        Apps::setReturnValue(1);
        return;
    }

    silenceWhisperCpp();

    try
    {
        if (request.live)
            runLive(request);
        else
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
    return Apps::run(benchmarkMain);
}
