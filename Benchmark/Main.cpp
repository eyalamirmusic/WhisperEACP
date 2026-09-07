#include <WhisperEACP/WhisperEACP.h>

#include <eacp/Core/App/App.h>
#include <eacp/GPU/GPU.h>

#include <ggml-backend.h>
#include <whisper.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
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
// the HuggingFace safetensors the build copied beside this binary, F32; theirs
// from the GGML conversion of the same, F16.
//
// Inside eacp::Apps::run for the reason every GPU-touching thing here is — the
// Metal backend is written against the run loop and autorelease pool that
// owns, and Transcribe is the same shape.

using namespace eacp;

namespace
{
constexpr auto buildType = WHISPER_EACP_BUILD_TYPE;

constexpr auto usage =
    "usage: Benchmark [runs] [wav file]\n"
    "\n"
    "  runs      timed runs per contestant after one warm-up, 10 by default\n"
    "  wav file  16 kHz mono, at most 30 seconds; Samples/jfk.wav by default\n";

constexpr auto missingBundledModel =
    "this build copied no model beside the binary. Configure with\n"
    "-DWHISPER_EACP_FETCH_MODEL=ON to get one.\n";

constexpr auto defaultRuns = 10;
constexpr auto warmUpRuns = 1;
constexpr auto defaultSample = WHISPER_EACP_SAMPLE_DIR "/jfk.wav";
constexpr auto ggmlModel = WHISPER_EACP_GGML_MODEL;

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
struct Run
{
    double wallSeconds = 0.0;
    double encodeSeconds = 0.0;
    double decodeSeconds = 0.0;
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

class OurRuntime final : public Contestant
{
public:
    OurRuntime()
    {
        whisper.loadBundled();
        whisper.prepare();
    }

    std::string name() const override
    {
        return "WhisperEACP on " + GPU::Device::shared().name();
    }

    std::string column() const override { return "WhisperEACP"; }

    Run transcribe(WSP::Span<const float> samples) override
    {
        const auto start = Clock::now();

        auto run = Run {};
        run.tokens = whisper.transcribe(samples);
        run.wallSeconds = secondsSince(start);
        run.encodeSeconds = whisper.lastEncodeSeconds();
        run.decodeSeconds = whisper.lastDecodeSeconds();
        run.steps = whisper.lastStepCount();

        return run;
    }

    std::string text(const WSP::Vector<int>& tokens) const override
    {
        return whisper.textForTokens(tokens);
    }

private:
    WSP::Whisper whisper;
};

struct ContextDeleter
{
    void operator()(whisper_context* context) const { whisper_free(context); }
};

using Context = std::unique_ptr<whisper_context, ContextDeleter>;

// Greedy, one segment, no timestamps, temperature 0 with no fallback, and the
// non-speech suppression whisper.cpp leaves off by default turned on: the
// policy our search runs, spelled the way Tests/Oracle spells it, so both
// sides do the same work and the transcripts compare token for token.
whisper_full_params greedyParameters(int threads)
{
    auto parameters = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    parameters.n_threads = threads;
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
    WhisperCpp(bool useGpu, std::string backend)
        : backendName(std::move(backend))
        , columnName(useGpu ? "whisper.cpp GPU" : "whisper.cpp CPU")
        , threads(whisperCppDefaultThreads())
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
                         greedyParameters(threads),
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

    for (auto index = 0; index < runs; ++index)
    {
        const auto run = contestant.transcribe(samples);
        wall.push_back(run.wallSeconds);
        encode.push_back(run.encodeSeconds);
        decode.push_back(run.decodeSeconds);
        summary.steps = run.steps;
    }

    summary.wallMedian = median(wall);
    summary.wallBest = *std::min_element(wall.begin(), wall.end());
    summary.encodeMedian = median(encode);
    summary.decodeMedian = median(decode);

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
             [](const Summary& s) { return cell(s.wallMedian, "s", 3); });

    printRow("transcribe, best",
             summaries,
             [](const Summary& s) { return cell(s.wallBest, "s", 3); });

    printRow("x real time, 30 s window",
             summaries,
             [](const Summary& s)
             { return cell(WSP::windowSeconds / s.wallMedian, "", 1); });

    printRow("encode, median",
             summaries,
             [](const Summary& s) { return cell(s.encodeMedian, "s", 3); });

    printRow("decode, median",
             summaries,
             [](const Summary& s) { return cell(s.decodeMedian, "s", 3); });

    printRow(
        "decode per step, median",
        summaries,
        [](const Summary& s)
        { return cell(1000.0 * s.decodeMedian / std::max(1, s.steps), "ms", 1); });

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

struct Request
{
    int runs = defaultRuns;
    std::string wavFile = defaultSample;
};

// Each argument is a run count if it parses as one and the WAV file otherwise,
// so both orders read naturally.
bool parse(const WSP::Vector<std::string>& arguments, Request& request)
{
    if (arguments.size() > 3)
        return false;

    for (auto index = 1; index < arguments.size(); ++index)
    {
        const auto& argument = arguments[index];
        auto runs = 0;
        const auto [end, error] = std::from_chars(
            argument.data(), argument.data() + argument.size(), runs);

        if (error == std::errc {} && end == argument.data() + argument.size())
        {
            if (runs < 1)
                return false;

            request.runs = runs;
        }
        else
        {
            request.wavFile = argument;
        }
    }

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

void run(const Request& request)
{
    auto audioSeconds = 0.0;
    const auto samples = readWindow(request.wavFile, audioSeconds);

    std::printf("WhisperEACP benchmark\n");
    std::printf("  audio %s: %.1f s, zero-filled to the %d s window on both "
                "sides\n",
                request.wavFile.c_str(),
                audioSeconds,
                WSP::windowSeconds);
    std::printf("  model tiny.en: %s for WhisperEACP, F32; %s for whisper.cpp, "
                "F16\n",
                WSP::Whisper::bundledModelDirectory().string().c_str(),
                ggmlModel);
    std::printf("  runs  %d timed after %d warm-up, per contestant\n",
                request.runs,
                warmUpRuns);
    printBuildType();
    printBackends();
    std::printf("\n");

    auto contestants = std::vector<std::unique_ptr<Contestant>> {};
    contestants.push_back(std::make_unique<OurRuntime>());

    if (const auto gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU))
        contestants.push_back(std::make_unique<WhisperCpp>(true, describe(gpu)));
    else
        std::printf("  whisper.cpp has no GPU backend in this build, so it "
                    "runs without one only\n\n");

    contestants.push_back(std::make_unique<WhisperCpp>(false, "CPU"));

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

    if (!std::filesystem::is_regular_file(ggmlModel))
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
