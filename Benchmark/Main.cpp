#include <WhisperEACP/WhisperEACP.h>

#include <eacp/Core/App/App.h>
#include <eacp/GPU/GPU.h>

#include <ggml-backend.h>
#include <whisper.h>

#include <algorithm>
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
// the HuggingFace safetensors, fetched at startup into this machine's own
// resource directory, F32; theirs from the GGML conversion of the same, which
// the configure fetched, F16.
//
// `--live` is the other thing this measures: the recording streamed through
// `LiveTranscriber` at the pace a microphone delivers it, so the number that
// comes out is what a machine that is listening pays rather than what one
// transcription costs. All three contestants take that stream, one after
// another — `LiveTranscriber` drives a std::function, so whisper.cpp goes
// through the same policy ours does, and the schedule is audio time and
// therefore identical across the three. The live section of README.md says why
// its numbers are not the comparison's.
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
    "       Benchmark --live [seconds] [gap seconds] [wav file] [--step=seconds]\n"
    "                 [--audio-ctx=audio]\n"
    "\n"
    "  runs        timed runs per contestant after one warm-up, 10 by default\n"
    "  wav file    16 kHz mono, at most 30 seconds; Samples/jfk.wav by default\n"
    "  --audio-ctx encoder positions both sides run over, out of 1500. Ours is\n"
    "              Whisper::setAudioContext and theirs is whisper_full_params\n"
    "              audio_ctx, which are the same knob; the whole window by\n"
    "              default. \"audio\" sizes it to the recording the way the\n"
    "              live loop does, and in --live turns that option on\n"
    "\n"
    "  --live      every contestant through LiveTranscriber, one after another,\n"
    "              streamed at real time: the recording on repeat with a gap of\n"
    "              silence between passes. The numbers, in order, are how long\n"
    "              to stream for - 30 s by default - and the gap, 1.5 s. The\n"
    "              recording may be any length here, since the segments are the\n"
    "              policy's\n"
    "  --step      LiveOptions::stepSeconds: how much new audio the open\n"
    "              segment takes before it is transcribed again, 0.5 s by\n"
    "              default. --live only\n";

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

    // The encoder positions the next run computes, which the live loop sets per
    // run out of the segment it holds. Whisper::setAudioContext on our side and
    // whisper_full_params::audio_ctx on theirs, as in the constructors.
    virtual void setAudioContext(int positions) = 0;
};

class OurRuntime final : public Contestant
{
public:
    explicit OurRuntime(int audioContext)
    {
        whisper.load(WSP::ModelFetch::directory());
        whisper.prepare();
        whisper.setAudioContext(audioContext);
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

    void setAudioContext(int positions) override
    {
        whisper.setAudioContext(positions);
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

    void setAudioContext(int positions) override { audioContext = positions; }

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

template <typename Row, typename Cell>
void printRow(const std::string& label,
              const std::vector<Row>& summaries,
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
    bool live = false;
    int runs = defaultRuns;
    double liveSeconds = defaultLiveSeconds;
    double gapSeconds = defaultLiveGapSeconds;
    double stepSeconds = WSP::LiveOptions {}.stepSeconds;
    std::string wavFile = defaultSample;

    // Encoder positions on both sides; 0 is the whole window, and
    // fromTheAudio is Whisper::audioContextForSamples over the recording.
    static constexpr auto fromTheAudio = -1;
    int audioContext = 0;
};

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
    constexpr auto stepFlag = std::string_view {"--step="};

    for (auto index = 1; index < arguments.size(); ++index)
    {
        const auto& argument = arguments[index];
        auto value = 0.0;

        if (std::string_view {argument}.starts_with(stepFlag))
        {
            if (!parseNumber(argument.substr(stepFlag.size()), value)
                || value <= 0.0)
                return false;

            request.stepSeconds = value;
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
// it, through the policy the demo app runs, and every contestant through the
// same one.
//
// What comes out is a duty cycle — model seconds per second of audio — rather
// than the cost of one transcription, because that is the number a machine that
// is listening pays. Nothing else can report it: macOS's own GPU utilisation
// counter is a short-window snapshot, so a 26 ms burst every 500 ms reads
// there as most of a core when the true share is a twentieth of one.
struct LiveSummary
{
    std::string name;
    std::string column;
    double audioSeconds = 0.0;
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
    std::string transcript;
};

// What the run that just happened made of the open segment: the pending text,
// or the line it committed when that run was the one that closed the segment.
std::string openSegmentText(const WSP::LiveTranscriber& live, int segmentsBefore)
{
    if (!live.pending().empty())
        return live.pending();

    if (live.committed().size() > segmentsBefore)
        return live.committed()[live.committed().size() - 1];

    return {};
}

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

// One tick's worth of samples, so the audio the policy sees advances with the
// tick count rather than with the wall clock. A contestant that keeps up sleeps
// between ticks and streams at real time; one that does not falls behind the
// clock but is handed the same blocks at the same audio times, which is what
// makes the run schedule the same for all three.
constexpr auto tickSamples = WSP::sampleRate * liveTickMilliseconds / 1000;

LiveSummary streamThroughTheTranscriber(Contestant& contestant,
                                        const WSP::LiveOptions& options,
                                        const WSP::Vector<float>& stream,
                                        double seconds)
{
    auto summary = LiveSummary {};
    summary.name = contestant.name();
    summary.column = contestant.column();

    auto live = WSP::LiveTranscriber {
        [&contestant, &options, &summary](WSP::Span<const float> segment)
        {
            if (options.encodeOnlyTheAudioThereIs)
                contestant.setAudioContext(WSP::Whisper::audioContextForSamples(
                    segment.size(), options.audioContextMarginSeconds));

            const auto run = contestant.transcribe(segment);

            summary.encodeSeconds += run.encodeSeconds;
            summary.decodeSeconds += run.decodeSeconds;
            summary.steps += run.steps;

            return contestant.text(run.tokens);
        },
        options};

    auto chunk = WSP::Vector<float> {};
    auto cursor = 0;
    auto pushed = (long long) 0;

    const auto streamSamples = (long long) (seconds * WSP::sampleRate);
    const auto start = Clock::now();

    for (auto tick = 1; pushed < streamSamples; ++tick)
    {
        std::this_thread::sleep_until(
            start + std::chrono::milliseconds(liveTickMilliseconds * tick));

        const auto due = std::min(streamSamples, (long long) tick * tickSamples);

        chunk.clear();

        while (pushed < due)
        {
            chunk.add(stream[cursor]);
            cursor = (cursor + 1) % stream.size();
            ++pushed;
        }

        live.push(chunk);

        const auto runsBefore = live.stats().runs;
        const auto segmentsBefore = live.committed().size();
        const auto textBefore = live.pending();

        live.update();

        if (live.stats().runs == runsBefore)
            continue;

        const auto runSeconds = live.stats().lastRunSeconds;

        ++summary.runs;
        summary.modelSeconds += runSeconds;
        summary.longestRunSeconds = std::max(summary.longestRunSeconds, runSeconds);

        if (openSegmentText(live, segmentsBefore) != textBefore)
            ++summary.runsThatChangedTheText;
    }

    summary.wallSeconds = secondsSince(start);
    summary.audioSeconds = (double) pushed / WSP::sampleRate;
    summary.segments = live.committed().size();

    for (const auto& line: live.committed())
        summary.transcript += line + "\n";

    if (summary.segments > 0)
        summary.firstLine = live.committed()[0];

    return summary;
}

void printLiveTable(const std::vector<LiveSummary>& summaries)
{
    const auto& reference = summaries.front();

    const auto perRun = [](const LiveSummary& s, double total)
    { return 1000.0 * total / std::max(1, s.runs); };

    printRow("", summaries, [](const LiveSummary& s) { return s.column; });

    printRow("stream, audio",
             summaries,
             [](const LiveSummary& s) { return cell(s.audioSeconds, "s", 1); });

    printRow("stream, wall clock",
             summaries,
             [](const LiveSummary& s) { return cell(s.wallSeconds, "s", 1); });

    printRow("runs",
             summaries,
             [](const LiveSummary& s) { return std::to_string(s.runs); });

    printRow("runs that changed the text",
             summaries,
             [](const LiveSummary& s)
             { return std::to_string(s.runsThatChangedTheText); });

    printRow("model",
             summaries,
             [](const LiveSummary& s) { return cell(s.modelSeconds, "s", 3); });

    printRow("duty",
             summaries,
             [](const LiveSummary& s)
             { return cell(100.0 * s.modelSeconds / s.audioSeconds, "%", 1); });

    printRow("per run, mean",
             summaries,
             [&perRun](const LiveSummary& s)
             { return cell(perRun(s, s.modelSeconds), "ms", 1); });

    printRow("per run, longest",
             summaries,
             [](const LiveSummary& s)
             { return cell(1000.0 * s.longestRunSeconds, "ms", 1); });

    printRow("encode, mean",
             summaries,
             [&perRun](const LiveSummary& s)
             { return cell(perRun(s, s.encodeSeconds), "ms", 1); });

    printRow("decode, mean",
             summaries,
             [&perRun](const LiveSummary& s)
             { return cell(perRun(s, s.decodeSeconds), "ms", 1); });

    printRow("steps per run",
             summaries,
             [](const LiveSummary& s)
             { return cell((double) s.steps / std::max(1, s.runs), "", 1); });

    printRow("segments committed",
             summaries,
             [](const LiveSummary& s) { return std::to_string(s.segments); });

    printRow("transcript",
             summaries,
             [&reference](const LiveSummary& s)
             {
                 if (&s == &reference)
                     return std::string {"reference"};

                 return std::string {
                     s.transcript == reference.transcript ? "same" : "DIFFERS"};
             });

    std::printf("\n  a run at this cadence costs more than the comparison "
                "measures: the GPU clocks\n  down between bursts half a second "
                "apart. whisper.cpp has no live layer, so its\n  columns are "
                "this policy driving whisper_full - the runs, the segments and "
                "the\n  audio each run saw are ours.\n");
}

void printLivePolicy(const WSP::LiveOptions& options, int positions)
{
    std::printf("  policy step %.2f s, new speech %.2f s, silence hold %.2f s, "
                "segment cut %.1f s\n",
                options.stepSeconds,
                options.minNewSpeechSeconds,
                options.silenceHoldSeconds,
                options.maxSegmentSeconds);

    if (options.encodeOnlyTheAudioThereIs)
        std::printf("  audio context the segment's audio plus the margin, on "
                    "every side\n");
    else if (positions > 0)
        std::printf("  audio context %d encoder positions on every side\n",
                    positions);
    else
        std::printf("  audio context the whole window, 1500 positions, on every "
                    "side\n");
}

// Our runtime, then whisper.cpp on its GPU backend if this build has one, then
// whisper.cpp without one. Each loaded once, and the audio context they start
// on is the one the mode asked for.
std::vector<std::unique_ptr<Contestant>> makeContestants(int audioContext)
{
    auto contestants = std::vector<std::unique_ptr<Contestant>> {};
    contestants.push_back(std::make_unique<OurRuntime>(audioContext));

    if (const auto gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU))
        contestants.push_back(
            std::make_unique<WhisperCpp>(true, describe(gpu), audioContext));
    else
        std::printf("  whisper.cpp has no GPU backend in this build, so it "
                    "runs without one only\n\n");

    contestants.push_back(std::make_unique<WhisperCpp>(false, "CPU", audioContext));

    return contestants;
}

void runLive(const Request& request)
{
    const auto recording = WSP::readWavFile(request.wavFile);
    const auto stream = withGap(recording, request.gapSeconds);

    auto options = WSP::LiveOptions {};
    options.stepSeconds = request.stepSeconds;
    options.encodeOnlyTheAudioThereIs =
        request.audioContext == Request::fromTheAudio;

    std::printf("WhisperEACP live benchmark\n");
    std::printf("  audio %s: %.1f s, on repeat with a %.1f s silence gap\n",
                request.wavFile.c_str(),
                (double) recording.size() / WSP::sampleRate,
                request.gapSeconds);
    std::printf("  model tiny.en: %s for WhisperEACP, F32; %s for whisper.cpp, "
                "F16\n",
                WSP::ModelFetch::directory().string().c_str(),
                ggmlModel);
    std::printf("  stream %.0f s at %d ms ticks, on %s\n",
                request.liveSeconds,
                liveTickMilliseconds,
                GPU::Device::shared().name().c_str());

    // A fixed context is the one every run starts on and keeps; "audio" is the
    // option above, which sets it per run out of the segment instead.
    const auto positions =
        options.encodeOnlyTheAudioThereIs ? 0 : request.audioContext;

    printLivePolicy(options, positions);
    printBuildType();
    printBackends();
    std::printf("\n");

    auto contestants = makeContestants(positions);
    auto summaries = std::vector<LiveSummary> {};

    for (auto& contestant: contestants)
    {
        summaries.push_back(streamThroughTheTranscriber(
            *contestant, options, stream, request.liveSeconds));

        std::printf("  %s\n   %s\n\n",
                    summaries.back().name.c_str(),
                    summaries.back().segments > 0
                        ? summaries.back().firstLine.c_str()
                        : "nothing committed: the stream ended before a segment "
                          "closed");
    }

    printLiveTable(summaries);
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

    std::printf("WhisperEACP benchmark\n");
    std::printf("  audio %s: %.1f s, zero-filled to the %d s window on both "
                "sides\n",
                request.wavFile.c_str(),
                audioSeconds,
                WSP::windowSeconds);
    std::printf("  model tiny.en: %s for WhisperEACP, F32; %s for whisper.cpp, "
                "F16\n",
                WSP::ModelFetch::directory().string().c_str(),
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
    std::printf("\n");

    auto contestants = makeContestants(audioContext);
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

// At file scope because the command line is parsed and the model fetched in
// main, before the loop this runs inside opens.
Request request;

// The HuggingFace weights on disk before anything else happens. Called from main
// rather than from here for the reason the comment at the top of this file gives:
// ModelFetch::fetch pumps the event loop, and a callback running inside that loop
// may not.
bool fetchOurModel()
{
    const auto announce = !WSP::ModelFetch::isAvailable();

    if (announce)
        std::printf("fetching %s into %s\n",
                    WSP::ModelFetch::repository,
                    WSP::ModelFetch::directory().string().c_str());

    const auto outcome = WSP::ModelFetch::fetch(
        WSP::ModelFetch::Freshness::trust,
        [](const WSP::ModelFetch::Progress& progress)
        {
            using Stage = eacp::OnlineResource::Progress::Stage;

            if (progress.file.stage != Stage::downloading)
                return;

            std::printf("\r  %-58s",
                        WSP::ModelFetch::progressText(progress).c_str());
            std::fflush(stdout);
        });

    if (announce)
        std::printf("\n");

    if (!outcome.ok)
        std::printf("the tiny.en weights could not be fetched: %s\n",
                    outcome.error.c_str());

    return outcome.ok;
}

void benchmarkMain()
{
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

    if (!parse(Apps::getAppEnvironment().commandLineArgs, request))
    {
        std::printf("%s", usage);
        return 2;
    }

    // Both downloads before the loop: ours pumps the loop to do it, and theirs
    // is a path the configure fixed, so a missing one is worth saying before a
    // window of work opens.
    if (!fetchOurModel())
        return 2;

    if (!std::filesystem::is_regular_file(ggmlModel))
    {
        std::printf("the GGML model the configure fetched is not at %s\n",
                    ggmlModel);
        return 2;
    }

    return Apps::run(benchmarkMain);
}
