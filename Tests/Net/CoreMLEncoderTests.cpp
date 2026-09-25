#include "Common.h"

// The tiny.en encoder run both ways over the same mel, the kernels in fp32 and
// Core ML in fp16, compared row by row at each of three contexts and under
// each compute-unit setting, with the times and the cache logged. Apple only:
// elsewhere eacp has no Core ML runner and this file compiles to nothing.
//
// Every test here skips without a device, without the model or where the OS
// cannot run the encoder's program. EACP_REQUIRE_ANE=1, as in eacp's own
// MLTests, turns the skips that would hide an absent Neural Engine into
// failures, holds the engine and GPU settings to their own bounds rather than
// the CPU's, and asserts where the compute plan placed the encoder, which is a
// read that costs the engine compile again, about 14 s.
// WHISPER_EACP_SLOW_TESTS=1 runs errorByDepth, twenty cold compiles.

#if EACP_HAS_COREML

#include <WhisperEACP/Encoder/CoreMLEncoder.h>

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;
using eacp::ML::ComputeUnits;

namespace
{
using Clock = std::chrono::steady_clock;

double millisecondsSince(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

bool isSwitchedOn(const char* variable)
{
    const auto* value = std::getenv(variable);
    return value != nullptr && std::string {value} == "1";
}

bool isAneRequired()
{
    return isSwitchedOn("EACP_REQUIRE_ANE");
}

bool areSlowTestsOn()
{
    return isSwitchedOn("WHISPER_EACP_SLOW_TESTS");
}

bool canRun()
{
    if (!eacp::GPU::Device::shared().isValid() || !hasWhisperModel())
        return false;

    if (Whisper::supportsEncoderBackend(EncoderBackend::coreML))
        return true;

    check(!isAneRequired());
    return false;
}

const char* nameOf(ComputeUnits units)
{
    switch (units)
    {
        case ComputeUnits::all:
            return "all";
        case ComputeUnits::cpuAndNeuralEngine:
            return "cpuAndNeuralEngine";
        case ComputeUnits::cpuAndGPU:
            return "cpuAndGPU";
        case ComputeUnits::cpu:
            return "cpu";
    }

    return "?";
}

// jfk.wav through the kernel front end, zero-filled to the window as Whisper
// does: an encoder's error on speech, not on a synthetic sinusoid. The
// synthetic mel stands in where the sample is missing.
Vector<float> windowMel(const EncoderShape& shape)
{
    if (!hasSampleFile(jfkSample))
        return syntheticMel(shape);

    const auto recording = readWavFile(sampleFile(jfkSample));
    auto samples = sized(windowSamples);

    for (auto index = 0; index < recording.size() && index < windowSamples; ++index)
        samples[index] = recording[index];

    const auto preprocessor =
        PreprocessorConfig::fromFile(modelFile("preprocessor_config.json"));
    const auto filterBank = preprocessor.makeMelFilterBuffer();

    auto frontEnd = MelSpectrogram {};
    frontEnd.prepare();

    const auto sampleBuffer = storageOf(samples);
    const auto mel = outputFor(shape.melElementCount());

    auto commands = eacp::GPU::Device::shared().makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        frontEnd.encode(pass, sampleBuffer, filterBank, mel);
    }

    commands.commit();

    return readBack(mel, shape.melElementCount());
}

// The file, and the host weights that view it, in one place that never moves.
struct TinyEn
{
    TinyEn()
        : shape(EncoderShape::fromConfig(
              ModelConfig::fromFile(modelFile("config.json")), windowFrames))
        , file(SafeTensors::fromFile(modelFile("model.safetensors")))
        , mel(windowMel(shape))
        , melBuffer(storageOf(mel))
        , hostWeights(file, shape, WeightPacking::Float, WeightPlacement::Host)
    {
    }

    EncoderShape shape;
    SafeTensors file;
    Vector<float> mel;
    eacp::GPU::Buffer melBuffer;
    EncoderWeights hostWeights;
};

TinyEn& tinyEn()
{
    static auto model = std::make_unique<TinyEn>();
    return *model;
}

const Vector<float>& kernelRows(int context)
{
    static auto computed = std::map<int, Vector<float>> {};

    if (!computed.contains(context))
    {
        auto& model = tinyEn();
        computed[context] = runEncoder(model.shape, model.file, model.mel, context);
    }

    return computed[context];
}

CoreMLEncoder& coreMLEncoder(ComputeUnits units)
{
    static auto encoders = std::map<ComputeUnits, std::unique_ptr<CoreMLEncoder>> {};

    if (auto found = encoders.find(units); found != encoders.end())
        return *found->second;

    auto& model = tinyEn();
    auto encoder = std::make_unique<CoreMLEncoder>(model.shape, audioContextSpan());

    auto options = CoreMLEncoderOptions {};
    options.units = units;
    options.cacheDirectory = sharedCoreMLCacheDirectory();

    const auto loaded = encoder->prepare(model.hostWeights, options);
    check(loaded.ok);

    if (!loaded.ok)
        std::cout << "  " << nameOf(units) << ": " << loaded.error << "\n";

    std::cout << "  " << nameOf(units) << ": loaded in "
              << encoder->lastLoadSeconds() * 1000.0 << " ms, "
              << (encoder->wasCacheHit() ? "cache hit" : "compiled")
              << (encoder->usesFusedAttention() ? ", fused attention"
                                                : ", unfused attention")
              << "\n";

    return *encoders.emplace(units, std::move(encoder)).first->second;
}

struct Agreement
{
    double maxAbs = 0.0;
    double maxMixed = 0.0;
    double meanAbs = 0.0;
    double tailAbs = 0.0;
};

// Max abs, and the mixed measure worstError asserts on elsewhere: an encoder
// row is mostly values near zero, where a relative error means nothing. The
// mean and the 99.9th percentile say whether the worst is a handful of
// elements or the whole output.
Agreement agreement(const Vector<float>& actual, const Vector<float>& expected)
{
    auto result = Agreement {};
    auto errors = Vector<double> {};

    for (auto index = 0; index < expected.size(); ++index)
    {
        const auto error = std::abs((double) actual[index] - expected[index]);
        result.maxAbs = std::max(result.maxAbs, error);
        result.maxMixed =
            std::max(result.maxMixed, error / (1.0 + std::abs(expected[index])));
        result.meanAbs += error;
        errors.add(error);
    }

    if (!errors.empty())
    {
        result.meanAbs /= errors.size();
        std::sort(errors.begin(), errors.end());
        result.tailAbs = errors[(int) (0.999 * (errors.size() - 1))];
    }

    return result;
}

std::ostream& operator<<(std::ostream& stream, const Agreement& measured)
{
    return stream << "max abs " << measured.maxAbs << " (99.9% " << measured.tailAbs
                  << ", mean " << measured.meanAbs << "), max abs/(1+|x|) "
                  << measured.maxMixed;
}

struct Encoded
{
    Vector<float> rows;
    double encodeMilliseconds = 0.0;
    double predictMilliseconds = 0.0;
};

Encoded encodeOnCoreML(CoreMLEncoder& encoder, int context)
{
    auto& model = tinyEn();
    auto rows = outputFor(model.shape.elementCount());

    const auto start = Clock::now();
    encoder.encode(model.melBuffer, rows, context);
    const auto elapsed = millisecondsSince(start);

    return {readBack(rows, context * model.shape.width),
            elapsed,
            encoder.lastPredictSeconds() * 1000.0};
}

struct Tolerance
{
    double maxAbs = 0.0;
    double tailAbs = 0.0;
    double meanAbs = 0.0;
};

// Measured on an M5 Max, macOS 27, over jfk.wav's mel at 1500 / 576 / 448
// positions against the kernels' fp32 rows, and asserted at about three times
// the worst of the three. The maximum is a handful of elements: at 448 on the
// engine it is 1.1 where the 99.9th percentile is 0.07, and it appears only
// after the fourth layer (errorByDepth), so the tail and the mean are held
// as well.
//
//   engine:  max 0.26 / 0.21 / 1.10, 99.9% 0.056 / 0.069 / 0.072,
//            mean 0.0076 / 0.0091 / 0.0090
//   GPU:     max 0.031 / 0.055 / 0.15, 99.9% 0.0084 / 0.011 / 0.010,
//            mean 0.0011 / 0.0015 / 0.0014
//   CPU:     max 0.42 / 0.20 / 2.60, 99.9% 0.14 / 0.070 / 0.098,
//            mean 0.018 / 0.010 / 0.010
constexpr auto cpuTolerance = Tolerance {8.0, 0.45, 0.06};

Tolerance ownToleranceFor(ComputeUnits units)
{
    switch (units)
    {
        case ComputeUnits::cpuAndNeuralEngine:
            return {3.5, 0.22, 0.03};
        case ComputeUnits::all:
        case ComputeUnits::cpuAndGPU:
            return {0.5, 0.035, 0.005};
        case ComputeUnits::cpu:
            return cpuTolerance;
    }

    return {};
}

// The macOS CI runner has no Neural Engine and places every op on the CPU
// under every setting, so a setting is held to its own device's bound only
// under EACP_REQUIRE_ANE=1, as eacp's MLTests do; otherwise to the CPU's.
Tolerance toleranceFor(ComputeUnits units)
{
    return isAneRequired() ? ownToleranceFor(units) : cpuTolerance;
}

// Max abs per depth, 0 to 4 layers, final norm included, at about three
// times the worst of 1500 and 448 positions: engine 0.064 / 0.15 / 0.21 /
// 0.24 / 0.26, GPU 0.0069 / 0.024 / 0.031 / 0.023 / 0.031. The full depth
// keeps the full encoder's bound, since at 448 the fourth layer amplifies one
// element to 1.1 on the engine and 0.15 on the GPU.
constexpr auto engineMaxByDepth = Array<double, 5> {0.2, 0.45, 0.65, 0.75, 3.5};
constexpr auto gpuMaxByDepth = Array<double, 5> {0.02, 0.075, 0.085, 0.075, 0.5};

Tolerance toleranceFor(ComputeUnits units, int depth)
{
    auto tolerance = toleranceFor(units);

    if (!isAneRequired() || units == ComputeUnits::cpu)
        return tolerance;

    const auto& bounds =
        units == ComputeUnits::cpuAndNeuralEngine ? engineMaxByDepth : gpuMaxByDepth;

    tolerance.maxAbs = bounds[depth];
    return tolerance;
}

// Exact on the CPU and the engine, whose arithmetic repeats; the GPU's
// reductions may not, and without a plan the placement is unknown.
double repeatBoundFor(ComputeUnits units)
{
    const auto repeats =
        units == ComputeUnits::cpu || units == ComputeUnits::cpuAndNeuralEngine;

    return repeats ? 0.0 : 1.0e-3;
}

void checkWithin(const Agreement& measured, const Tolerance& tolerance)
{
    check(measured.maxAbs <= tolerance.maxAbs);
    check(measured.tailAbs <= tolerance.tailAbs);
    check(measured.meanAbs <= tolerance.meanAbs);
}

void compareAtEachContext(ComputeUnits units)
{
    auto& encoder = coreMLEncoder(units);

    if (!encoder.isPrepared())
        return;

    for (auto context: {1500, 576, 448})
    {
        const auto first = encodeOnCoreML(encoder, context);
        const auto warm = encodeOnCoreML(encoder, context);
        const auto& expected = kernelRows(context);

        check(warm.rows.size() == expected.size());

        const auto measured = agreement(warm.rows, expected);

        std::cout << "  " << nameOf(units) << " at " << context << ": " << measured
                  << "; encode " << first.encodeMilliseconds << " ms first, "
                  << warm.encodeMilliseconds << " ms warm (predict "
                  << warm.predictMilliseconds << " ms)\n";

        checkWithin(measured, toleranceFor(units));

        // Two runs over the same mel agree: nothing of the first run's rows
        // survives into the second's.
        check(agreement(first.rows, warm.rows).maxAbs <= repeatBoundFor(units));
    }
}

bool isEncoderOp(const std::string& type)
{
    return type == "linear" || type == "conv" || type == "layer_norm"
           || type == "scaled_dot_product_attention";
}
} // namespace

auto tMatchesKernelsOnTheEngine =
    test("Net/CoreML/tinyEnMatchesTheKernelsOnTheEngine") = []
{
    if (!canRun())
        return;

    compareAtEachContext(ComputeUnits::cpuAndNeuralEngine);
};

auto tMatchesKernelsUnderAll =
    test("Net/CoreML/tinyEnMatchesTheKernelsUnderAll") = []
{
    if (!canRun())
        return;

    compareAtEachContext(ComputeUnits::all);
};

auto tMatchesKernelsOnTheCPU =
    test("Net/CoreML/tinyEnMatchesTheKernelsOnTheCPU") = []
{
    if (!canRun())
        return;

    compareAtEachContext(ComputeUnits::cpu);
};

// The error per stage, since layer norm is where an fp16 encoder drifts
// first: the encoder cut after 0 to 4 layers, final norm included, on both
// backends at one fixed context. A fixed program compiles for the engine in
// about a second where the enumerated one takes fourteen, and each depth is
// a program of its own, so these go to a cache of their own that is removed
// afterwards.
auto tErrorByDepth = test("Net/CoreML/errorByDepth") = []
{
    if (!areSlowTestsOn() || !canRun())
        return;

    auto& model = tinyEn();
    const auto cache = std::filesystem::temp_directory_path()
                       / ("whisper-net-depth-" + std::to_string(::getpid()));

    for (auto units: {ComputeUnits::cpuAndNeuralEngine, ComputeUnits::cpuAndGPU})
    {
        for (auto context: {1500, 448})
        {
            for (auto depth = 0; depth <= model.shape.layers; ++depth)
            {
                auto shape = model.shape;
                shape.layers = depth;

                const auto weights = EncoderWeights {
                    model.file, shape, WeightPacking::Float, WeightPlacement::Host};
                const auto contexts = Array<int, 1> {context};

                auto encoder =
                    CoreMLEncoder {shape, Span<const int> {contexts.data(), 1}};

                auto options = CoreMLEncoderOptions {};
                options.units = units;
                options.cacheDirectory = eacp::FilePath {cache};

                const auto loaded = encoder.prepare(weights, options);
                check(loaded.ok);

                if (!loaded.ok)
                    continue;

                auto rows = outputFor(shape.elementCount());
                encoder.encode(model.melBuffer, rows, context);

                const auto measured =
                    agreement(readBack(rows, context * shape.width),
                              runEncoder(shape, model.file, model.mel, context));

                std::cout << "  " << nameOf(units) << " at " << context << ", "
                          << depth << " layers: " << measured << "\n";

                checkWithin(measured, toleranceFor(units, depth));
            }
        }
    }

    auto error = std::error_code {};
    std::filesystem::remove_all(cache, error);
};

auto tAsyncMatchesBlocking = test("Net/CoreML/encodeAsyncMatchesEncode") = []
{
    if (!canRun())
        return;

    auto& encoder = coreMLEncoder(ComputeUnits::cpuAndNeuralEngine);

    if (!encoder.isPrepared())
        return;

    constexpr auto context = 576;
    auto& model = tinyEn();

    const auto blocking = encodeOnCoreML(encoder, context);

    auto rows = outputFor(model.shape.elementCount());
    auto resolvedOnMain = false;

    const auto noteThread = [&resolvedOnMain](const eacp::ML::Result&)
    { resolvedOnMain = eacp::Threads::isMainThread(); };

    auto pending = encoder.encodeAsync(model.melBuffer, rows, context);
    pending.then(noteThread);

    const auto result = pending.waitFor(eacp::Time::MS {30000});
    check(result.ok);
    check(resolvedOnMain);

    const auto asynchronous = readBack(rows, context * model.shape.width);
    check(agreement(asynchronous, blocking.rows).maxAbs
          <= repeatBoundFor(ComputeUnits::cpuAndNeuralEngine));
};

// Each context has one pair of arrays, so a second encodeAsync at a context
// whose first is still pending, or a blocking encode there, would overwrite
// the input the queued prediction reads. Both are refused until it resolves.
auto tRefusesASecondPendingEncode =
    test("Net/CoreML/refusesASecondPendingEncodeAtOneContext") = []
{
    if (!canRun())
        return;

    auto& encoder = coreMLEncoder(ComputeUnits::cpuAndNeuralEngine);

    if (!encoder.isPrepared())
        return;

    constexpr auto context = 448;
    auto& model = tinyEn();
    auto rows = outputFor(model.shape.elementCount());
    auto otherRows = outputFor(model.shape.elementCount());

    auto pending = encoder.encodeAsync(model.melBuffer, rows, context);

    check(throwsLogicError(
        [&] { encoder.encodeAsync(model.melBuffer, otherRows, context); }));
    check(throwsLogicError(
        [&] { encoder.encode(model.melBuffer, otherRows, context); }));

    check(pending.waitFor(eacp::Time::MS {30000}).ok);

    auto next = encoder.encodeAsync(model.melBuffer, otherRows, context);
    check(next.waitFor(eacp::Time::MS {30000}).ok);

    check(agreement(readBack(otherRows, context * model.shape.width),
                    readBack(rows, context * model.shape.width))
              .maxAbs
          <= repeatBoundFor(ComputeUnits::cpuAndNeuralEngine));
};

// MultiArray::copyTo would copy nothing into a buffer too small for the rows
// and leave the caller reading whatever was there.
auto tRefusesRowsTooSmall = test("Net/CoreML/refusesARowsBufferTooSmall") = []
{
    if (!canRun())
        return;

    auto& encoder = coreMLEncoder(ComputeUnits::cpuAndNeuralEngine);

    if (!encoder.isPrepared())
        return;

    constexpr auto context = 448;
    auto& model = tinyEn();
    auto rows = outputFor(context * model.shape.width - 1);

    check(throwsLogicError([&] { encoder.encode(model.melBuffer, rows, context); }));
    check(throwsLogicError(
        [&] { encoder.encodeAsync(model.melBuffer, rows, context); }));
};

auto tRefusesAContextOutsideTheSet =
    test("Net/CoreML/refusesAContextOutsideTheSet") = []
{
    auto whisper = Whisper {};
    whisper.setEncoderBackend(EncoderBackend::coreML);
    check(throwsModelError([&] { whisper.setAudioContext(600); }));

    if (!canRun())
        return;

    auto& encoder = coreMLEncoder(ComputeUnits::cpuAndNeuralEngine);
    auto rows = outputFor(tinyEn().shape.elementCount());

    check(throwsModelError([&] { encoder.encode(tinyEn().melBuffer, rows, 600); }));
    check(throwsModelError([&] { encoder.encode(tinyEn().melBuffer, rows, 0); }));
};

// One read of the plan, and only when asked for: it is the engine compile
// again. Every conv, linear, layer norm and attention on the engine.
auto tPlacedOnTheEngine = test("Net/CoreML/encoderIsPlacedOnTheEngine") = []
{
    if (!isAneRequired() || !canRun())
        return;

    check(eacp::ML::hasNeuralEngine());
    check(eacp::ML::hasComputePlan());

    auto& encoder = coreMLEncoder(ComputeUnits::cpuAndNeuralEngine);

    const auto start = Clock::now();
    const auto& plan = encoder.computePlan();
    std::cout << "  compute plan read in " << millisecondsSince(start) << " ms, "
              << plan.ops.size() << " ops\n";

    check(!plan.isEmpty());

    auto placed = 0;

    for (const auto& op: plan.ops)
    {
        if (!isEncoderOp(op.type))
            continue;

        ++placed;

        if (op.device != eacp::ML::ComputePlan::Device::neuralEngine)
            std::cout << "  off the engine: " << op.type << " " << op.name << " on "
                      << eacp::ML::toString(op.device) << "\n";

        check(op.device == eacp::ML::ComputePlan::Device::neuralEngine);
    }

    // 2 conv, 24 linear, 9 layer_norm and 4 attention.
    check(placed == 39);
};

// The whole runtime with its encoder on Core ML: the transcript is the
// kernels' token for token, and the times say where the encode went.
auto tWhisperOnCoreML = test("Net/CoreML/whisperTranscribesOnCoreML") = []
{
    if (!canRun() || !hasSampleFile(jfkSample))
        return;

    auto whisper = Whisper {};
    whisper.load(modelDirectory());
    whisper.setEncoderBackend(EncoderBackend::coreML);
    whisper.setEncoderCacheDirectory(sharedCoreMLCacheDirectory());

    const auto start = Clock::now();
    whisper.prepare();
    const auto prepareMilliseconds = millisecondsSince(start);

    check(throwsLogicError(
        [&] { whisper.setEncoderComputeUnits(EncoderComputeUnits::all); }));
    check(throwsModelError([&] { whisper.setAudioContext(600); }));

    const auto samples = readWavFile(sampleFile(jfkSample));

    whisper.transcribe(samples);
    const auto tokens = whisper.transcribe(samples);
    const auto text = whisper.textForTokens(tokens);

    std::cout << "  prepared in " << prepareMilliseconds << " ms ("
              << (whisper.encoderWasCacheHit() ? "cache hit" : "compiled")
              << "); encode " << whisper.lastEncodeSeconds() * 1000.0
              << " ms, of which predict "
              << whisper.lastEncoderPredictSeconds() * 1000.0 << " ms; decode "
              << whisper.lastDecodeSeconds() * 1000.0 << " ms\n";
    std::cout << "  transcript:" << text << "\n";

    check(trimmed(text) == jfkTranscript);

    const auto kernelTokens = preparedModel().transcribe(samples);
    check(tokens.size() == kernelTokens.size());

    for (auto index = 0; index < tokens.size() && index < kernelTokens.size();
         ++index)
        check(tokens[index] == kernelTokens[index]);

    whisper.setAudioContext(576);
    const auto reducedTokens = whisper.transcribe(samples);
    std::cout << "  at 576 positions:" << whisper.textForTokens(reducedTokens)
              << "\n";
};

#endif
