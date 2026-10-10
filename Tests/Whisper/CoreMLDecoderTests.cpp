#include "Common.h"

#include "../Decoder/Common.h"

// The tiny.en decoder run both ways over the same encoder rows and the same
// tokens, the kernels in fp32 and Core ML in fp16, compared logit by logit
// and token by token under each compute-unit setting, over the whole window
// and over a shorter context that only the cross mask keeps the rest of the
// window out of. Then the runtime on it, with the encoder on either backend:
// the pinned jfk.wav transcript. Apple only: elsewhere eacp has no Core ML
// runner and this file compiles to nothing.
//
// Every test skips without a device, without the model or where the OS cannot
// run the decoder's programs. Every Core ML model compiles into the shared
// test cache, so a machine pays each compile once.

#if EACP_HAS_COREML

#include <WhisperEACP/Decoder/CoreMLDecoder.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

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

bool canRun()
{
    return eacp::GPU::Device::shared().isValid() && hasWhisperModel()
           && Whisper::supportsDecoderBackend(DecoderBackend::coreML);
}

bool canTranscribe()
{
    return canRun() && hasSampleFile(jfkSample);
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
            break;
    }

    return "cpu";
}

// About three times the worst measured over both contexts and eleven steps,
// per setting: CPU 0.46, CPU and GPU 0.038, CPU and engine 0.22, `all` 0.034
// (Core ML puts `all` on the GPU), and 0.46 on the CPU with the attentions
// fused, with no token differing anywhere. The
// largest logit is 29.5, where fp16's last place is 0.016: the GPU is two or
// three of those out, and the CPU's fp16 path, the least accurate of the three
// as eacp's own step suite found, about thirty. The engine settings keep their
// own bound only under EACP_REQUIRE_ANE=1, as the encoder's do, since a
// machine without an engine runs them on the CPU.
struct Tolerance
{
    ComputeUnits units;
    double maxAbs;
};

constexpr auto cpuTolerance = 1.5;

const auto tolerances = Array<Tolerance, 4> {{
    {ComputeUnits::cpu, cpuTolerance},
    {ComputeUnits::cpuAndGPU, 0.12},
    {ComputeUnits::cpuAndNeuralEngine, 0.7},
    {ComputeUnits::all, 0.12},
}};

bool isAneRequired()
{
    const auto* value = std::getenv("EACP_REQUIRE_ANE");
    return value != nullptr && std::string {value} == "1";
}

double boundFor(const Tolerance& tolerance)
{
    const auto isEngineSetting = tolerance.units == ComputeUnits::cpuAndNeuralEngine
                                 || tolerance.units == ComputeUnits::all;

    return isEngineSetting && !isAneRequired() ? cpuTolerance : tolerance.maxAbs;
}

constexpr auto followedTokens = 10;

struct Comparison
{
    double maxAbs = 0.0;
    double largestLogit = 0.0;
    int tokenMismatches = 0;
    int steps = 0;
    double medianStepMs = 0.0;
};

const DecoderShape& tinyEnShape()
{
    static const auto shape = []
    {
        const auto config = ModelConfig::fromFile(modelFile("config.json"));
        return DecoderShape::fromConfig(config, encoderPositions);
    }();

    return shape;
}

const SafeTensors& tinyEnFile()
{
    static const auto file = SafeTensors::fromFile(modelFile("model.safetensors"));
    return file;
}

Vector<float> lastRow(const Vector<float>& rows, int rowLength)
{
    auto row = Vector<float> {};

    for (auto index = rows.size() - rowLength; index < rows.size(); ++index)
        row.add(rows[index]);

    return row;
}

double maxAbsDifference(const Vector<float>& expected, const Vector<float>& actual)
{
    auto worst = 0.0;

    for (auto index = 0; index < expected.size() && index < actual.size(); ++index)
        worst = std::max(worst, (double) std::abs(expected[index] - actual[index]));

    return expected.size() == actual.size() ? worst : INFINITY;
}

int argmaxOf(const Vector<float>& row)
{
    return (int) (std::max_element(row.begin(), row.end()) - row.begin());
}

// Which of the step's three CPU-bound ops are spelled in the engine's forms.
struct EngineForms
{
    bool token = false;
    bool position = false;
    bool selection = false;
};

// The kernels decide the path: the prompt, in one step on the kernels and in
// the sequence start on Core ML, then each token the kernels chose, fed to
// both, so a Core ML token that differs is counted rather than followed into
// a different sequence.
Comparison compare(ComputeUnits units,
                   int crossPositionCount,
                   bool fused = false,
                   const EngineForms& forms = {})
{
    const auto& shape = tinyEnShape();
    const auto encoderOutput = syntheticEncoderOutput(shape);
    const auto rowLength = shape.vocabularySize;
    const auto prompt = preparedModel().prompt();

    auto kernels = DecoderRun {shape, tinyEnFile()};
    kernels.begin(encoderOutput, crossPositionCount);

    const auto hostWeights = DecoderWeights {
        tinyEnFile(), shape, WeightPacking::Float, WeightPlacement::Host};

    auto options = CoreMLDecoderOptions {};
    options.units = units;
    options.cacheDirectory = sharedCoreMLCacheDirectory();
    options.keepsLogits = true;
    options.fusedAttention = fused;
    options.embedsTokenByProduct = forms.token;
    options.embedsPositionByProduct = forms.position;
    options.selectsOnTheEngine = forms.selection;
    options.promptLength = prompt.size();

    auto coreML = CoreMLDecoder {shape};
    const auto loaded = coreML.prepare(hostWeights, options);
    check(loaded.ok, loaded.error);

    if (!loaded.ok)
        return {};

    const auto rows = eacp::GPU::Device::shared().makeBuffer(
        encoderOutput.data(),
        encoderOutput.size() * (int) sizeof(float),
        eacp::GPU::BufferUsage::Storage);

    const auto nothingSuppressed = Vector<float> {};
    const auto noSuppression = coreML.makeSuppression(nothingSuppressed);

    auto tokens = std::vector<int> {};

    for (auto token: prompt)
        tokens.push_back(token);

    auto comparison = Comparison {};
    auto stepTimes = std::vector<double> {};
    auto next = 0;

    for (auto step = 0; step <= followedTokens; ++step)
    {
        const auto expected = lastRow(kernels.step(tokens).logits, rowLength);

        const auto start = Clock::now();
        const auto chosen =
            step == 0 ? coreML.beginSequence(
                            rows, crossPositionCount, prompt, noSuppression)
                      : coreML.step(next, noSuppression);

        if (step > 0)
            stepTimes.push_back(millisecondsSince(start));

        const auto actual = coreML.lastLogits();
        const auto kernelChoice = argmaxOf(expected);

        for (auto logit: expected)
            comparison.largestLogit =
                std::max(comparison.largestLogit, (double) std::abs(logit));

        comparison.maxAbs =
            std::max(comparison.maxAbs, maxAbsDifference(expected, actual));
        comparison.tokenMismatches += chosen == kernelChoice ? 0 : 1;
        ++comparison.steps;

        next = kernelChoice;
        tokens = {kernelChoice};
    }

    std::sort(stepTimes.begin(), stepTimes.end());
    comparison.medianStepMs = stepTimes[stepTimes.size() / 2];

    return comparison;
}

void checkEverySetting(int crossPositionCount)
{
    for (const auto& tolerance: tolerances)
    {
        const auto result = compare(tolerance.units, crossPositionCount);

        std::cout << "  decoder at " << crossPositionCount << " cross rows ["
                  << nameOf(tolerance.units) << "]: logits maxAbs " << result.maxAbs
                  << " (largest |logit| " << result.largestLogit << ")"
                  << " over " << result.steps << " steps, " << result.tokenMismatches
                  << " tokens differ, step median " << result.medianStepMs
                  << " ms (Debug)\n";

        check(result.steps == followedTokens + 1);
        const auto bound = boundFor(tolerance);

        check(result.maxAbs <= bound,
              std::string {"decoder logits under "} + nameOf(tolerance.units)
                  + ": max abs error " + std::to_string(result.maxAbs) + " exceeds "
                  + std::to_string(bound));
        check(result.tokenMismatches == 0);
    }
}

std::unique_ptr<Whisper> modelOn(EncoderBackend encoderBackend,
                                 DecoderBackend decoderBackend)
{
    auto whisper = std::make_unique<Whisper>();
    whisper->load(modelDirectory());
    whisper->setEncoderBackend(encoderBackend);
    whisper->setDecoderBackend(decoderBackend);
    whisper->setCoreMLCacheDirectory(sharedCoreMLCacheDirectory());
    whisper->prepare();

    return whisper;
}

Whisper& coreMLDecoderModel()
{
    static const auto model =
        modelOn(EncoderBackend::kernels, DecoderBackend::coreML);
    return *model;
}

Whisper& allCoreMLModel()
{
    static const auto model =
        modelOn(EncoderBackend::coreML, DecoderBackend::coreML);
    return *model;
}

bool sameTokens(const Vector<TokenId>& first, const Vector<TokenId>& second)
{
    if (first.size() != second.size())
        return false;

    for (auto index = 0; index < first.size(); ++index)
        if (first[index] != second[index])
            return false;

    return true;
}
} // namespace

auto tStepMatchesTheKernels =
    test("Whisper/CoreMLDecoder/stepMatchesTheKernelsOverTheWindow") = []
{
    if (canRun())
        checkEverySetting(encoderPositions);
};

// A shorter context: the kernels attend to the first 576 rows, and Core ML to
// all 1500 behind a mask that allows only those, the rest holding rows that
// would move every logit if the mask were ignored.
auto tStepMatchesTheKernelsAtAContext =
    test("Whisper/CoreMLDecoder/stepMatchesTheKernelsAtAShortContext") = []
{
    if (canRun())
        checkEverySetting(576);
};

// The masked attentions as eacp's fused op under an input mask, which the
// decoder leaves off by default for speed, held to the same bound: CPU only,
// at the short context, where an ignored mask would show.
auto tFusedStepMatchesTheKernels =
    test("Whisper/CoreMLDecoder/fusedStepMatchesTheKernelsAtAShortContext") = []
{
    if (!canRun() || !eacp::ML::supportsSpecification(9))
        return;

    const auto result = compare(ComputeUnits::cpu, 576, true);

    std::cout << "  fused decoder at 576 cross rows [cpu]: logits maxAbs "
              << result.maxAbs << ", " << result.tokenMismatches
              << " tokens differ\n";

    check(result.maxAbs <= cpuTolerance);
    check(result.tokenMismatches == 0);
};

// The step with its token and position embedded by one-hot products and its
// argmax out of reductions, every op the engine's, and with the token alone
// gathered, which is what Whisper builds under cpuAndNeuralEngine: the same
// logits within each setting's bound, and the same tokens, since the argmax
// is exact and gives a tie to the lowest index as reduce_argmax does. On the
// engine and on the CPU, at the short context.
auto tEngineFormsMatchTheKernels =
    test("Whisper/CoreMLDecoder/engineFormsMatchTheKernels") = []
{
    if (!canRun())
        return;

    const auto everyForm = EngineForms {true, true, true};
    const auto tokenGathered = EngineForms {false, true, true};

    for (const auto& tolerance: tolerances)
    {
        if (tolerance.units != ComputeUnits::cpuAndNeuralEngine
            && tolerance.units != ComputeUnits::cpu)
            continue;

        for (const auto& forms: {everyForm, tokenGathered})
        {
            const auto result = compare(tolerance.units, 576, false, forms);

            std::cout << "  engine forms" << (forms.token ? "" : ", token gathered")
                      << " [" << nameOf(tolerance.units) << "]: logits maxAbs "
                      << result.maxAbs << ", " << result.tokenMismatches
                      << " tokens differ, step median " << result.medianStepMs
                      << " ms (Debug)\n";

            check(result.steps == followedTokens + 1);
            check(result.maxAbs <= boundFor(tolerance));
            check(result.tokenMismatches == 0);
        }
    }
};

// Ties. With every token suppressed a score is its logit less 30000, where
// fp16 steps by 16, so the top of the vocabulary ties across many ids: both
// argmaxes see the same scores and must both give the tie to the lowest of
// them. And with all but two ids suppressed, one of those two.
auto tEngineArgmaxBreaksTiesLow =
    test("Whisper/CoreMLDecoder/engineArgmaxGivesATieToTheLowestIndex") = []
{
    if (!canRun())
        return;

    const auto& shape = tinyEnShape();
    const auto hostWeights = DecoderWeights {
        tinyEnFile(), shape, WeightPacking::Float, WeightPlacement::Host};
    const auto encoderOutput = syntheticEncoderOutput(shape);
    const auto rows = eacp::GPU::Device::shared().makeBuffer(
        encoderOutput.data(),
        encoderOutput.size() * (int) sizeof(float),
        eacp::GPU::BufferUsage::Storage);
    const auto token = (int) preparedModel().prompt()[0];

    auto everything = Vector<float> {};
    everything.resize(shape.vocabularySize, 1.0f);

    auto allButTwo = everything;
    allButTwo[300] = 0.0f;
    allButTwo[51000] = 0.0f;

    const auto choose = [&](ComputeUnits units, bool onTheEngine, const auto& mask)
    {
        auto options = CoreMLDecoderOptions {};
        options.units = units;
        options.cacheDirectory = sharedCoreMLCacheDirectory();
        options.selectsOnTheEngine = onTheEngine;

        auto decoder = CoreMLDecoder {shape};
        check(decoder.prepare(hostWeights, options).ok);

        const auto suppression = decoder.makeSuppression(mask);
        decoder.beginSequence(rows, 0, {&token, 1}, suppression);

        return decoder.step(token, suppression);
    };

    for (auto units: {ComputeUnits::cpu, ComputeUnits::cpuAndNeuralEngine})
    {
        const auto byReduction = choose(units, false, everything);
        const auto onTheEngine = choose(units, true, everything);

        std::cout << "  every token suppressed [" << nameOf(units)
                  << "]: reduce_argmax " << byReduction << ", engine argmax "
                  << onTheEngine << "\n";

        check(onTheEngine == byReduction);

        const auto ofTwo = choose(units, true, allButTwo);
        check(ofTwo == 300 || ofTwo == 51000);
        check(ofTwo == choose(units, false, allButTwo));
    }
};

auto tRefusesMisuse = test("Whisper/CoreMLDecoder/refusesMisuse") = []
{
    if (!canRun())
        return;

    const auto& shape = tinyEnShape();
    auto decoder = CoreMLDecoder {shape};

    const auto noSuppression = eacp::ML::MultiArray {};
    check(throwsLogicError([&] { decoder.step(0, noSuppression); }));

    const auto deviceWeights = DecoderWeights {tinyEnFile(), shape};
    check(throwsLogicError([&] { decoder.prepare(deviceWeights, {}); }));

    auto options = CoreMLDecoderOptions {};
    options.cacheDirectory = sharedCoreMLCacheDirectory();

    const auto hostWeights = DecoderWeights {
        tinyEnFile(), shape, WeightPacking::Float, WeightPlacement::Host};
    check(decoder.prepare(hostWeights, options).ok);

    const auto nothingSuppressed = Vector<float> {};
    const auto suppression = decoder.makeSuppression(nothingSuppressed);
    check(throwsLogicError([&] { decoder.step(0, suppression); }));
    check(throwsLogicError([&] { decoder.lastLogits(); }));

    const auto prompt = preparedModel().prompt();
    const auto rows =
        eacp::GPU::Device::shared().makeBuffer(16, eacp::GPU::BufferUsage::Storage);
    check(throwsLogicError(
        [&] { decoder.beginSequence(rows, 0, prompt, suppression); }));

    const auto encoderOutput = syntheticEncoderOutput(shape);
    const auto window = eacp::GPU::Device::shared().makeBuffer(
        encoderOutput.data(),
        encoderOutput.size() * (int) sizeof(float),
        eacp::GPU::BufferUsage::Storage);
    check(throwsLogicError(
        [&] { decoder.beginSequence(window, 0, prompt, suppression); }));

    auto small = DecoderShape {shape};
    small.crossPositions = 32;
    auto other = CoreMLDecoder {small};
    check(throwsModelError([&] { other.prepare(hostWeights, options); }));
};

// The runtime with its decoder on Core ML, the encoder on the kernels: the
// pinned sentence, the kernels' very tokens, and as many steps.
auto tJfkOnTheCoreMLDecoder = test("Whisper/CoreMLDecoder/jfkTranscribes") = []
{
    if (!canTranscribe())
        return;

    auto& whisper = coreMLDecoderModel();
    const auto samples = readWavFile(sampleFile(jfkSample));

    const auto kernelTokens = preparedModel().transcribe(samples);
    const auto tokens = whisper.transcribe(samples);

    check(whisper.decoderBackend() == DecoderBackend::coreML);
    check(trimmed(whisper.textForTokens(tokens)) == jfkTranscript);
    check(sameTokens(tokens, kernelTokens));
    check(whisper.lastStepCount() == jfkTokenCount + 1);
    check(whisper.lastSequenceStartSeconds() > 0.0);
    check(whisper.lastDecoderPredictSeconds() > 0.0);
    check(whisper.lastDecodeSeconds() > whisper.lastDecoderPredictSeconds());

    std::cout << "  jfk.wav, decoder on Core ML [cpu]: decode "
              << whisper.lastDecodeSeconds() * 1000.0 << " ms, start "
              << whisper.lastSequenceStartSeconds() * 1000.0 << " ms, steps "
              << whisper.lastDecoderPredictSeconds() * 1000.0 << " ms (Debug)\n";
};

// Both halves on Core ML, the rows handed across in fp16 without the GPU:
// the same sentence, over the window and at a context, and the asynchronous
// run lands on the blocking one's tokens.
auto tJfkAllOnCoreML = test("Whisper/CoreMLDecoder/jfkTranscribesAllOnCoreML") = []
{
    if (!canTranscribe() || !Whisper::supportsEncoderBackend(EncoderBackend::coreML))
        return;

    auto& whisper = allCoreMLModel();
    const auto samples = readWavFile(sampleFile(jfkSample));

    const auto tokens = whisper.transcribe(samples);
    check(trimmed(whisper.textForTokens(tokens)) == jfkTranscript);
    check(sameTokens(tokens, preparedModel().transcribe(samples)));

    const auto asynchronous =
        whisper.transcribeAsync(samples).waitFor(eacp::Time::MS {30000});
    check(sameTokens(asynchronous, tokens));

    auto& kernels = preparedModel();
    whisper.setAudioContext(576);
    kernels.setAudioContext(576);

    const auto reduced = whisper.transcribe(samples);
    const auto kernelReduced = kernels.transcribe(samples);

    whisper.setAudioContext(0);
    kernels.setAudioContext(0);

    check(!reduced.empty());
    check(sameTokens(reduced, kernelReduced));
};

#endif
