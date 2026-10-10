#include "Common.h"

// The mel front end recorded into a Core ML program (Net/CoreMLMel.h), run
// over four windows under each compute-unit setting and compared cell by cell
// with the kernels' fp32 mel: jfk.wav, a tone at full scale, a tone and noise
// a thousand times quieter, and silence. Apple only.
//
// The skips are CoreMLEncoderTests.cpp's: no device, no model or an OS that
// cannot run the program. EACP_REQUIRE_ANE=1 holds the engine and GPU
// settings to their own bounds, and asserts every op on the engine.

#if EACP_HAS_COREML

#include <WhisperEACP/Net/CoreMLMel.h>

#include <eacp/ML/Model/Model.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <numbers>

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;
using eacp::ML::ComputeUnits;

namespace
{
bool isAneRequired()
{
    const auto* value = std::getenv("EACP_REQUIRE_ANE");
    return value != nullptr && std::string {value} == "1";
}

bool canRun()
{
    if (!eacp::GPU::Device::shared().isValid() || !hasWhisperModel())
        return false;

    if (eacp::ML::isSupported() && eacp::ML::supportsSpecification(8))
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

const PreprocessorConfig& preprocessor()
{
    static const auto config =
        PreprocessorConfig::fromFile(modelFile("preprocessor_config.json"));
    return config;
}

Vector<float> kernelMel(const Vector<float>& samples)
{
    const auto shape = MelShape {};
    const auto filterBank = preprocessor().makeMelFilterBuffer();

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

eacp::ML::Model& melModel(ComputeUnits units)
{
    static auto models = std::map<ComputeUnits, std::unique_ptr<eacp::ML::Model>> {};

    if (auto found = models.find(units); found != models.end())
        return *found->second;

    const auto graph = recordLogMelProgram(MelShape {}, preprocessor().melFilters);
    check(graph.isValid());

    if (!graph.isValid())
        std::cout << "  " << graph.errors()[0] << "\n";

    auto options = eacp::ML::Options {};
    options.units = units;
    options.cacheDirectory = sharedCoreMLCacheDirectory();

    auto model = std::make_unique<eacp::ML::Model>();
    const auto loaded = model->load(graph.build(), options);
    check(loaded.ok);

    if (!loaded.ok)
        std::cout << "  " << nameOf(units) << ": " << loaded.error << "\n";

    return *models.emplace(units, std::move(model)).first->second;
}

// The program's frames-major mel, turned band major to sit beside the kernels'.
Vector<float> coreMLMel(eacp::ML::Model& model, const Vector<float>& samples)
{
    const auto shape = MelShape {};
    auto input = eacp::ML::MultiArray::create(logMelInputShape(shape),
                                              eacp::ML::DType::float16);
    input.fromFloats(samples);

    auto inputs = eacp::ML::Inputs {};
    inputs[samplesFeature] = input;

    auto outputs = eacp::ML::Outputs {};
    outputs[melFeature] = eacp::ML::MultiArray::create(
        {shape.frameCount, shape.melCount}, eacp::ML::DType::float16);

    const auto result = model.predict(inputs, outputs);
    check(result.ok);

    if (!result.ok)
        std::cout << "  " << result.error << "\n";

    const auto framesMajor = outputs[melFeature].toFloats();
    auto bandMajor = sized(shape.melElementCount());

    for (auto frame = 0; frame < shape.frameCount; ++frame)
        for (auto band = 0; band < shape.melCount; ++band)
            bandMajor[band * shape.frameCount + frame] =
                framesMajor[frame * shape.melCount + band];

    return bandMajor;
}

Vector<float> windowOf(const Vector<float>& recording)
{
    auto samples = sized(windowSamples);

    for (auto index = 0; index < recording.size() && index < windowSamples; ++index)
        samples[index] = recording[index];

    return samples;
}

Vector<float> tones(double loudness, double noise)
{
    auto samples = sized(windowSamples);
    auto state = 12345u;

    for (auto index = 0; index < windowSamples; ++index)
    {
        state = state * 1664525u + 1013904223u;

        const auto random = (double) (state >> 8) / (double) (1u << 24) - 0.5;
        const auto phase = 2.0 * std::numbers::pi * index / sampleRate;

        samples[index] = (float) (loudness
                                      * (0.7 * std::sin(phase * 440.0)
                                         + 0.2 * std::sin(phase * 3000.0))
                                  + noise * random);
    }

    return samples;
}

struct Tolerance
{
    double maxAbs = 0.0;
    double tailAbs = 0.0;
    double meanAbs = 0.0;
};

// Each window has two bounds, the CPU's and the engine's and GPU's, since a
// machine with no Neural Engine places everything on the CPU under every
// setting, and those two are held to their own only under EACP_REQUIRE_ANE=1.
// Measured on an M5 Max, macOS 27, against the kernels' fp32 mel, the CPU's
// again on an M4 Max, macOS 27.0.1, to the same figures, and set at about
// three times the worst setting of each pair:
//
//   jfk.wav   engine 0.059 / 0.016 / 0.00044, GPU 0.055 / 0.011 / 0.00033,
//             CPU 0.080 / 0.013 / 0.00043 (max / 99.9% / mean)
//   quiet     engine 0.030 / 0.0091 / 0.0010, GPU 0.028 / 0.0082 / 0.00069,
//             CPU 0.20 / 0.067 / 0.0025
//   tones     engine 0.21 / 0.21 / 0.00061, GPU 0.17 / 0.17 / 0.00064,
//             CPU 0.56 / 0.56 / 0.040
//
// The tones are the hard case and not the realistic one: two sinusoids and no
// noise leave the spectrum's far bins eight decades down, which fp16 does not
// resolve on any device, so what the window asserts is that full scale
// neither overflows nor moves the loud cells. Silence is exact everywhere.
struct Window
{
    const char* name;
    Vector<float> samples;
    Tolerance onTheCpu;
    Tolerance onTheEngineOrGpu;
};

// The loudest cell, which sets every floor: within one fp16 step of the
// kernels' at the top of the range.
constexpr auto peakTolerance = 0.004;

double peakOf(const Vector<float>& mel)
{
    return *std::max_element(mel.begin(), mel.end());
}

const Vector<Window>& windows()
{
    static const auto all = [&]
    {
        auto made = Vector<Window> {};

        if (hasSampleFile(jfkSample))
            made.add({"jfk.wav",
                      windowOf(readWavFile(sampleFile(jfkSample))),
                      {0.25, 0.05, 0.0015},
                      {0.2, 0.05, 0.0015}});

        made.add({"full-scale tones",
                  tones(1.0 / 0.9, 0.0),
                  {1.5, 1.5, 0.12},
                  {0.65, 0.65, 0.002}});
        made.add({"quiet tones and noise",
                  tones(1.0e-3, 1.0e-4),
                  {0.6, 0.2, 0.008},
                  {0.1, 0.03, 0.003}});
        made.add({"silence", sized(windowSamples), {}, {}});

        return made;
    }();

    return all;
}

Tolerance toleranceFor(const Window& window, ComputeUnits units)
{
    if (isAneRequired() && units != ComputeUnits::cpu)
        return window.onTheEngineOrGpu;

    return window.onTheCpu;
}

struct Agreement
{
    double maxAbs = 0.0;
    double tailAbs = 0.0;
    double meanAbs = 0.0;
};

Agreement agreement(const Vector<float>& actual, const Vector<float>& expected)
{
    auto result = Agreement {};
    auto errors = Vector<double> {};

    for (auto index = 0; index < expected.size(); ++index)
    {
        const auto error = std::abs((double) actual[index] - expected[index]);
        result.maxAbs = std::max(result.maxAbs, error);
        result.meanAbs += error;
        errors.add(error);
    }

    result.meanAbs /= std::max(1, errors.size());
    std::sort(errors.begin(), errors.end());
    result.tailAbs = errors[(int) (0.999 * (errors.size() - 1))];

    return result;
}

std::ostream& operator<<(std::ostream& stream, const Agreement& measured)
{
    return stream << "max abs " << measured.maxAbs << " (99.9% " << measured.tailAbs
                  << ", mean " << measured.meanAbs << ")";
}

constexpr auto everySetting =
    Array<ComputeUnits, 4> {ComputeUnits::cpuAndNeuralEngine,
                            ComputeUnits::all,
                            ComputeUnits::cpuAndGPU,
                            ComputeUnits::cpu};
} // namespace

auto tCoreMLMelMatchesTheKernels = test("Net/CoreMLMel/matchesTheKernels") = []
{
    if (!canRun())
        return;

    for (auto units: everySetting)
    {
        auto& model = melModel(units);

        for (const auto& window: windows())
        {
            const auto expected = kernelMel(window.samples);
            const auto actual = coreMLMel(model, window.samples);
            const auto measured = agreement(actual, expected);

            const auto tolerance = toleranceFor(window, units);

            std::cout << "  " << nameOf(units) << ", " << window.name << ": "
                      << measured << "\n";

            check(actual.size() == expected.size());
            check(measured.maxAbs <= tolerance.maxAbs);
            check(measured.tailAbs <= tolerance.tailAbs);
            check(measured.meanAbs <= tolerance.meanAbs);
            check(std::abs(peakOf(actual) - peakOf(expected)) <= peakTolerance);
        }
    }
};

auto tCoreMLMelIsAllOnTheEngine = test("Net/CoreMLMel/isAllOnTheEngine") = []
{
    if (!canRun() || !isAneRequired() || !eacp::ML::hasComputePlan())
        return;

    const auto plan = melModel(ComputeUnits::cpuAndNeuralEngine).computePlan();

    std::cout << describePlacement(plan, "  ");
    check(plan.allOn(eacp::ML::ComputePlan::Device::neuralEngine));
};

#endif
