#include "Common.h"

#include <eacp/GPU/GPU.h>

// Core ML refused where the CPU would trap: before macOS 27, on a machine with
// no Neural Engine, the encoder's enumerated shapes under cpuAndNeuralEngine
// are the CPU's alone. eacp gives no way to make this machine that one, so the
// platform is replaced with one that answers as it would, and the refusal has
// to arrive from the setters and the switches, before anything is built, with
// the kernels running as before.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;

namespace
{
constexpr auto everyUnits =
    Array<CoreMLComputeUnits, 4> {CoreMLComputeUnits::all,
                                  CoreMLComputeUnits::cpuAndNeuralEngine,
                                  CoreMLComputeUnits::cpuAndGPU,
                                  CoreMLComputeUnits::cpu};

CoreMLPlatform platformBeforeMacOS27(bool hasNeuralEngine)
{
    return {.enumeratedShapesRunOnTheCpu = [] { return false; },
            .isCpuOnly =
                [hasNeuralEngine](CoreMLComputeUnits units)
            {
                return units == CoreMLComputeUnits::cpu
                       || (units == CoreMLComputeUnits::cpuAndNeuralEngine
                           && !hasNeuralEngine);
            }};
}

CoreMLPlatform noEngineBeforeMacOS27()
{
    return platformBeforeMacOS27(false);
}

bool runsOnCoreMLAtAll()
{
    return Whisper::supportsEncoderBackend(EncoderBackend::coreML,
                                           EncoderComputeUnits::all);
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

auto tEnumeratedShapesTrap = test("Whisper/Refusal/enumeratedShapesTrap") = []
{
    static_assert(!enumeratedShapesTrap(true, true));
    static_assert(!enumeratedShapesTrap(true, false));
    static_assert(enumeratedShapesTrap(false, true));
    static_assert(!enumeratedShapesTrap(false, false));

    for (auto units: everyUnits)
    {
        const auto isCpu = units == CoreMLComputeUnits::cpu;
        const auto isEngine = units == CoreMLComputeUnits::cpuAndNeuralEngine;

        check(enumeratedShapesTrap(units, noEngineBeforeMacOS27())
              == (isCpu || isEngine));
        check(enumeratedShapesTrap(units, platformBeforeMacOS27(true)) == isCpu);

        auto fromMacOS27 = noEngineBeforeMacOS27();
        fromMacOS27.enumeratedShapesRunOnTheCpu = [] { return true; };
        check(!enumeratedShapesTrap(units, fromMacOS27));
    }

    check(!Whisper::supportsEncoderBackend(EncoderBackend::coreML,
                                           EncoderComputeUnits::cpuAndNeuralEngine,
                                           noEngineBeforeMacOS27()));
    check(!Whisper::supportsEncoderBackend(EncoderBackend::coreMLWithMel,
                                           EncoderComputeUnits::cpuAndNeuralEngine,
                                           noEngineBeforeMacOS27()));
    check(!Whisper::supportsEncoderBackend(
        EncoderBackend::coreML, EncoderComputeUnits::cpu, noEngineBeforeMacOS27()));
    check(Whisper::supportsEncoderBackend(
        EncoderBackend::kernels, EncoderComputeUnits::cpu, noEngineBeforeMacOS27()));
    check(Whisper::supportsEncoderBackend(EncoderBackend::coreML,
                                          EncoderComputeUnits::cpuAndGPU,
                                          noEngineBeforeMacOS27())
          == runsOnCoreMLAtAll());
    check(Whisper::supportsDecoderBackend(DecoderBackend::coreML)
          == runsOnCoreMLAtAll());

    if (runsOnCoreMLAtAll())
        check(mentions(
            Whisper::encoderBackendRefusal(EncoderBackend::coreML,
                                           EncoderComputeUnits::cpuAndNeuralEngine,
                                           noEngineBeforeMacOS27()),
            "macOS 27"));
};

// Every setter that would choose the trap throws before it changes anything,
// whatever order the backend and the units are chosen in.
auto tSettersRefuseTheTrap = test("Whisper/Refusal/settersRefuseTheTrap") = []
{
    auto whisper = Whisper {};
    whisper.setCoreMLPlatform(noEngineBeforeMacOS27());

    const auto refusal =
        modelErrorFrom([&] { whisper.setEncoderBackend(EncoderBackend::coreML); });
    check(mentions(refusal, "Core ML"));
    check(throwsModelError(
        [&] { whisper.setEncoderBackend(EncoderBackend::coreMLWithMel); }));
    check(throwsModelError(
        [&] { whisper.switchEncoderBackend(EncoderBackend::coreML); }));
    check(throwsModelError([&] { whisper.setNeuralEngineThroughout(); }));

    check(whisper.encoderBackend() == EncoderBackend::kernels);
    check(whisper.encoderComputeUnits() == EncoderComputeUnits::cpuAndNeuralEngine);
    check(whisper.decoderBackend() == DecoderBackend::kernels);
    check(!whisper.isOnNeuralEngineThroughout());

    if (!runsOnCoreMLAtAll())
        return;

    whisper.setDecoderComputeUnits(DecoderComputeUnits::cpuAndNeuralEngine);
    whisper.setDecoderBackend(DecoderBackend::coreML);
    check(whisper.decoderBackend() == DecoderBackend::coreML);

    whisper.setEncoderComputeUnits(EncoderComputeUnits::all);
    whisper.setEncoderBackend(EncoderBackend::coreMLWithMel);
    check(whisper.encoderBackend() == EncoderBackend::coreMLWithMel);

    check(throwsModelError(
        [&] { whisper.setEncoderComputeUnits(EncoderComputeUnits::cpu); }));
    check(throwsModelError([&] { whisper.setNeuralEngineThroughout(); }));
    check(whisper.encoderComputeUnits() == EncoderComputeUnits::all);
    check(whisper.encoderBackend() == EncoderBackend::coreMLWithMel);
};

// The run the plan asks for: Core ML refused up front, the kernels prepared
// instead, and jfk.wav the kernels' very tokens; then a switch at run time
// refused the same way, the model still Ready on the kernels.
auto tFallsToTheKernels = test("Whisper/Refusal/fallsToTheKernels") = []
{
    if (!eacp::GPU::Device::shared().isValid() || !hasWhisperModel()
        || !hasSampleFile(jfkSample))
        return;

    const auto samples = readWavFile(sampleFile(jfkSample));
    const auto reference = preparedModel().transcribe(samples);
    check(trimmed(preparedModel().textForTokens(reference)) == jfkTranscript);

    auto whisper = Whisper {};
    whisper.load(modelDirectory());
    whisper.setCoreMLPlatform(noEngineBeforeMacOS27());
    whisper.setCoreMLCacheDirectory(sharedCoreMLCacheDirectory());

    check(throwsModelError([&]
                           { whisper.setEncoderBackend(EncoderBackend::coreML); }));
    check(throwsModelError([&] { whisper.setNeuralEngineThroughout(); }));

    whisper.prepare();
    check(whisper.encoderBackend() == EncoderBackend::kernels);

    const auto tokens = whisper.transcribe(samples);
    check(sameTokens(tokens, reference));
    check(trimmed(whisper.textForTokens(tokens)) == jfkTranscript);

    const auto refusal = modelErrorFrom(
        [&] { whisper.switchEncoderBackend(EncoderBackend::coreML); });
    check(mentions(refusal, "Core ML"));
    check(throwsModelError(
        [&] { whisper.switchEncoderBackendAsync(EncoderBackend::coreML); }));
    check(throwsModelError(
        [&] { whisper.switchEncoderBackendAsync(EncoderBackend::coreMLWithMel); }));

    check(whisper.isPrepared());
    check(!whisper.isSwitchingEncoder());
    check(whisper.encoderBackend() == EncoderBackend::kernels);
    check(!whisper.encoderWasCacheHit());
    check(sameTokens(whisper.transcribe(samples), reference));
};

// A platform replaced after the backend was chosen still stops prepare, before
// anything is built.
auto tPrepareRefusesTheTrap = test("Whisper/Refusal/prepareRefusesTheTrap") = []
{
    if (!hasWhisperModel() || !runsOnCoreMLAtAll()
        || !Whisper::supportsEncoderBackend(EncoderBackend::coreML))
        return;

    auto whisper = Whisper {};
    whisper.load(modelDirectory());
    whisper.setEncoderBackend(EncoderBackend::coreML);
    whisper.setCoreMLPlatform(noEngineBeforeMacOS27());

    const auto refusal = modelErrorFrom([&] { whisper.prepare(); });
    check(mentions(refusal, "macOS 27"));
    check(!whisper.isPrepared());
};
