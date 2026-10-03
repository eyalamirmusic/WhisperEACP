#include "Common.h"

#include <eacp/GPU/GPU.h>

#include <iostream>

// Everything on the Neural Engine: Whisper::setNeuralEngineThroughout(), the
// mel on Core ML in front of the encoder (EncoderBackend::coreMLWithMel) and
// the decoder on Core ML under cpuAndNeuralEngine. The preset, the pinned
// transcript against the kernels' tokens, where Core ML placed each of the
// four models, and the switches that reach the same configuration at run
// time: the encoder to and from coreMLWithMel, and the decoder between its
// units.
//
// Every test skips without a device, without the model or the sample, or
// where the OS cannot run the programs. The placement is read only under
// EACP_REQUIRE_ANE=1, as elsewhere, since the CI runner has no engine.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;

namespace
{
bool canRunOnCoreML()
{
    return eacp::GPU::Device::shared().isValid() && hasWhisperModel()
           && hasSampleFile(jfkSample)
           && Whisper::supportsEncoderBackend(EncoderBackend::coreMLWithMel)
           && Whisper::supportsDecoderBackend(DecoderBackend::coreML);
}

bool isAneRequired()
{
    const auto* value = std::getenv("EACP_REQUIRE_ANE");
    return value != nullptr && std::string {value} == "1";
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

Whisper& neuralEngineModel()
{
    static const auto model = []
    {
        auto whisper = std::make_unique<Whisper>();
        whisper->load(modelDirectory());
        whisper->setNeuralEngineThroughout();
        whisper->setCoreMLCacheDirectory(sharedCoreMLCacheDirectory());
        whisper->prepare();
        return whisper;
    }();

    return *model;
}

std::unique_ptr<Whisper> kernelModelToSwitch()
{
    auto whisper = std::make_unique<Whisper>();
    whisper->load(modelDirectory());
    whisper->setCoreMLCacheDirectory(sharedCoreMLCacheDirectory());
    whisper->prepare();

    return whisper;
}

const Vector<float>& jfkSamples()
{
    static const auto samples = readWavFile(sampleFile(jfkSample));
    return samples;
}

const Vector<TokenId>& kernelTokens()
{
    static const auto tokens = preparedModel().transcribe(jfkSamples());
    return tokens;
}

#if EACP_HAS_COREML
int countOffTheEngine(const eacp::ML::ComputePlan& plan, const std::string& type)
{
    auto count = 0;

    for (const auto& op: plan.ops)
        if (op.device != eacp::ML::ComputePlan::Device::neuralEngine)
            count += op.type == type ? 1 : 0;

    return count;
}

int countOffTheEngine(const eacp::ML::ComputePlan& plan)
{
    auto count = 0;

    for (const auto& op: plan.ops)
        count += op.device != eacp::ML::ComputePlan::Device::neuralEngine ? 1 : 0;

    return count;
}
#endif
} // namespace

auto tPresetSetsTheFour = test("Whisper/NeuralEngine/presetSetsTheFour") = []
{
    auto whisper = Whisper {};
    check(!whisper.isOnNeuralEngineThroughout());

    whisper.setNeuralEngineThroughout();

    check(whisper.isOnNeuralEngineThroughout());
    check(whisper.encoderBackend() == EncoderBackend::coreMLWithMel);
    check(whisper.encoderComputeUnits() == EncoderComputeUnits::cpuAndNeuralEngine);
    check(whisper.decoderBackend() == DecoderBackend::coreML);
    check(whisper.decoderComputeUnits() == DecoderComputeUnits::cpuAndNeuralEngine);

    check(isOnCoreML(EncoderBackend::coreMLWithMel));
    check(!isOnCoreML(EncoderBackend::kernels));
    check(Whisper::supportsEncoderBackend(EncoderBackend::coreMLWithMel)
          == Whisper::supportsEncoderBackend(EncoderBackend::coreML));

    // A context outside the enumerated set is refused as on coreML.
    check(throwsModelError([&] { whisper.setAudioContext(600); }));

    whisper.setDecoderComputeUnits(DecoderComputeUnits::cpu);
    check(!whisper.isOnNeuralEngineThroughout());

    if (!eacp::GPU::Device::shared().isValid() || !hasWhisperModel())
        return;

    check(throwsLogicError([&] { preparedModel().setNeuralEngineThroughout(); }));
};

// The pinned sentence, the kernels' very tokens, and the clocks of a run that
// put nothing on the GPU: the mel's prediction and the encoder's apart, and
// the decoder's. Then the asynchronous run, and a shorter context.
auto tJfkOnTheEngine = test("Whisper/NeuralEngine/jfkTranscribes") = []
{
    if (!canRunOnCoreML())
        return;

    auto& whisper = neuralEngineModel();
    const auto tokens = whisper.transcribe(jfkSamples());

    check(whisper.isOnNeuralEngineThroughout());
    check(trimmed(whisper.textForTokens(tokens)) == jfkTranscript);
    check(sameTokens(tokens, kernelTokens()));
    check(whisper.lastStepCount() == jfkTokenCount + 1);
    check(whisper.lastEncoderMelSeconds() > 0.0);
    check(whisper.lastEncoderPredictSeconds() > 0.0);
    check(whisper.lastDecoderPredictSeconds() > 0.0);

    std::cout << "  jfk.wav on the engine: encode "
              << whisper.lastEncodeSeconds() * 1000.0 << " ms (mel "
              << whisper.lastEncoderMelSeconds() * 1000.0 << ", encoder "
              << whisper.lastEncoderPredictSeconds() * 1000.0 << "), decode "
              << whisper.lastDecodeSeconds() * 1000.0 << " ms (Debug)\n";

    const auto asynchronous =
        whisper.transcribeAsync(jfkSamples()).waitFor(eacp::Time::MS {30000});
    check(sameTokens(asynchronous, tokens));
    check(whisper.lastEncoderMelSeconds() > 0.0);

    auto& kernels = preparedModel();
    whisper.setAudioContext(576);
    kernels.setAudioContext(576);

    const auto reduced = whisper.transcribe(jfkSamples());
    const auto kernelReduced = kernels.transcribe(jfkSamples());

    whisper.setAudioContext(0);
    kernels.setAudioContext(0);

    check(!reduced.empty());
    check(sameTokens(reduced, kernelReduced));
};

#if EACP_HAS_COREML
// Every op of the mel, the encoder and the sequence start on the engine, and
// of the step every one but the token's gather, the one op left on the CPU on
// purpose (Whisper::decoderComputeUnits()).
auto tPlacedOnTheEngine = test("Whisper/NeuralEngine/isPlacedOnTheEngine") = []
{
    if (!isAneRequired() || !canRunOnCoreML() || !eacp::ML::hasComputePlan())
        return;

    auto& whisper = neuralEngineModel();

    const auto mel = whisper.melComputePlan();
    const auto& encoder = whisper.encoderComputePlan();
    const auto start = whisper.sequenceStartComputePlan();
    const auto step = whisper.decoderStepComputePlan();

    std::cout << "  mel\n"
              << describePlacement(mel, "    ") << "  encoder\n"
              << describePlacement(encoder, "    ") << "  sequence start\n"
              << describePlacement(start, "    ") << "  step\n"
              << describePlacement(step, "    ");

    using Device = eacp::ML::ComputePlan::Device;

    check(mel.allOn(Device::neuralEngine));
    check(encoder.allOn(Device::neuralEngine));
    check(start.allOn(Device::neuralEngine));
    check(countOffTheEngine(step) == 1);
    check(countOffTheEngine(step, "gather") == 1);
};
#endif

// A prepared kernel runtime moved to the mel on Core ML, to the mel on the GPU
// with the encoder on Core ML, and back, the transcript the kernels' at each.
auto tSwitchToTheMelOnCoreML =
    test("Whisper/switchEncoderToTheMelOnCoreMLAndBack") = []
{
    if (!canRunOnCoreML())
        return;

    auto whisper = kernelModelToSwitch();

    whisper->switchEncoderBackend(EncoderBackend::coreMLWithMel);
    check(whisper->encoderBackend() == EncoderBackend::coreMLWithMel);
    check(sameTokens(whisper->transcribe(jfkSamples()), kernelTokens()));
    check(whisper->lastEncoderMelSeconds() > 0.0);

    whisper->switchEncoderBackend(EncoderBackend::coreML);
    check(whisper->encoderBackend() == EncoderBackend::coreML);
    check(sameTokens(whisper->transcribe(jfkSamples()), kernelTokens()));
    check(throwsLogicError([&] { whisper->melComputePlan(); }));

    auto pending = whisper->switchEncoderBackendAsync(EncoderBackend::coreMLWithMel);
    check(whisper->isSwitchingEncoder());
    pending.waitFor(eacp::Time::MS {60000});
    check(!whisper->isSwitchingEncoder());
    check(whisper->encoderBackend() == EncoderBackend::coreMLWithMel);
    check(sameTokens(whisper->transcribe(jfkSamples()), kernelTokens()));

    whisper->switchEncoderBackend(EncoderBackend::kernels);
    check(whisper->encoderBackend() == EncoderBackend::kernels);
    check(sameTokens(whisper->transcribe(jfkSamples()), kernelTokens()));
};

// The decoder's units moved at run time: onto the engine and back to the CPU,
// each a rebuild, and the units already in use nothing at all.
auto tSwitchDecoderUnits = test("Whisper/switchDecoderBetweenUnits") = []
{
    if (!canRunOnCoreML())
        return;

    auto whisper = kernelModelToSwitch();

    whisper->switchDecoderBackend(DecoderBackend::coreML,
                                  DecoderComputeUnits::cpuAndNeuralEngine);
    check(whisper->decoderBackend() == DecoderBackend::coreML);
    check(whisper->decoderComputeUnits() == DecoderComputeUnits::cpuAndNeuralEngine);
    check(sameTokens(whisper->transcribe(jfkSamples()), kernelTokens()));

    auto pending = whisper->switchDecoderBackendAsync(DecoderBackend::coreML,
                                                      DecoderComputeUnits::cpu);
    check(whisper->isSwitchingDecoder());
    pending.waitFor(eacp::Time::MS {60000});
    check(!whisper->isSwitchingDecoder());
    check(whisper->decoderComputeUnits() == DecoderComputeUnits::cpu);
    check(sameTokens(whisper->transcribe(jfkSamples()), kernelTokens()));

    auto nothingNew = whisper->switchDecoderBackendAsync(DecoderBackend::coreML,
                                                         DecoderComputeUnits::cpu);
    check(nothingNew.isResolved());

    whisper->switchDecoderBackend(DecoderBackend::kernels,
                                  DecoderComputeUnits::cpuAndNeuralEngine);
    check(whisper->decoderBackend() == DecoderBackend::kernels);
    check(sameTokens(whisper->transcribe(jfkSamples()), kernelTokens()));
};
