#include "Common.h"

#include <eacp/GPU/GPU.h>

#include <fstream>

// The choice of decoder backend, before prepare and at run time: the setters,
// the refusals, and a prepared runtime switched between the kernels and Core
// ML and back, blocking and not, with the pinned transcript on each side.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;

namespace
{
bool canSwitch()
{
    return eacp::GPU::Device::shared().isValid() && hasWhisperModel()
           && hasSampleFile(jfkSample);
}

bool canSwitchToCoreML()
{
    return canSwitch() && Whisper::supportsDecoderBackend(DecoderBackend::coreML);
}

std::unique_ptr<Whisper> kernelModelToSwitch(const eacp::FilePath& cache)
{
    auto whisper = std::make_unique<Whisper>();
    whisper->load(modelDirectory());
    whisper->setCoreMLCacheDirectory(cache);
    whisper->prepare();

    return whisper;
}

std::unique_ptr<Whisper> kernelModelToSwitch()
{
    return kernelModelToSwitch(sharedCoreMLCacheDirectory());
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

auto tDecoderBackendDefaults = test("Whisper/decoderBackendDefaults") = []
{
    auto whisper = Whisper {};

    check(whisper.decoderBackend() == DecoderBackend::kernels);
    check(whisper.decoderComputeUnits() == DecoderComputeUnits::cpu);
    check(Whisper::supportsDecoderBackend(DecoderBackend::kernels));

#if EACP_HAS_COREML
    check(Whisper::supportsDecoderBackend(DecoderBackend::coreML)
          == (eacp::ML::isSupported() && eacp::ML::supportsSpecification(8)));
#else
    check(!Whisper::supportsDecoderBackend(DecoderBackend::coreML));
#endif

    whisper.setDecoderComputeUnits(DecoderComputeUnits::cpuAndNeuralEngine);
    check(whisper.decoderComputeUnits() == DecoderComputeUnits::cpuAndNeuralEngine);

    whisper.setDecoderBackend(DecoderBackend::coreML);
    check(whisper.decoderBackend() == DecoderBackend::coreML);

    check(whisper.coreMLCacheDirectory().empty());
    whisper.setEncoderCacheDirectory(sharedCoreMLCacheDirectory());
    check(whisper.coreMLCacheDirectory() == sharedCoreMLCacheDirectory());

    check(!whisper.isSwitchingDecoder());
    check(whisper.lastSequenceStartSeconds() == 0.0);
    check(whisper.lastDecoderPredictSeconds() == 0.0);
};

auto tDecoderBackendIsChosenBeforePrepare =
    test("Whisper/decoderBackendIsChosenBeforePrepare") = []
{
    if (!eacp::GPU::Device::shared().isValid() || !hasWhisperModel())
        return;

    auto& whisper = preparedModel();

    check(throwsLogicError([&]
                           { whisper.setDecoderBackend(DecoderBackend::coreML); }));
    check(throwsLogicError(
        [&] { whisper.setDecoderComputeUnits(DecoderComputeUnits::all); }));
    check(throwsLogicError([&]
                           { whisper.setCoreMLCacheDirectory(eacp::FilePath {}); }));
    check(whisper.decoderBackend() == DecoderBackend::kernels);
    check(!whisper.decoderWasCacheHit());
    check(whisper.decoderLoadSeconds() == 0.0);
};

// Unprepared, a switch is the setter; prepared, one to the backend in use is
// nothing, and one this machine cannot make is refused with nothing changed.
auto tDecoderSwitchRefusesWhatItCannotRun =
    test("Whisper/decoderSwitchRefusesWhatItCannotRun") = []
{
    auto unprepared = Whisper {};
    unprepared.switchDecoderBackend(DecoderBackend::coreML);
    check(unprepared.decoderBackend() == DecoderBackend::coreML);

    auto resolved = unprepared.switchDecoderBackendAsync(DecoderBackend::kernels);
    check(resolved.isResolved());
    check(unprepared.decoderBackend() == DecoderBackend::kernels);

    if (!canSwitch())
        return;

    auto whisper = kernelModelToSwitch();
    whisper->switchDecoderBackend(DecoderBackend::kernels);
    check(whisper->decoderBackend() == DecoderBackend::kernels);

    if (!Whisper::supportsDecoderBackend(DecoderBackend::coreML))
    {
        check(throwsModelError(
            [&] { whisper->switchDecoderBackend(DecoderBackend::coreML); }));
        check(whisper->decoderBackend() == DecoderBackend::kernels);
    }
};

// A switch Core ML cannot complete — here, a cache directory that is a file,
// so nothing can be compiled into it — throws, and the runtime decodes on the
// kernels as if nothing had been asked, both ways of asking.
auto tAFailedDecoderSwitchChangesNothing =
    test("Whisper/aFailedDecoderSwitchChangesNothing") = []
{
    if (!canSwitchToCoreML())
        return;

    const auto blocker =
        eacp::FilePath::tempDirectory() / "whisper-eacp-tests" / "not-a-directory";
    std::filesystem::create_directories(blocker.parentDirectory().str());
    std::ofstream {blocker.str()} << "in the way";

    auto whisper = kernelModelToSwitch(blocker);
    const auto samples = readWavFile(sampleFile(jfkSample));
    const auto before = whisper->transcribe(samples);

    check(throwsModelError(
        [&] { whisper->switchDecoderBackend(DecoderBackend::coreML); }));
    check(whisper->decoderBackend() == DecoderBackend::kernels);
    check(sameTokens(whisper->transcribe(samples), before));

    auto rejected = false;
    auto switching = whisper->switchDecoderBackendAsync(DecoderBackend::coreML);
    switching.then([] {}, [&rejected](const std::string&) { rejected = true; });
    try
    {
        switching.waitFor(eacp::Time::MS {60000});
    }
    catch (const std::exception&)
    {
    }

    check(rejected);
    check(!whisper->isSwitchingDecoder());
    check(whisper->decoderBackend() == DecoderBackend::kernels);
    check(sameTokens(whisper->transcribe(samples), before));

    std::filesystem::remove(blocker.str());
};

// Kernels, Core ML, kernels, on one prepared runtime: the same tokens each
// time, and the Core ML clocks only while it is on Core ML.
auto tDecoderSwitchRoundTrip = test("Whisper/switchDecoderKernelsCoreMLKernels") = []
{
    if (!canSwitchToCoreML())
        return;

    auto whisper = kernelModelToSwitch();
    const auto samples = readWavFile(sampleFile(jfkSample));
    const auto kernelTokens = whisper->transcribe(samples);

    whisper->switchDecoderBackend(DecoderBackend::coreML);

    check(whisper->decoderBackend() == DecoderBackend::coreML);
    check(whisper->isPrepared());
    check(whisper->decoderLoadSeconds() > 0.0);
    check(sameTokens(whisper->transcribe(samples), kernelTokens));
    check(whisper->lastDecoderPredictSeconds() > 0.0);

    whisper->switchDecoderBackend(DecoderBackend::kernels);

    check(whisper->decoderBackend() == DecoderBackend::kernels);
    check(whisper->decoderLoadSeconds() == 0.0);
    check(sameTokens(whisper->transcribe(samples), kernelTokens));
    check(whisper->lastDecoderPredictSeconds() == 0.0);
};

// The switch with both loads on the models' queues. Until it settles the
// runtime is still on the kernels and refuses to run, prepare, change its
// context or switch either half; once it resolves, on the main thread, it
// decodes on Core ML.
auto tDecoderSwitchAsync = test("Whisper/switchDecoderBackendAsync") = []
{
    if (!canSwitchToCoreML())
        return;

    auto whisper = kernelModelToSwitch();
    const auto samples = readWavFile(sampleFile(jfkSample));
    const auto kernelTokens = whisper->transcribe(samples);

    auto resolvedOnMain = false;
    auto switching = whisper->switchDecoderBackendAsync(DecoderBackend::coreML);
    switching.then([&resolvedOnMain]
                   { resolvedOnMain = eacp::Threads::isMainThread(); });

    check(!switching.isReady());
    check(whisper->isSwitchingDecoder());
    check(whisper->decoderBackend() == DecoderBackend::kernels);

    check(throwsLogicError([&] { whisper->transcribe(samples); }));
    check(throwsLogicError([&] { whisper->transcribeAsync(samples); }));
    check(throwsLogicError([&] { whisper->setAudioContext(576); }));
    check(throwsLogicError([&] { whisper->prepare(); }));
    check(throwsLogicError(
        [&] { whisper->switchDecoderBackend(DecoderBackend::kernels); }));
    check(throwsLogicError(
        [&] { whisper->switchDecoderBackendAsync(DecoderBackend::coreML); }));
    check(throwsLogicError(
        [&] { whisper->switchEncoderBackend(EncoderBackend::coreML); }));

    switching.waitFor(eacp::Time::MS {120000});

    check(switching.isResolved());
    check(resolvedOnMain);
    check(!whisper->isSwitchingDecoder());
    check(whisper->decoderBackend() == DecoderBackend::coreML);
    check(sameTokens(whisper->transcribe(samples), kernelTokens));

    auto back = whisper->switchDecoderBackendAsync(DecoderBackend::kernels);

    check(back.isResolved());
    check(whisper->decoderBackend() == DecoderBackend::kernels);
    check(sameTokens(whisper->transcribe(samples), kernelTokens));
};

// A run in the air, with the encoder on Core ML, refuses a decoder switch as
// it refuses an encoder one; an encoder switch in the air refuses it too.
auto tDecoderSwitchRefusedWhileBusy =
    test("Whisper/decoderSwitchIsRefusedWhileBusy") = []
{
    if (!canSwitchToCoreML()
        || !Whisper::supportsEncoderBackend(EncoderBackend::coreML))
        return;

    auto whisper = kernelModelToSwitch();
    const auto samples = readWavFile(sampleFile(jfkSample));

    whisper->switchEncoderBackend(EncoderBackend::coreML);

    auto run = whisper->transcribeAsync(samples);
    check(whisper->isTranscribing());

    check(throwsLogicError(
        [&] { whisper->switchDecoderBackend(DecoderBackend::coreML); }));
    check(throwsLogicError(
        [&] { whisper->switchDecoderBackendAsync(DecoderBackend::coreML); }));
    check(whisper->decoderBackend() == DecoderBackend::kernels);

    const auto tokens = run.waitFor(eacp::Time::MS {30000});
    check(trimmed(whisper->textForTokens(tokens)) == jfkTranscript);

    whisper->switchEncoderBackend(EncoderBackend::kernels);
    auto encoderSwitch = whisper->switchEncoderBackendAsync(EncoderBackend::coreML);

    check(throwsLogicError(
        [&] { whisper->switchDecoderBackend(DecoderBackend::coreML); }));

    encoderSwitch.waitFor(eacp::Time::MS {120000});

    whisper->switchDecoderBackend(DecoderBackend::coreML);
    check(whisper->decoderBackend() == DecoderBackend::coreML);
    check(trimmed(whisper->transcribeText(samples)) == jfkTranscript);
};
