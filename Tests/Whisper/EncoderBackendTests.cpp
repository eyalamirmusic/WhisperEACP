#include "Common.h"

#include <eacp/GPU/GPU.h>

// The choice of encoder backend and the audio contexts a fixed-shape backend is
// compiled for. None of it needs a device but the one test that prepares.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;

namespace
{
constexpr auto hopSamples = 160;
} // namespace

auto tEnumeratedAudioContexts = test("Whisper/enumeratedAudioContexts") = []
{
    constexpr auto expected = Array<int, 18> {448,
                                              512,
                                              576,
                                              640,
                                              704,
                                              768,
                                              832,
                                              896,
                                              960,
                                              1024,
                                              1088,
                                              1152,
                                              1216,
                                              1280,
                                              1344,
                                              1408,
                                              1472,
                                              1500};

    static_assert(Whisper::enumeratedAudioContextCount == 18);
    static_assert(Whisper::enumeratedAudioContexts() == expected);

    check(Whisper::enumeratedAudioContexts() == expected);

    for (auto context: expected)
        check(Whisper::isEnumeratedAudioContext(context));

    check(!Whisper::isEnumeratedAudioContext(0));
    check(!Whisper::isEnumeratedAudioContext(384));
    check(!Whisper::isEnumeratedAudioContext(600));
    check(!Whisper::isEnumeratedAudioContext(1499));
    check(!Whisper::isEnumeratedAudioContext(1536));
};

// Every recording from nothing to past the window, a hop at a time, at the
// three margins the tree uses: each count is a member, and at no margin every
// member is reached.
auto tAudioContextForSamplesIsEnumerated =
    test("Whisper/audioContextForSamplesIsEnumerated") = []
{
    const auto margins =
        Array<double, 3> {0.0, 1.0, Whisper::defaultAudioContextMargin};

    auto reached = Array<bool, Whisper::enumeratedAudioContextCount> {};
    reached.fill(false);

    for (auto margin: margins)
    {
        for (auto samples = 0; samples <= windowSamples + 4 * hopSamples;
             samples += hopSamples)
        {
            const auto context = Whisper::audioContextForSamples(samples, margin);

            check(Whisper::isEnumeratedAudioContext(context));

            if (margin == 0.0)
                for (auto index = 0; index < reached.size(); ++index)
                    if (Whisper::enumeratedAudioContexts()[index] == context)
                        reached[index] = true;
        }
    }

    for (auto wasReached: reached)
        check(wasReached);
};

auto tEncoderBackendDefaults = test("Whisper/encoderBackendDefaults") = []
{
    auto whisper = Whisper {};

    check(whisper.encoderBackend() == EncoderBackend::kernels);
    check(Whisper::supportsEncoderBackend(EncoderBackend::kernels));

#if EACP_HAS_COREML
    check(Whisper::supportsEncoderBackend(EncoderBackend::coreML)
          == (eacp::ML::isSupported() && eacp::ML::supportsSpecification(8)));
#else
    check(!Whisper::supportsEncoderBackend(EncoderBackend::coreML));
#endif

    check(whisper.encoderComputeUnits() == EncoderComputeUnits::cpuAndNeuralEngine);
    whisper.setEncoderComputeUnits(EncoderComputeUnits::cpu);
    check(whisper.encoderComputeUnits() == EncoderComputeUnits::cpu);

    whisper.setEncoderBackend(EncoderBackend::coreML);
    check(whisper.encoderBackend() == EncoderBackend::coreML);

    whisper.setEncoderBackend(EncoderBackend::kernels);
    check(whisper.encoderBackend() == EncoderBackend::kernels);
};

// The kernels take any count in the window; Core ML only the members and zero.
auto tCoreMLAudioContextIsEnumerated =
    test("Whisper/coreMLAudioContextIsEnumerated") = []
{
    auto kernels = Whisper {};
    kernels.setAudioContext(600);
    check(kernels.audioContext() == 600);

    auto coreML = Whisper {};
    coreML.setEncoderBackend(EncoderBackend::coreML);

    const auto refusal = modelErrorFrom([&] { coreML.setAudioContext(600); });
    check(mentions(refusal, "600"));
    check(mentions(refusal, "Core ML"));
    check(coreML.audioContext() == 0);

    check(throwsModelError([&] { coreML.setAudioContext(1499); }));
    check(throwsModelError([&] { coreML.setAudioContext(-1); }));

    for (auto context: Whisper::enumeratedAudioContexts())
    {
        coreML.setAudioContext(context);
        check(coreML.audioContext() == context);
    }

    coreML.setAudioContext(0);
    check(coreML.audioContext() == 0);
};

// prepare() checks what setAudioContext could not, since the context may have
// been set before the backend was, and refuses a backend it cannot run. Both
// before anything reaches the device.
auto tPrepareRefusesAnUnrunnableEncoder =
    test("Whisper/prepareRefusesAnUnrunnableEncoder") = []
{
    if (!hasWhisperModel())
        return;

    auto early = Whisper {};
    early.load(modelDirectory());
    early.setAudioContext(600);
    early.setEncoderBackend(EncoderBackend::coreML);

    const auto contextRefusal = modelErrorFrom([&] { early.prepare(); });
    check(mentions(contextRefusal, "600"));
    check(!early.isPrepared());

    if (Whisper::supportsEncoderBackend(EncoderBackend::coreML))
        return;

    auto unsupported = Whisper {};
    unsupported.load(modelDirectory());
    unsupported.setEncoderBackend(EncoderBackend::coreML);

    const auto backendRefusal = modelErrorFrom([&] { unsupported.prepare(); });
    check(mentions(backendRefusal, "Core ML"));
    check(!unsupported.isPrepared());
};

auto tEncoderBackendIsChosenBeforePrepare =
    test("Whisper/encoderBackendIsChosenBeforePrepare") = []
{
    if (!eacp::GPU::Device::shared().isValid() || !hasWhisperModel())
        return;

    auto& whisper = preparedModel();

    check(throwsLogicError([&]
                           { whisper.setEncoderBackend(EncoderBackend::coreML); }));
    check(whisper.encoderBackend() == EncoderBackend::kernels);

    check(throwsLogicError(
        [&] { whisper.setEncoderComputeUnits(EncoderComputeUnits::all); }));
    check(!whisper.encoderWasCacheHit());
    check(whisper.lastEncoderPredictSeconds() == 0.0);
};

auto tEncoderCacheDirectoryIsChosenBeforePrepare =
    test("Whisper/encoderCacheDirectoryIsChosenBeforePrepare") = []
{
    auto unprepared = Whisper {};
    check(unprepared.encoderCacheDirectory().empty());

    unprepared.setEncoderCacheDirectory(sharedCoreMLCacheDirectory());
    check(unprepared.encoderCacheDirectory() == sharedCoreMLCacheDirectory());

    if (!eacp::GPU::Device::shared().isValid() || !hasWhisperModel())
        return;

    check(throwsLogicError(
        [&] { preparedModel().setEncoderCacheDirectory(eacp::FilePath {}); }));
};

namespace
{
// tiny.en's config.json with the mel bins large-v3 asks for, which the
// front-end refuses at prepare.
Vector<std::uint8_t> configWithMelBins(int bins)
{
    const auto& original = modelFileBytes().config;
    auto text = std::string {original.begin(), original.end()};

    const auto field = std::string {"\"num_mel_bins\": 80"};
    const auto at = text.find(field);
    check(at != std::string::npos);

    if (at != std::string::npos)
        text.replace(at, field.size(), "\"num_mel_bins\": " + std::to_string(bins));

    auto bytes = Vector<std::uint8_t> {};

    for (auto character: text)
        bytes.add((std::uint8_t) character);

    return bytes;
}

// A prepare, then a load whose second prepare throws: what is left must be
// unprepared rather than the first prepare's decoder with no encoder behind it.
void checkAFailedPrepareUnprepares(EncoderBackend backend)
{
    auto whisper = Whisper {};
    whisper.load(modelFileBytes().files());
    whisper.setEncoderBackend(backend);
    whisper.setEncoderCacheDirectory(sharedCoreMLCacheDirectory());
    whisper.prepare();
    check(whisper.isPrepared());

    const auto config = configWithMelBins(128);
    auto files = modelFileBytes().files();
    files.config = config;
    whisper.load(files);

    const auto refusal = modelErrorFrom([&] { whisper.prepare(); });
    check(mentions(refusal, "128"));
    check(!whisper.isPrepared());
    check(!whisper.encoderWasCacheHit());
    check(throwsModelError([&] { whisper.transcribe(Span<const float> {}); }));

#if EACP_HAS_COREML
    check(throwsLogicError([&] { whisper.encoderComputePlan(); }));
#endif
}
} // namespace

auto tAFailedPrepareLeavesItUnprepared =
    test("Whisper/aFailedPrepareLeavesItUnprepared") = []
{
    if (!eacp::GPU::Device::shared().isValid() || !hasWhisperModel())
        return;

    checkAFailedPrepareUnprepares(EncoderBackend::kernels);

    if (Whisper::supportsEncoderBackend(EncoderBackend::coreML))
        checkAFailedPrepareUnprepares(EncoderBackend::coreML);
};

// On the kernels transcribeAsync is transcribe(), resolved before it returns.
auto tTranscribeAsyncOnTheKernels =
    test("Whisper/transcribeAsyncResolvesAtOnceOnTheKernels") = []
{
    if (!eacp::GPU::Device::shared().isValid() || !hasWhisperModel()
        || !hasSampleFile(jfkSample))
        return;

    auto& whisper = preparedModel();
    const auto samples = readWavFile(sampleFile(jfkSample));

    auto run = whisper.transcribeAsync(samples);

    check(run.isResolved());
    check(!whisper.isTranscribing());

    const auto tokens = run.waitFor(eacp::Time::MS {0});
    check(trimmed(whisper.textForTokens(tokens)) == jfkTranscript);
    check(whisper.lastEncoderMelSeconds() == 0.0);
};

// Under Core ML the run is in the air when the call returns: the runtime
// refuses a second run and a new context until it lands, and what lands is
// the blocking run's transcript, with the decode clocked as before.
auto tTranscribeAsyncOnCoreML = test("Whisper/transcribeAsyncOnCoreML") = []
{
    if (!eacp::GPU::Device::shared().isValid() || !hasWhisperModel()
        || !hasSampleFile(jfkSample)
        || !Whisper::supportsEncoderBackend(EncoderBackend::coreML))
        return;

    auto& whisper = coreMLPreparedModel();
    const auto samples = readWavFile(sampleFile(jfkSample));

    const auto blocking = whisper.transcribe(samples);
    check(whisper.lastEncoderMelSeconds() > 0.0);
    check(whisper.lastEncoderPredictSeconds() > 0.0);
    check(whisper.lastEncodeSeconds()
          > whisper.lastEncoderMelSeconds() + whisper.lastEncoderPredictSeconds());

    auto run = whisper.transcribeAsync(samples);

    check(!run.isReady());
    check(whisper.isTranscribing());
    check(throwsLogicError([&] { whisper.transcribe(samples); }));
    check(throwsLogicError([&] { whisper.transcribeAsync(samples); }));
    check(throwsLogicError([&] { whisper.setAudioContext(576); }));
    check(throwsLogicError([&] { whisper.prepare(); }));
    check(whisper.isPrepared());

    const auto tokens = run.waitFor(eacp::Time::MS {30000});

    check(!whisper.isTranscribing());
    check(tokens.size() == blocking.size());

    for (auto index = 0; index < tokens.size() && index < blocking.size(); ++index)
        check(tokens[index] == blocking[index]);

    check(trimmed(whisper.textForTokens(tokens)) == jfkTranscript);
    check(whisper.lastDecodeSeconds() > 0.0);
    check(whisper.lastStepCount() == jfkTokenCount + 1);

    whisper.setAudioContext(576);
    const auto reduced = whisper.transcribe(samples);
    const auto reducedAsync =
        whisper.transcribeAsync(samples).waitFor(eacp::Time::MS {30000});
    whisper.setAudioContext(0);

    check(reduced.size() == reducedAsync.size());

    for (auto index = 0; index < reduced.size() && index < reducedAsync.size();
         ++index)
        check(reduced[index] == reducedAsync[index]);
};
