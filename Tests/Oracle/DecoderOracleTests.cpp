#include "Common.h"

#include <WhisperEACP/Decoder/Decoder.h>
#include <WhisperEACP/Encoder/Encoder.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

// The comparison the hook in Common.h was written for: our mel through *our*
// encoder and *our* decoder against whisper.cpp's logits for the same mel and
// the same prompt.
//
// This is the deepest stage-level assertion the oracle can make. Everything
// between a mel and a logit is ours on one side and the reference's on the
// other — two encoders, two decoders, and two sets of weights, since the GGML
// file is an fp16 conversion of the fp32 safetensors. That last one is why the
// number to read here is the argmax agreement rather than the residual: the
// residual is dominated by the conversion, and CLAUDE.md says to rule that out
// first rather than to explain it afterwards.
//
// Skips without a device, without the HuggingFace files, or without the GGML
// model.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;
using namespace eacp::GPU;

namespace
{
constexpr auto configName = "config.json";
constexpr auto weightsName = "model.safetensors";

// The prompt a real run opens with: <|startoftranscript|> then
// <|notimestamps|>. Written out rather than read from the tokenizer because
// this file hands the same two numbers to both implementations, and a shared
// lookup could not say they were the same.
constexpr auto startOfTranscript = 50257;
constexpr auto noTimestamps = 50362;

// How many steps past the prompt to carry, each one fed the token the
// *reference* chose so the two sequences cannot drift apart on a disagreement
// and then be compared over different histories.
constexpr auto followOnSteps = 4;

// Room for the longest step either test takes, which is the prompt.
constexpr auto logitRowCapacity = 4;

bool canRun()
{
    return Device::shared().isValid() && hasGgmlModel() && hasModelFile(configName)
           && hasModelFile(weightsName) && hasModelFile(preprocessorFile);
}

// The whole of our own model above the mel, driven by hand: the encoder over
// one mel, then a decoder sequence whose steps are the caller's.
class OurDecoding
{
public:
    OurDecoding()
        : config(ModelConfig::fromFile(modelFile(configName)))
        , file(SafeTensors::fromFile(modelFile(weightsName)))
        , encoderShape(EncoderShape::fromConfig(config, config.encoderInputFrames()))
        , decoderShape(DecoderShape::fromConfig(config, encoderShape.positions()))
        , encoder(encoderShape)
        , decoder(decoderShape)
        , encoderWeights(file, encoderShape)
        , decoderWeights(file, decoderShape)
    {
        auto& device = Device::shared();

        encoder.prepare(device);
        decoder.prepare(device);

        encoded.emplace(melTest::allocate(device, encoderShape.elementCount()));
        logits.emplace(melTest::allocate(
            device, logitRowCapacity * decoderShape.logitElementCount()));
    }

    int vocabularySize() const { return decoderShape.logitElementCount(); }

    // The encoder over the caller's mel, then a fresh decoder sequence over its
    // output. One command buffer, committed, which is what the runtime does.
    void beginFrom(const std::vector<float>& mel, int audioContext = 0)
    {
        auto& device = Device::shared();
        const auto melBuffer = melTest::upload(device, mel);

        auto commands = device.makeCommandBuffer();

        {
            auto pass = commands.beginCompute();
            encoder.encode(pass, melBuffer, encoderWeights, *encoded, audioContext);
            decoder.beginSequence(pass, *encoded, decoderWeights, audioContext);
        }

        commands.commit();
    }

    // Appends the tokens and returns the last row of their logits, which is the
    // distribution over the token that follows them — the same row
    // oracleLogitsForPrompt returns.
    std::vector<float> step(const std::vector<int>& tokens)
    {
        auto& device = Device::shared();

        auto ids = std::vector<std::uint32_t>(tokens.size());

        for (auto index = std::size_t {}; index < tokens.size(); ++index)
            ids[index] = (std::uint32_t) tokens[index];

        const auto tokenBuffer =
            device.makeBuffer(ids.data(),
                              (int) (ids.size() * sizeof(std::uint32_t)),
                              BufferUsage::Storage);

        auto commands = device.makeCommandBuffer();

        {
            auto pass = commands.beginCompute();
            decoder.step(
                pass, tokenBuffer, (int) tokens.size(), decoderWeights, *logits);
        }

        commands.commit();

        const auto width = vocabularySize();
        const auto rows = melTest::download(*logits, (int) tokens.size() * width);

        return {rows.end() - width, rows.end()};
    }

private:
    ModelConfig config;
    SafeTensors file;
    EncoderShape encoderShape;
    DecoderShape decoderShape;
    Encoder encoder;
    Decoder decoder;
    EncoderWeights encoderWeights;
    DecoderWeights decoderWeights;
    std::optional<Buffer> encoded;
    std::optional<Buffer> logits;
};

// One of these for the module: preparing compiles every kernel and uploads
// every tensor of a 151 MB file, and neither test below is about that.
OurDecoding& ourDecoding()
{
    static const auto decoding = std::make_unique<OurDecoding>();
    return *decoding;
}

float rootMeanSquareDifference(const std::vector<float>& a,
                               const std::vector<float>& b)
{
    auto total = 0.0;

    for (auto index = std::size_t {}; index < a.size() && index < b.size(); ++index)
    {
        const auto difference = (double) a[index] - (double) b[index];
        total += difference * difference;
    }

    return (float) std::sqrt(total / (double) a.size());
}

float largestMagnitude(const std::vector<float>& values)
{
    auto worst = 0.0f;

    for (auto value: values)
        worst = std::max(worst, std::abs(value));

    return worst;
}
} // namespace

// jfk.wav, zero-filled to the window so both front-ends see the same 3000
// frames, through our mel, our encoder and our decoder — against whisper.cpp
// given that same mel.
//
// **What the tolerance is.** Not a number picked to pass: the test measures how
// far the reference's own logit row moves when every mel cell is nudged by one
// ulp (oracleLogitResolution, 0.0405 for tiny.en), and states our residual as a
// multiple of that. Measured here it is 0.469 over logits ranging to 20.8, or
// 11.6x the reference's own resolution, with an rms of 0.268. The bound asserted
// is 25x — twice the measured ratio, which is the headroom MelOracleTests gives
// its own comparison.
//
// **Whose residual it is.** The GGML conversion's, and that is not a guess.
// CLAUDE.md says to rule the format conversion out first, and it already is:
// Tests/Decoder puts this decoder within 5.4e-5 of a double-precision scalar
// reference written from HuggingFace's own definition, driven by the same
// tiny.en weights, so our side of this comparison is exact to four more digits
// than the disagreement. What is left is the reference's fp16 weights through
// four encoder and four decoder layers.
//
// The bound is therefore deliberately loose on the residual and tight on the
// argmax. A logit row 51864 wide whose largest entry agrees, after eight layers
// of independently written arithmetic, is the assertion with the information in
// it; the residual is reported so the number is on the record and a regression
// in it is visible.
auto tDecoderAgreesWithTheOracle = test("Oracle/decoderAgreesWithTheOracle") = []
{
    if (!canRun())
        return;

    const auto context = loadOracle();

    if (context == nullptr)
        return;

    const auto samples = paddedToWindow(readPcm16Wav(WHISPER_EACP_JFK_WAV));

    if (samples.empty())
        return;

    const auto mel = ourMelSpectrogram(Device::shared(), samples);
    const auto prompt = std::vector<whisper_token> {startOfTranscript, noTimestamps};

    const auto reference = oracleLogitsForMel(*context, mel, prompt);
    check(!reference.empty());

    auto& ours = ourDecoding();
    ours.beginFrom(mel);
    const auto mine = ours.step({startOfTranscript, noTimestamps});

    check((int) mine.size() == (int) reference.size());
    check((int) mine.size() == ours.vocabularySize());

    for (auto value: mine)
        check(std::isfinite(value));

    const auto resolution = oracleLogitResolution(*context, mel, reference, prompt);
    const auto worst = maximumAbsoluteDifference(mine, reference);
    const auto rms = rootMeanSquareDifference(mine, reference);

    check(resolution > 0.0f);
    check(worst <= 25.0f * resolution);
    check(argmaxOf(mine) == argmaxOf(reference));

    std::cout << "  prompt logits: max |a - e| " << worst << ", rms " << rms
              << ", over a range of " << largestMagnitude(reference)
              << "; the reference's own one-ulp resolution is " << resolution << " ("
              << worst / resolution << "x)\n";
};

// The same, carried forward: each step feeds *the reference's* argmax to both
// sides, so the two never decode different histories and a disagreement at step
// k is about step k rather than about a token one of them chose at step k - 1.
//
// The reference re-decodes the whole prefix from position zero every time
// (whisper_decode with n_past 0), where ours appends one token to a KV cache
// that has been growing since beginSequence. So this also says the cache
// reproduces a full re-run, at the real width, against an implementation that
// shares none of its code.
auto tDecoderFollowsTheOracle = test("Oracle/decoderFollowsTheOracle") = []
{
    if (!canRun())
        return;

    const auto context = loadOracle();

    if (context == nullptr)
        return;

    const auto samples = paddedToWindow(readPcm16Wav(WHISPER_EACP_JFK_WAV));

    if (samples.empty())
        return;

    const auto mel = ourMelSpectrogram(Device::shared(), samples);

    auto prompt = std::vector<whisper_token> {startOfTranscript, noTimestamps};
    auto reference = oracleLogitsForMel(*context, mel, prompt);
    check(!reference.empty());

    auto& ours = ourDecoding();
    ours.beginFrom(mel);
    auto mine = ours.step({startOfTranscript, noTimestamps});

    const auto resolution = oracleLogitResolution(*context, mel, reference, prompt);

    // whisper_set_mel and whisper_encode ran inside oracleLogitResolution with a
    // perturbed mel, so the reference's encoder state has to be put back before
    // the follow-on steps read it again.
    reference = oracleLogitsForMel(*context, mel, prompt);

    auto agreements = 0;
    auto worstOverall = 0.0f;

    for (auto step = 0; step < followOnSteps; ++step)
    {
        const auto chosen = argmaxOf(reference);

        agreements += argmaxOf(mine) == chosen ? 1 : 0;
        worstOverall =
            std::max(worstOverall, maximumAbsoluteDifference(mine, reference));

        prompt.push_back((whisper_token) chosen);
        reference = oracleLogitsForPrompt(*context, prompt);
        mine = ours.step({chosen});

        check(!reference.empty());
        check((int) mine.size() == (int) reference.size());
    }

    agreements += argmaxOf(mine) == argmaxOf(reference) ? 1 : 0;
    worstOverall =
        std::max(worstOverall, maximumAbsoluteDifference(mine, reference));

    check(agreements == followOnSteps + 1);
    check(worstOverall <= 25.0f * resolution);

    std::cout << "  " << followOnSteps + 1 << " steps: argmax agrees " << agreements
              << " times, worst |a - e| " << worstOverall << " ("
              << worstOverall / resolution << "x the reference's resolution)\n";
};

// The same comparison at a reduced audio context — whisper.cpp's audio_ctx,
// and our Encoder::encode over a prefix — which is the only reference there is
// for a truncated encoder. Both sides read the same 3000-frame mel of ours and
// both truncate it the same way: whisper.cpp copies the first 2 * n_ctx frames
// into a tensor of that width (whisper_encode_internal, "set the input"), and
// ours dispatches the convolutions over the same prefix with zero past it.
//
// 576 positions is the interesting count rather than a safe one: it is where
// jfk.wav's transcript changes, so a run at it is one where the truncation is
// doing something the full window does not, and agreeing there is a claim
// about the arithmetic rather than about a margin big enough to hide it.
auto tDecoderAgreesWithTheOracleAtAReducedContext =
    test("Oracle/decoderAgreesWithTheOracleAtAReducedContext") = []
{
    if (!canRun())
        return;

    const auto context = loadOracle();

    if (context == nullptr)
        return;

    const auto samples = paddedToWindow(readPcm16Wav(WHISPER_EACP_JFK_WAV));

    if (samples.empty())
        return;

    const auto mel = ourMelSpectrogram(Device::shared(), samples);
    const auto prompt = std::vector<whisper_token> {startOfTranscript, noTimestamps};
    const auto atTheWindow = oracleLogitsForMel(*context, mel, prompt);

    auto& ours = ourDecoding();

    for (auto audioContext: {1024, 768, 576})
    {
        check(armOracleAudioContext(*context, samples, audioContext));

        const auto reference = oracleLogitsForMel(*context, mel, prompt);
        check(!reference.empty());

        // The parameter took: a reference encoding this many positions is not
        // one encoding 1500, and every number below would be meaningless if it
        // were.
        check(maximumAbsoluteDifference(atTheWindow, reference) > 0.5f);

        ours.beginFrom(mel, audioContext);
        const auto mine = ours.step({startOfTranscript, noTimestamps});

        check((int) mine.size() == (int) reference.size());

        const auto resolution =
            oracleLogitResolution(*context, mel, reference, prompt);
        const auto worst = maximumAbsoluteDifference(mine, reference);
        const auto rms = rootMeanSquareDifference(mine, reference);

        check(resolution > 0.0f);
        check(worst <= 100.0f * resolution);
        check(argmaxOf(mine) == argmaxOf(reference));

        std::cout << "  prompt logits at " << audioContext
                  << " positions: max |a - e| " << worst << ", rms " << rms
                  << ", over a range of " << largestMagnitude(reference)
                  << "; the reference's own one-ulp resolution is " << resolution
                  << " (" << worst / resolution << "x)\n";
    }
};
