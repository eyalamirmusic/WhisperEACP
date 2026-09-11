#include "Common.h"

// The decoder suite's harness rather than a second copy of it: DecoderRun is
// what drives one step of the real weights and reads its logits back, and
// syntheticEncoderOutput is the rows to attend to. Both come out of the
// module whose test this one generalises — Decoder/TinyEn's bit-identical
// logits check, over the whole packing policy instead of over the one tensor.
#include "../Decoder/Common.h"

#include <iostream>
#include <vector>

// What Whisper::setPacksWeights is worth and what it costs, on the real
// weights: the bytes a run reads are halved wherever narrowing is bit-exact,
// and nothing else about the run moves. tiny.en's tensors are fp16 values in
// an F32 container — Model/TinyEn/weightsAreExactlyHalves says so over all
// 37,760,256 of them — so every projection packs and every number the runtime
// computes is the number it computed before.
//
// Skips without a device, without the model, or without the sample, the same
// shape as every other test here that needs a download.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;
using namespace eacp::GPU;

namespace
{
// 32 encoder rows rather than 1500, for the same reason Tests/Decoder uses
// that number: every stride a step is walked at is the same at 32 as at 1500,
// and the packed and float runs have to differ in nothing but the load width.
constexpr auto shortCrossPositions = 32;

bool canRun()
{
    return Device::shared().isValid() && hasWhisperModel()
           && hasSampleFile(jfkSample);
}

Whisper& unpackedModel()
{
    static const auto model = []
    {
        auto whisper = std::make_unique<Whisper>();
        whisper->load(modelDirectory());
        whisper->setPacksWeights(false);
        whisper->prepare();
        return whisper;
    }();

    return *model;
}
} // namespace

// The whole runtime twice over the same recording, once with the weights
// packed and once with them as the file holds them: **the same tokens**, in
// the same order, and the sentence this suite pins either way.
//
// A transcript is the coarsest thing this comparison could assert and the one
// that matters most — greedy sampling turns a single flipped argmax into a
// different sentence from there on, so token-for-token equality over 24 tokens
// is 24 arg maxima agreeing after 25 steps of a four-layer decoder.
auto tPackedWeightsTranscribeIdentically =
    test("Whisper/TinyEn/packedWeightsTranscribeIdentically") = []
{
    if (!canRun())
        return;

    auto& packed = preparedModel();
    auto& asShipped = unpackedModel();

    // The default, which is what every other test in this suite and every app
    // in the tree runs at.
    check(packed.packsWeights());
    check(!asShipped.packsWeights());

    const auto samples = readWavFile(sampleFile(jfkSample));

    const auto packedTokens = packed.transcribe(samples);
    const auto floatTokens = asShipped.transcribe(samples);

    check(packedTokens.size() == jfkTokenCount);
    check(packedTokens.size() == floatTokens.size());

    auto differing = 0;

    for (auto index = 0; index < packedTokens.size(); ++index)
        if (packedTokens[index] != floatTokens[index])
            ++differing;

    check(differing == 0);
    check(trimmed(packed.textForTokens(packedTokens)) == jfkTranscript);
    check(trimmed(asShipped.textForTokens(floatTokens)) == jfkTranscript);

    std::cout << "  jfk.wav packed against as-shipped: " << packedTokens.size()
              << " tokens, " << differing << " differing\n";
};

// And underneath the transcript, the numbers it was sampled from: one decode
// step of the runtime's own prompt, every logit compared **bit for bit**.
//
// An equality rather than a tolerance is the honest assertion here. A packed
// weight is the same number as the float one it was narrowed from — that is
// the whole condition makeExactHalfBuffer packs under — and the same products
// are summed in the same order by the same kernel at the same split, so only
// the load width differs. Anything but equality would mean one of those
// sentences is false.
//
// The check that the packing fired at all is what keeps this from holding
// vacuously: a file whose weights were refused would be compared against
// itself, and pass.
auto tPackedWeightsLogitsAreIdentical =
    test("Whisper/TinyEn/packedWeightsLogitsAreIdentical") = []
{
    if (!canRun())
        return;

    const auto config = ModelConfig::fromFile(modelFile("config.json"));
    const auto shape = DecoderShape::fromConfig(config, shortCrossPositions);

    const auto file = SafeTensors::fromFile(modelFile("model.safetensors"));
    const auto encoderOutput = syntheticEncoderOutput(shape);

    // The tokens a run actually opens with — `<|startoftranscript|>` and
    // `<|notimestamps|>` — taken from the runtime rather than written out
    // here, so this step is the step the transcript above began with.
    auto tokens = std::vector<int> {};

    for (auto token: preparedModel().prompt())
        tokens.push_back(token);

    check(tokens.size() == 2);

    // Without these the equality below would hold vacuously for a file whose
    // packing was refused rather than one whose packing was exact.
    check(
        file.makeExactHalfBuffer(decoderTensor("embed_tokens.weight")).has_value());
    check(file.makeExactHalfBuffer(decoderLayerTensor(0, "fc1.weight")).has_value());

    const auto logitsFrom = [&](WeightPacking packing)
    {
        auto run = DecoderRun {shape, file, packing};
        run.begin(encoderOutput);
        return run.step(tokens).logits;
    };

    const auto asShipped = logitsFrom(WeightPacking::Float);
    const auto packed = logitsFrom(WeightPacking::ExactHalf);

    check(asShipped.size() == packed.size());
    check(asShipped.size() == (int) tokens.size() * shape.vocabularySize);

    auto differing = 0;

    for (auto index = 0; index < asShipped.size(); ++index)
        if (asShipped[index] != packed[index])
            ++differing;

    check(differing == 0);

    std::cout << "  one step of " << tokens.size() << " tokens: " << asShipped.size()
              << " logits, " << differing << " differing\n";
};
