#include "Common.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

// The real openai/whisper-tiny.en weights, which are a download and never a
// commit: configure with -DWHISPER_EACP_FETCH_MODEL=ON, or point
// WHISPER_MODEL_DIR at a checkout that already has them. Both tests here return
// early when the files are absent, the same shape as a GPU test returning early
// on an invalid device.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;
using namespace eacp::GPU;

namespace
{
constexpr auto configFile = "config.json";
constexpr auto weightsFile = "model.safetensors";

// 32 encoder rows rather than 1500: the cross-attention projections are the
// expensive half of a scalar decoder, and every stride they are walked at is
// the same at 32 as at 1500.
constexpr auto shortCrossPositions = 32;

// <|notimestamps|>, which is not in config.json under a name — it is the second
// half of the one forced_decoder_ids entry, [[1, 50362]], and the id
// tokenizer.json gives <|notimestamps|> in the English-only vocabulary. The
// tokenizer file is not among the three this suite downloads, so the id is
// written here rather than read.
constexpr auto noTimestampsToken = 50362;

bool hasModel()
{
    return hasModelFile(configFile) && hasModelFile(weightsFile);
}

double secondsSince(std::chrono::steady_clock::time_point start)
{
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double>(elapsed).count();
}

// Both halves of a step against the reference rows it should have reproduced,
// and the greedy token each of them chose, reported as a number rather than
// only as a pass.
double checkStepAgainstReference(const StepResult& result,
                                 const ReferenceDecoding& expected,
                                 const DecoderShape& shape,
                                 int firstRow,
                                 int rowCount,
                                 double tolerance)
{
    const auto hidden = worstError(
        result.hidden, rowsOf(expected.hidden, firstRow, rowCount, shape.width));

    const auto logits = worstError(
        result.logits,
        rowsOf(expected.logits, firstRow, rowCount, shape.logitElementCount()));

    check(hidden <= tolerance);
    check(logits <= tolerance);

    for (auto row = 0; row < rowCount; ++row)
        check((int) result.argmax[row]
              == argmaxRow(
                  expected.logits, firstRow + row, shape.logitElementCount()));

    return std::max(hidden, logits);
}
} // namespace

// Every decoder tensor of the real file under the name this suite writes out,
// at the shape config.json implies. There is no proj_out.weight to look for:
// tiny.en's 167 tensors are all under model.encoder. or model.decoder., and the
// logits projection is embed_tokens itself.
auto tTinyEnDecoderWeightsLoad = test("Decoder/TinyEn/weightsLoad") = []
{
    if (!hasModel() || !Device::shared().isValid())
        return;

    const auto config = ModelConfig::fromFile(modelFile(configFile));
    const auto shape = DecoderShape::fromConfig(config, shortCrossPositions);

    check(shape.width == 384);
    check(shape.heads == 6);
    check(shape.headWidth() == 64);
    check(shape.layers == 4);
    check(shape.vocabularySize == 51864);
    check(shape.maxPositions == 448);
    check(shape.crossPositions == shortCrossPositions);

    const auto file = SafeTensors::fromFile(modelFile(weightsFile));
    check(!file.contains("proj_out.weight"));

    const auto weights = DecoderWeights {file, shape};

    check(weights.shape == shape);
    check(weights.layers.size() == 4);
    check(weights.tokenEmbedding.buffer.size()
          == (int) sizeof(float) * shape.vocabularySize * shape.width);
    check(weights.positionalEmbedding.buffer.size()
          == (int) sizeof(float) * shape.maxPositions * shape.width);

    for (const auto& layer: weights.layers)
    {
        check(layer.selfKeyWeight.buffer.isValid());
        check(layer.crossKeyWeight.buffer.isValid());
        check(layer.feedForwardOutputWeight.buffer.size()
              == (int) sizeof(float) * shape.width * shape.feedForwardWidth);
    }

    // F32 throughout, which is what tiny.en ships; a repo that shipped fp16
    // would leave the projections packed and these two float anyway.
    check(!weights.tokenEmbedding.isPackedHalf());
    check(!weights.positionalEmbedding.isPackedHalf());
};

// The scalar reference driven by the real weights, over the prompt a run
// actually starts with — the reference on its own, before the GPU decoder below
// is compared against it. What it pins is that the reference survives a real
// model: finite everywhere, the two prompt rows different from each other, and
// an argmax that is one of the special tokens.
//
// That last one is the property this input can carry, and it is worth more than
// it looks: the encoder output here is a synthetic sweep rather than anything a
// real encoder produced, and a model asked what follows <|startoftranscript|>
// for something that is not speech should reach for a marker rather than a word
// piece. 1608 of the 51864 ids are special, so a decoder wired wrongly enough
// to be predicting noise fails this about 97 times in 100. Measured, it comes
// out <|nospeech|> (50361) and then <|endoftext|>, which is exactly the pair.
//
// The obvious stronger assertion — that the first row is not <|endoftext|> —
// was tried and rejected: over five other sweeps of the same shape, four of
// them put <|endoftext|> first. An empty transcript is a perfectly reasonable
// answer to noise, so that version passes on the input it was written against
// and says nothing.
auto tTinyEnDecoderReferencePrompt = test("Decoder/TinyEn/referencePrompt") = []
{
    if (!hasModel())
        return;

    const auto config = ModelConfig::fromFile(modelFile(configFile));
    const auto shape = DecoderShape::fromConfig(config, shortCrossPositions);

    check(config.decoderStartToken != ModelConfig::noToken);
    check(config.endOfSequenceToken != ModelConfig::noToken);

    const auto file = SafeTensors::fromFile(modelFile(weightsFile));
    const auto model = readReferenceDecoderModel(file, shape);
    const auto tokens =
        std::vector<int> {config.decoderStartToken, noTimestampsToken};

    const auto start = std::chrono::steady_clock::now();
    const auto decoded =
        referenceDecode(model, shape, syntheticEncoderOutput(shape), tokens);
    const auto elapsed = secondsSince(start);

    check((int) decoded.hidden.size() == (int) tokens.size() * shape.width);
    check((int) decoded.logits.size() == (int) tokens.size() * shape.vocabularySize);

    for (auto value: decoded.logits)
        check(std::isfinite(value));

    const auto first = argmaxRow(decoded.logits, 0, shape.vocabularySize);
    const auto second = argmaxRow(decoded.logits, 1, shape.vocabularySize);

    // <|endoftext|> is the first id past the byte-level pieces, so this is
    // "a special token" written as the vocabulary lays them out.
    check(first >= config.endOfSequenceToken);
    check(first < shape.vocabularySize);

    auto rowsDiffer = false;

    for (auto token = 0; token < shape.vocabularySize; ++token)
        rowsDiffer =
            rowsDiffer
            || decoded.logits[(std::size_t) token]
                   != decoded.logits[(std::size_t) (shape.vocabularySize + token)];

    check(rowsDiffer);

    std::cout << "  tiny.en over " << tokens.size() << " tokens and "
              << shortCrossPositions << " encoder rows: " << elapsed
              << " s on the CPU, argmax " << first << " then " << second << "\n";
};

// The same scalar reference the synthetic model is checked against, driven by
// the real weights: the prompt a run actually starts with, then two steps of
// whatever the reference itself would have sampled, so the tokens the cache is
// exercised with are the model's own rather than made up.
//
// A shape mistake that a small random model happens to survive — a head width
// that only divides evenly at 8, a positional row read at the wrong offset, a
// cache bound one row out — has nowhere to hide over four layers of 384-wide
// blocks and a 51864-token vocabulary.
//
// 5.4e-5 measured against 1e-4 asserted, the encoder's number for the encoder's
// reason: four blocks of 384- and 1536-wide float32 sums, and then a 384-long
// one per vocabulary entry, accumulate further from a double reference than the
// two-layer synthetic model does, and the same assertion has to hold against
// HLSL's intrinsics as against MSL's.
auto tTinyEnDecoderMatchesReference =
    test("Decoder/TinyEn/matchesScalarReference") = []
{
    if (!hasModel() || !Device::shared().isValid())
        return;

    const auto config = ModelConfig::fromFile(modelFile(configFile));
    const auto shape = DecoderShape::fromConfig(config, shortCrossPositions);

    const auto file = SafeTensors::fromFile(modelFile(weightsFile));
    const auto model = readReferenceDecoderModel(file, shape);
    const auto encoderOutput = syntheticEncoderOutput(shape);

    auto tokens = std::vector<int> {config.decoderStartToken, noTimestampsToken};

    auto run = DecoderRun {shape, file};
    run.begin(encoderOutput);

    auto expected = referenceDecode(model, shape, encoderOutput, tokens);
    auto worst = checkStepAgainstReference(
        run.step(tokens), expected, shape, 0, (int) tokens.size(), 1e-4);

    // Two more tokens, each the argmax of the row the reference just produced,
    // fed to the GPU one at a time through the cache.
    for (auto index = 0; index < 2; ++index)
    {
        const auto next = argmaxRow(
            expected.logits, (int) tokens.size() - 1, shape.vocabularySize);

        tokens.push_back(next);
        expected = referenceDecode(model, shape, encoderOutput, tokens);

        worst = std::max(worst,
                         checkStepAgainstReference(run.step({next}),
                                                   expected,
                                                   shape,
                                                   (int) tokens.size() - 1,
                                                   1,
                                                   1e-4));
    }

    check(run.position() == (int) tokens.size());

    std::cout << "  tiny.en over " << tokens.size() << " tokens and "
              << shortCrossPositions << " encoder rows: worst error " << worst
              << "\n";
};

// The shape a step is actually run at: a full 30 second window behind it, so
// cross-attention reads 1500 encoder rows in every layer. No reference — a
// scalar decoder over 1500 cross positions is minutes — so what this asserts is
// that every logit came back a number, which is what a dispatch that ran off
// the end of a buffer or an unnormalised softmax row would not produce.
//
// The two numbers are what plan.md wants on the record: what opening a sequence
// costs, which is the cross-attention keys and values projected once out of
// 1500 rows in every layer, and what a single-token step costs after it. Wall
// clock around commit() from a Debug host build, which is the only clock there
// is — eacp's FrameTimer is driven by Frame and a CommandBuffer has no
// timestamp hook.
auto tTinyEnDecoderStepTiming = test("Decoder/TinyEn/stepTiming") = []
{
    auto& device = Device::shared();

    if (!hasModel() || !device.isValid())
        return;

    const auto config = ModelConfig::fromFile(modelFile(configFile));
    const auto shape = DecoderShape::fromConfig(config, config.maxSourcePositions);

    check(shape.crossPositions == 1500);

    const auto file = SafeTensors::fromFile(modelFile(weightsFile));
    const auto encoderOutput = spreadValues(shape.crossElementCount(), 31u, 1.f);

    auto run = DecoderRun {shape, file};
    const auto opening = run.begin(encoderOutput);

    const auto result = run.step({config.decoderStartToken});

    check(result.logits.size() == shape.vocabularySize);
    check(run.position() == 1);

    for (auto index = 0; index < result.logits.size(); ++index)
        check(std::isfinite(result.logits[index]));

    check((int) result.argmax[0] < shape.vocabularySize);

    std::cout << "  tiny.en at 1500 cross positions: " << opening
              << " s to project the cross keys and values, " << run.stepSeconds()
              << " s for a single-token step\n";
};
