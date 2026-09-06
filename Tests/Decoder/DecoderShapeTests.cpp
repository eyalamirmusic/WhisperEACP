#include "Common.h"

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;

namespace
{
// Every decoder field different from its encoder counterpart, so a fromConfig
// that read the wrong side of the config could not come out right by accident.
ModelConfig lopsidedConfig()
{
    auto config = ModelConfig {};
    config.melBins = 80;
    config.modelWidth = 12;
    config.encoderLayers = 3;
    config.decoderLayers = 5;
    config.encoderHeads = 4;
    config.decoderHeads = 6;
    config.encoderFeedForwardWidth = 20;
    config.decoderFeedForwardWidth = 28;
    config.maxSourcePositions = 90;
    config.maxTargetPositions = 34;
    config.vocabularySize = 77;

    return config;
}
} // namespace

// No device needed: these are the numbers the whole module is indexed by, and
// getting one of them wrong misplaces every dispatch after it.
auto tDecoderShapeCounts = test("Decoder/shapeCounts") = []
{
    const auto shape = smallDecoderShape();

    check(shape.headWidth() == 4);
    check(shape.attentionScale() == 0.5f);

    check(shape.rowElementCount() == 8);
    check(shape.feedForwardElementCount() == 16);
    check(shape.logitElementCount() == 24);

    check(shape.stepElementCount() == 128);
    check(shape.stepFeedForwardElementCount() == 256);
    check(shape.cacheElementCount() == 128);
    check(shape.crossElementCount() == 48);

    // A step may be a whole prompt, so the score buffers are sized for
    // maxPositions queries rather than one: [heads, queries, keys] over the
    // cache's capacity, and over the encoder's rows.
    check(shape.selfScoreElementCount() == 512);
    check(shape.crossScoreElementCount() == 192);
    check(shape.scoreRowCount(1) == 2);
    check(shape.scoreRowCount(4) == 8);
};

auto tDecoderShapeFromConfig = test("Decoder/shapeFromConfig") = []
{
    const auto config = lopsidedConfig();
    const auto shape = DecoderShape::fromConfig(config, 45);

    check(shape.width == 12);
    check(shape.heads == 6);
    check(shape.layers == 5);
    check(shape.feedForwardWidth == 28);
    check(shape.vocabularySize == 77);
    check(shape.maxPositions == 34);
    check(shape.crossPositions == 45);
    check(shape.headWidth() == config.decoderHeadWidth());

    // The cross-attention side is the encoder's row count, which is a
    // parameter here for the reason inputFrames is one next door: the real
    // weights get checked over an encoder output short enough for a CPU.
    const auto whole = DecoderShape::fromConfig(config, config.maxSourcePositions);
    check(whole.crossPositions == 90);
    check(whole.crossElementCount() == 90 * 12);
};

// Weights are loaded against one of these and dispatched against another, and
// nothing about a GPU buffer says which — so the comparison has to be the
// value's, not a field-by-field one at the call site.
auto tDecoderShapeEquality = test("Decoder/shapeEquality") = []
{
    const auto shape = smallDecoderShape();
    check(shape == smallDecoderShape());

    auto other = shape;
    other.crossPositions += 1;
    check(!(shape == other));
};
