#include "Common.h"

#include <functional>
#include <string>
#include <string_view>

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;
using namespace eacp::GPU;

namespace
{
std::string modelErrorText(const std::function<void()>& body)
{
    try
    {
        body();
    }
    catch (const ModelError& error)
    {
        return error.what();
    }
    catch (...)
    {
        return "some other exception";
    }

    return "no exception";
}

bool mentions(const std::string& text, std::string_view fragment)
{
    return text.find(fragment) != std::string::npos;
}

bool holdsFloats(const TensorBuffer& tensor, int elementCount)
{
    return tensor.buffer.isValid() && !tensor.isPackedHalf()
           && tensor.buffer.size() == (int) sizeof(float) * elementCount;
}

// An odd count of halves is padded to a whole 32-bit word, because readHalf
// fetches the word and would otherwise read past the allocation.
bool holdsHalves(const TensorBuffer& tensor, int elementCount)
{
    const auto bytes = 2 * elementCount + 2 * (elementCount % 2);

    return tensor.buffer.isValid() && tensor.isPackedHalf()
           && tensor.buffer.size() == bytes;
}

void checkLayerLoaded(const DecoderLayerWeights& layer,
                      const DecoderShape& shape,
                      ProjectionStorage projectionStorage)
{
    const auto width = shape.width;

    auto projection = [&](const TensorBuffer& tensor, int outputWidth, int inner)
    {
        return projectionStorage == ProjectionStorage::PackedHalf
                   ? holdsHalves(tensor, outputWidth * inner)
                   : holdsFloats(tensor, outputWidth * inner);
    };

    check(holdsFloats(layer.selfAttentionNormWeight, width));
    check(holdsFloats(layer.selfAttentionNormBias, width));
    check(projection(layer.selfQueryWeight, width, width));
    check(holdsFloats(layer.selfQueryBias, width));
    check(projection(layer.selfKeyWeight, width, width));
    check(projection(layer.selfValueWeight, width, width));
    check(holdsFloats(layer.selfValueBias, width));
    check(projection(layer.selfAttentionOutputWeight, width, width));
    check(holdsFloats(layer.selfAttentionOutputBias, width));

    check(holdsFloats(layer.crossAttentionNormWeight, width));
    check(holdsFloats(layer.crossAttentionNormBias, width));
    check(projection(layer.crossQueryWeight, width, width));
    check(holdsFloats(layer.crossQueryBias, width));
    check(projection(layer.crossKeyWeight, width, width));
    check(projection(layer.crossValueWeight, width, width));
    check(holdsFloats(layer.crossValueBias, width));
    check(projection(layer.crossAttentionOutputWeight, width, width));
    check(holdsFloats(layer.crossAttentionOutputBias, width));

    check(holdsFloats(layer.finalNormWeight, width));
    check(holdsFloats(layer.finalNormBias, width));
    check(projection(layer.feedForwardWeight, shape.feedForwardWidth, width));
    check(holdsFloats(layer.feedForwardBias, shape.feedForwardWidth));
    check(projection(layer.feedForwardOutputWeight, width, shape.feedForwardWidth));
    check(holdsFloats(layer.feedForwardOutputBias, width));
}

void checkLoaded(ProjectionStorage projectionStorage)
{
    const auto shape = smallDecoderShape();
    const auto file = syntheticDecoderFile(shape, projectionStorage);
    const auto weights = DecoderWeights {file, shape};

    check(weights.shape == shape);
    check(weights.layers.size() == shape.layers);

    // Float whatever the projections are: the gather subscripts this one, and
    // the logits projection reads the same buffer.
    check(holdsFloats(weights.tokenEmbedding, shape.vocabularySize * shape.width));
    check(holdsFloats(weights.positionalEmbedding, shape.cacheElementCount()));
    check(holdsFloats(weights.finalNormWeight, shape.width));
    check(holdsFloats(weights.finalNormBias, shape.width));

    for (const auto& layer: weights.layers)
        checkLayerLoaded(layer, shape, projectionStorage);
}
} // namespace

// Every tensor of a whole synthetic decoder, uploaded and named correctly. The
// names come from this suite rather than from the loader, so a rename on either
// side is a failure rather than a shared typo.
auto tDecoderWeightsLoad = test("Decoder/weightsLoad") = []
{
    if (!Device::shared().isValid())
        return;

    checkLoaded(ProjectionStorage::Float);
};

// The half of the fp16 rule that is a permission: a projection weight stays
// packed, because Linear has a variant that reads it that way.
auto tDecoderWeightsLoadPackedProjections =
    test("Decoder/weightsLoadPackedProjections") = []
{
    if (!Device::shared().isValid())
        return;

    checkLoaded(ProjectionStorage::PackedHalf);
};

auto tDecoderMissingTensorIsAnError = test("Decoder/missingTensorIsAnError") = []
{
    if (!Device::shared().isValid())
        return;

    const auto shape = smallDecoderShape();

    auto builder = TensorFileBuilder {};
    builder.addFloats(decoderTensor("embed_tokens.weight"),
                      {shape.vocabularySize, shape.width},
                      spreadValues(shape.vocabularySize * shape.width, 11u, 0.3f));

    const auto file = builder.parse();
    const auto text = modelErrorText([&] { DecoderWeights {file, shape}; });

    check(mentions(text, "model.decoder.embed_positions.weight"));
    check(mentions(text, "the file has none"));
};

// A layer tensor rather than one of the four at the top, reached by loading a
// file that is entirely valid against a shape it was not written for.
auto tDecoderMissingLayerIsAnError = test("Decoder/missingLayerIsAnError") = []
{
    if (!Device::shared().isValid())
        return;

    const auto shape = smallDecoderShape();
    const auto file = syntheticDecoderFile(shape, ProjectionStorage::Float);

    auto deeper = shape;
    deeper.layers += 1;

    const auto text = modelErrorText([&] { DecoderWeights {file, deeper}; });

    check(mentions(text, "model.decoder.layers.2.self_attn_layer_norm.weight"));
    check(mentions(text, "the file has none"));
};

auto tDecoderWrongRankIsAnError = test("Decoder/wrongRankIsAnError") = []
{
    if (!Device::shared().isValid())
        return;

    const auto shape = smallDecoderShape();
    const auto count = shape.vocabularySize * shape.width;

    auto builder = TensorFileBuilder {};
    builder.addFloats(decoderTensor("embed_tokens.weight"),
                      {shape.vocabularySize, shape.width, 1},
                      spreadValues(count, 12u, 0.3f));

    const auto file = builder.parse();
    const auto text = modelErrorText([&] { DecoderWeights {file, shape}; });

    check(mentions(text, "model.decoder.embed_tokens.weight"));
    check(mentions(text, "[24, 8, 1]"));
    check(mentions(text, "[24, 8]"));
};

// The dimension that is wrong is a layer's, so the message names the layer
// rather than the model — which is the difference between a shape mistake that
// bisects and one that only says the file is not this model.
auto tDecoderWrongDimensionIsAnError = test("Decoder/wrongDimensionIsAnError") = []
{
    if (!Device::shared().isValid())
        return;

    const auto shape = smallDecoderShape();
    const auto file = syntheticDecoderFile(shape, ProjectionStorage::Float);

    auto wider = shape;
    wider.feedForwardWidth += 1;

    const auto text = modelErrorText([&] { DecoderWeights {file, wider}; });

    check(mentions(text, "model.decoder.layers.0.fc1.weight"));
    check(mentions(text, "[16, 8]"));
    check(mentions(text, "[17, 8]"));
};

// A model that carries fewer target positions than the run needs, which is the
// one axis where "not equal" is not the test.
auto tDecoderShortEmbeddingIsAnError = test("Decoder/shortEmbeddingIsAnError") = []
{
    if (!Device::shared().isValid())
        return;

    const auto shape = smallDecoderShape();
    const auto file = syntheticDecoderFile(shape, ProjectionStorage::Float);

    auto longer = shape;
    longer.maxPositions = 32;

    const auto text = modelErrorText([&] { DecoderWeights {file, longer}; });

    check(mentions(text, "model.decoder.embed_positions.weight"));
    check(mentions(text, "carries 16 positions"));
};

// The tensor the fp16 rule turns on: embed_tokens is the gather's table as well
// as the logits projection's weight, and the gather has no packed read, so it
// is F32 or it is an error — never a buffer bound at half the stride it was
// written at, which is silent on both backends.
auto tDecoderPackedTokenEmbeddingIsAnError =
    test("Decoder/packedTokenEmbeddingIsAnError") = []
{
    if (!Device::shared().isValid())
        return;

    const auto shape = smallDecoderShape();

    auto builder = TensorFileBuilder {};
    builder.addHalves(decoderTensor("embed_tokens.weight"),
                      {shape.vocabularySize, shape.width},
                      spreadValues(shape.vocabularySize * shape.width, 13u, 0.3f));

    const auto file = builder.parse();
    const auto text = modelErrorText([&] { DecoderWeights {file, shape}; });

    check(mentions(text, "model.decoder.embed_tokens.weight"));
    check(mentions(text, "fp16"));
    check(mentions(text, "the embedding gather"));
};

auto tDecoderPackedNormIsAnError = test("Decoder/packedNormIsAnError") = []
{
    if (!Device::shared().isValid())
        return;

    const auto shape = smallDecoderShape();

    auto builder = TensorFileBuilder {};
    builder.addFloats(decoderTensor("embed_tokens.weight"),
                      {shape.vocabularySize, shape.width},
                      spreadValues(shape.vocabularySize * shape.width, 14u, 0.3f));
    builder.addFloats(decoderTensor("embed_positions.weight"),
                      {shape.maxPositions, shape.width},
                      spreadValues(shape.cacheElementCount(), 15u, 0.3f));
    builder.addHalves(decoderTensor("layer_norm.weight"),
                      {shape.width},
                      spreadValues(shape.width, 16u, 0.2f));

    const auto file = builder.parse();
    const auto text = modelErrorText([&] { DecoderWeights {file, shape}; });

    check(mentions(text, "model.decoder.layer_norm.weight"));
    check(mentions(text, "fp16"));
    check(mentions(text, "LayerNorm"));
};
