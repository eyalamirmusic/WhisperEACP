#include "DecoderWeights.h"

#include <WhisperEACP/Model/TensorLoader.h>

#include <string>

namespace WSP
{
namespace
{
constexpr auto decoderPrefix = "model.decoder.";

std::string decoderName(std::string_view suffix)
{
    return std::string {decoderPrefix} + std::string {suffix};
}

// The word every refusal below uses for this loader, which is all the checks
// in Model/TensorLoader.h have to be told to serve it as well as the encoder.
TensorLoader decoderTensors(const SafeTensors& file, WeightPlacement placement)
{
    return TensorLoader {file, "decoder", placement};
}

// The logits projection's weight as well as the gather's table, and the gather
// is the reader with no packed form — so this one is float whatever the repo
// ships, widened here for an fp16 repo. The logits keep their own packed copy
// beside it, which is the same values to the bit.
TensorBuffer loadTokenEmbedding(const SafeTensors& file,
                                const DecoderShape& shape,
                                WeightPlacement placement)
{
    const auto tensors = decoderTensors(file, placement);
    const auto name = decoderName("embed_tokens.weight");
    tensors.checkShape(tensors.require(name), {shape.vocabularySize, shape.width});

    if (placement == WeightPlacement::Host)
        return file.makeHostTensor(name);

    return file.makeFloatBuffer(name);
}

TensorBuffer loadPositionalEmbedding(const SafeTensors& file,
                                     const DecoderShape& shape,
                                     WeightPlacement placement)
{
    const auto tensors = decoderTensors(file, placement);
    const auto name = decoderName("embed_positions.weight");
    const auto& tensor = tensors.require(name);

    // The one tensor whose first axis is a capacity rather than an extent: a
    // model carrying more target positions than this run will ever reach is a
    // model that runs perfectly well, and only the shortfall is an error.
    tensors.checkPositionalWidth(tensor, shape.width);

    if (tensor.dimension(0) < shape.maxPositions)
        throw ModelError {"tensor '" + name + "' carries "
                          + std::to_string(tensor.dimension(0))
                          + " positions, and this decoder was built for "
                          + std::to_string(shape.maxPositions)};

    if (placement == WeightPlacement::Host)
        return file.makeHostTensor(name);

    return file.makeFloatBuffer(name);
}

std::string decoderLayerPrefix(int index)
{
    return decoderName("layers.") + std::to_string(index) + ".";
}
} // namespace

// One layer's loads, which differ only in the name, the shape and whether the
// tensor is a projection weight that may stay packed.
struct DecoderLayerWeights::Loader
{
    TensorBuffer vector(const std::string& suffix, int length) const
    {
        return decoderTensors(file, placement)
            .loadFloatTensor(decoderLayerPrefix(index) + suffix, {length});
    }

    TensorBuffer projection(const std::string& suffix, int rows, int columns) const
    {
        return decoderTensors(file, placement)
            .loadProjectionWeight(
                decoderLayerPrefix(index) + suffix, {rows, columns}, packing);
    }

    const SafeTensors& file;
    int index = 0;
    WeightPacking packing = WeightPacking::Float;
    WeightPlacement placement = WeightPlacement::Device;
};

DecoderLayerWeights::DecoderLayerWeights(const SafeTensors& file,
                                         const DecoderShape& shape,
                                         int index,
                                         WeightPacking packing,
                                         WeightPlacement placement)
    : DecoderLayerWeights(Loader {file, index, packing, placement}, shape)
{
}

DecoderLayerWeights::DecoderLayerWeights(const Loader& load,
                                         const DecoderShape& shape)
    : selfAttentionNormWeight(
          load.vector("self_attn_layer_norm.weight", shape.width))
    , selfAttentionNormBias(load.vector("self_attn_layer_norm.bias", shape.width))
    , selfQueryWeight(
          load.projection("self_attn.q_proj.weight", shape.width, shape.width))
    , selfQueryBias(load.vector("self_attn.q_proj.bias", shape.width))
    , selfKeyWeight(
          load.projection("self_attn.k_proj.weight", shape.width, shape.width))
    , selfValueWeight(
          load.projection("self_attn.v_proj.weight", shape.width, shape.width))
    , selfValueBias(load.vector("self_attn.v_proj.bias", shape.width))
    , selfAttentionOutputWeight(
          load.projection("self_attn.out_proj.weight", shape.width, shape.width))
    , selfAttentionOutputBias(load.vector("self_attn.out_proj.bias", shape.width))
    , crossAttentionNormWeight(
          load.vector("encoder_attn_layer_norm.weight", shape.width))
    , crossAttentionNormBias(
          load.vector("encoder_attn_layer_norm.bias", shape.width))
    , crossQueryWeight(
          load.projection("encoder_attn.q_proj.weight", shape.width, shape.width))
    , crossQueryBias(load.vector("encoder_attn.q_proj.bias", shape.width))
    , crossKeyWeight(
          load.projection("encoder_attn.k_proj.weight", shape.width, shape.width))
    , crossValueWeight(
          load.projection("encoder_attn.v_proj.weight", shape.width, shape.width))
    , crossValueBias(load.vector("encoder_attn.v_proj.bias", shape.width))
    , crossAttentionOutputWeight(
          load.projection("encoder_attn.out_proj.weight", shape.width, shape.width))
    , crossAttentionOutputBias(
          load.vector("encoder_attn.out_proj.bias", shape.width))
    , finalNormWeight(load.vector("final_layer_norm.weight", shape.width))
    , finalNormBias(load.vector("final_layer_norm.bias", shape.width))
    , feedForwardWeight(
          load.projection("fc1.weight", shape.feedForwardWidth, shape.width))
    , feedForwardBias(load.vector("fc1.bias", shape.feedForwardWidth))
    , feedForwardOutputWeight(
          load.projection("fc2.weight", shape.width, shape.feedForwardWidth))
    , feedForwardOutputBias(load.vector("fc2.bias", shape.width))
{
}

DecoderWeights::DecoderWeights(const SafeTensors& file,
                               const DecoderShape& shapeToUse,
                               WeightPacking packingToUse,
                               WeightPlacement placementToUse)
    : shape(shapeToUse)
    , placement(placementToUse)
    , tokenEmbedding(loadTokenEmbedding(file, shapeToUse, placementToUse))
    , positionalEmbedding(loadPositionalEmbedding(file, shapeToUse, placementToUse))
    , finalNormWeight(
          decoderTensors(file, placementToUse)
              .loadFloatTensor(decoderName("layer_norm.weight"), {shapeToUse.width}))
    , finalNormBias(
          decoderTensors(file, placementToUse)
              .loadFloatTensor(decoderName("layer_norm.bias"), {shapeToUse.width}))
{
    // The shape of the table was checked above, so this narrows a tensor
    // already known to be the matrix it claims to be — and keeps the result
    // only if narrowing changed none of it. For an fp16 file the packed upload
    // is the blob's own halves, so the copy direction simply reverses: the
    // gather got the widened table, and the logits get the original.
    if (packingToUse == WeightPacking::ExactHalf
        && placementToUse == WeightPlacement::Device)
        packedTokenEmbedding =
            file.makeExactHalfBuffer(decoderName("embed_tokens.weight"));

    layers.reserve(shapeToUse.layers);

    for (auto index = 0; index < shapeToUse.layers; ++index)
        layers.emplace_back(file, shapeToUse, index, packingToUse, placementToUse);
}
} // namespace WSP
