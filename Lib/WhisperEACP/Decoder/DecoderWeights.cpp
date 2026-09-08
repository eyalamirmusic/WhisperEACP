#include "DecoderWeights.h"

#include <WhisperEACP/Model/TensorLoader.h>

#include <string>

namespace WSP
{
namespace
{
constexpr auto decoderPrefix = "model.decoder.";

// The gather has no class in Kernels/ yet — it arrives with the Decoder — so
// the tensors it reads name it by what it does rather than by a type this file
// would have to keep in step.
constexpr auto embeddingReader = "the embedding gather";

std::string decoderName(std::string_view suffix)
{
    return std::string {decoderPrefix} + std::string {suffix};
}

// The word every refusal below uses for this loader, which is all the checks
// in Model/TensorLoader.h have to be told to serve it as well as the encoder.
TensorLoader decoderTensors(const SafeTensors& file)
{
    return TensorLoader {file, "decoder"};
}

// The logits projection's weight as well as the gather's table, and the gather
// is the reader with no packed form — so this one is float whatever the repo
// ships, and the tie between the two stays exact rather than exact up to a
// conversion.
TensorBuffer loadTokenEmbedding(const SafeTensors& file, const DecoderShape& shape)
{
    const auto tensors = decoderTensors(file);
    const auto name = decoderName("embed_tokens.weight");
    tensors.checkShape(tensors.require(name), {shape.vocabularySize, shape.width});

    auto loaded = file.makeBuffer(name);
    tensors.rejectPackedHalf(loaded, name, embeddingReader);

    return loaded;
}

TensorBuffer loadPositionalEmbedding(const SafeTensors& file,
                                     const DecoderShape& shape)
{
    const auto tensors = decoderTensors(file);
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

    auto loaded = file.makeBuffer(name);
    tensors.rejectPackedHalf(loaded, name, embeddingReader);

    return loaded;
}

std::string decoderLayerPrefix(int index)
{
    return decoderName("layers.") + std::to_string(index) + ".";
}
} // namespace

DecoderLayerWeights::DecoderLayerWeights(const SafeTensors& file,
                                         const DecoderShape& shape,
                                         int index)
    : selfAttentionNormWeight(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "self_attn_layer_norm.weight",
          {shape.width},
          "LayerNorm"))
    , selfAttentionNormBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "self_attn_layer_norm.bias",
          {shape.width},
          "LayerNorm"))
    , selfQueryWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "self_attn.q_proj.weight",
          {shape.width, shape.width}))
    , selfQueryBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "self_attn.q_proj.bias",
          {shape.width},
          "Linear"))
    , selfKeyWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "self_attn.k_proj.weight",
          {shape.width, shape.width}))
    , selfValueWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "self_attn.v_proj.weight",
          {shape.width, shape.width}))
    , selfValueBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "self_attn.v_proj.bias",
          {shape.width},
          "Linear"))
    , selfAttentionOutputWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "self_attn.out_proj.weight",
          {shape.width, shape.width}))
    , selfAttentionOutputBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "self_attn.out_proj.bias",
          {shape.width},
          "Linear"))
    , crossAttentionNormWeight(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "encoder_attn_layer_norm.weight",
          {shape.width},
          "LayerNorm"))
    , crossAttentionNormBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "encoder_attn_layer_norm.bias",
          {shape.width},
          "LayerNorm"))
    , crossQueryWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "encoder_attn.q_proj.weight",
          {shape.width, shape.width}))
    , crossQueryBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "encoder_attn.q_proj.bias",
          {shape.width},
          "Linear"))
    , crossKeyWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "encoder_attn.k_proj.weight",
          {shape.width, shape.width}))
    , crossValueWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "encoder_attn.v_proj.weight",
          {shape.width, shape.width}))
    , crossValueBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "encoder_attn.v_proj.bias",
          {shape.width},
          "Linear"))
    , crossAttentionOutputWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "encoder_attn.out_proj.weight",
          {shape.width, shape.width}))
    , crossAttentionOutputBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "encoder_attn.out_proj.bias",
          {shape.width},
          "Linear"))
    , finalNormWeight(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "final_layer_norm.weight",
          {shape.width},
          "LayerNorm"))
    , finalNormBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "final_layer_norm.bias",
          {shape.width},
          "LayerNorm"))
    , feedForwardWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "fc1.weight",
          {shape.feedForwardWidth, shape.width}))
    , feedForwardBias(decoderTensors(file).loadFloatTensor(decoderLayerPrefix(index)
                                                               + "fc1.bias",
                                                           {shape.feedForwardWidth},
                                                           "Linear"))
    , feedForwardOutputWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "fc2.weight",
          {shape.width, shape.feedForwardWidth}))
    , feedForwardOutputBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "fc2.bias", {shape.width}, "Linear"))
{
}

DecoderWeights::DecoderWeights(const SafeTensors& file,
                               const DecoderShape& shapeToUse,
                               LogitsWeight logitsWeightToUse)
    : shape(shapeToUse)
    , tokenEmbedding(loadTokenEmbedding(file, shapeToUse))
    , positionalEmbedding(loadPositionalEmbedding(file, shapeToUse))
    , finalNormWeight(decoderTensors(file).loadFloatTensor(
          decoderName("layer_norm.weight"), {shapeToUse.width}, "LayerNorm"))
    , finalNormBias(decoderTensors(file).loadFloatTensor(
          decoderName("layer_norm.bias"), {shapeToUse.width}, "LayerNorm"))
{
    // The shape and the storage of the float table were checked above, so this
    // narrows a tensor already known to be the matrix it claims to be — and
    // keeps the result only if narrowing changed none of it.
    if (logitsWeightToUse == LogitsWeight::PackedHalfCopy)
        packedTokenEmbedding =
            file.makeExactHalfBuffer(decoderName("embed_tokens.weight"));

    layers.reserve(shapeToUse.layers);

    for (auto index = 0; index < shapeToUse.layers; ++index)
        layers.emplace_back(file, shapeToUse, index);
}
} // namespace WSP
