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
TensorLoader decoderTensors(const SafeTensors& file)
{
    return TensorLoader {file, "decoder"};
}

// The logits projection's weight as well as the gather's table, and the gather
// is the reader with no packed form — so this one is float whatever the repo
// ships, widened here for an fp16 repo. The logits keep their own packed copy
// beside it, which is the same values to the bit.
TensorBuffer loadTokenEmbedding(const SafeTensors& file, const DecoderShape& shape)
{
    const auto tensors = decoderTensors(file);
    const auto name = decoderName("embed_tokens.weight");
    tensors.checkShape(tensors.require(name), {shape.vocabularySize, shape.width});

    return file.makeFloatBuffer(name);
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

    return file.makeFloatBuffer(name);
}

std::string decoderLayerPrefix(int index)
{
    return decoderName("layers.") + std::to_string(index) + ".";
}
} // namespace

DecoderLayerWeights::DecoderLayerWeights(const SafeTensors& file,
                                         const DecoderShape& shape,
                                         int index,
                                         WeightPacking packing)
    : selfAttentionNormWeight(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "self_attn_layer_norm.weight", {shape.width}))
    , selfAttentionNormBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "self_attn_layer_norm.bias", {shape.width}))
    , selfQueryWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "self_attn.q_proj.weight",
          {shape.width, shape.width},
          packing))
    , selfQueryBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "self_attn.q_proj.bias", {shape.width}))
    , selfKeyWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "self_attn.k_proj.weight",
          {shape.width, shape.width},
          packing))
    , selfValueWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "self_attn.v_proj.weight",
          {shape.width, shape.width},
          packing))
    , selfValueBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "self_attn.v_proj.bias", {shape.width}))
    , selfAttentionOutputWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "self_attn.out_proj.weight",
          {shape.width, shape.width},
          packing))
    , selfAttentionOutputBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "self_attn.out_proj.bias", {shape.width}))
    , crossAttentionNormWeight(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "encoder_attn_layer_norm.weight",
          {shape.width}))
    , crossAttentionNormBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "encoder_attn_layer_norm.bias", {shape.width}))
    , crossQueryWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "encoder_attn.q_proj.weight",
          {shape.width, shape.width},
          packing))
    , crossQueryBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "encoder_attn.q_proj.bias", {shape.width}))
    , crossKeyWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "encoder_attn.k_proj.weight",
          {shape.width, shape.width},
          packing))
    , crossValueWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "encoder_attn.v_proj.weight",
          {shape.width, shape.width},
          packing))
    , crossValueBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "encoder_attn.v_proj.bias", {shape.width}))
    , crossAttentionOutputWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "encoder_attn.out_proj.weight",
          {shape.width, shape.width},
          packing))
    , crossAttentionOutputBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "encoder_attn.out_proj.bias", {shape.width}))
    , finalNormWeight(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "final_layer_norm.weight", {shape.width}))
    , finalNormBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "final_layer_norm.bias", {shape.width}))
    , feedForwardWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "fc1.weight",
          {shape.feedForwardWidth, shape.width},
          packing))
    , feedForwardBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "fc1.bias", {shape.feedForwardWidth}))
    , feedForwardOutputWeight(decoderTensors(file).loadProjectionWeight(
          decoderLayerPrefix(index) + "fc2.weight",
          {shape.width, shape.feedForwardWidth},
          packing))
    , feedForwardOutputBias(decoderTensors(file).loadFloatTensor(
          decoderLayerPrefix(index) + "fc2.bias", {shape.width}))
{
}

DecoderWeights::DecoderWeights(const SafeTensors& file,
                               const DecoderShape& shapeToUse,
                               WeightPacking packingToUse)
    : shape(shapeToUse)
    , tokenEmbedding(loadTokenEmbedding(file, shapeToUse))
    , positionalEmbedding(loadPositionalEmbedding(file, shapeToUse))
    , finalNormWeight(decoderTensors(file).loadFloatTensor(
          decoderName("layer_norm.weight"), {shapeToUse.width}))
    , finalNormBias(decoderTensors(file).loadFloatTensor(
          decoderName("layer_norm.bias"), {shapeToUse.width}))
{
    // The shape of the table was checked above, so this narrows a tensor
    // already known to be the matrix it claims to be — and keeps the result
    // only if narrowing changed none of it. For an fp16 file the packed upload
    // is the blob's own halves, so the copy direction simply reverses: the
    // gather got the widened table, and the logits get the original.
    if (packingToUse == WeightPacking::ExactHalf)
        packedTokenEmbedding =
            file.makeExactHalfBuffer(decoderName("embed_tokens.weight"));

    layers.reserve(shapeToUse.layers);

    for (auto index = 0; index < shapeToUse.layers; ++index)
        layers.emplace_back(file, shapeToUse, index, packingToUse);
}
} // namespace WSP
