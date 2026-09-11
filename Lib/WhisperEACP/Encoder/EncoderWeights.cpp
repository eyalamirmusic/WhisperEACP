#include "EncoderWeights.h"

#include <WhisperEACP/Model/TensorLoader.h>

#include <string>

namespace WSP
{
namespace
{
constexpr auto encoderPrefix = "model.encoder.";

std::string encoderName(std::string_view suffix)
{
    return std::string {encoderPrefix} + std::string {suffix};
}

// The word every refusal below uses for this loader, which is all the checks
// in Model/TensorLoader.h have to be told to serve it as well as the decoder.
TensorLoader encoderTensors(const SafeTensors& file)
{
    return TensorLoader {file, "encoder"};
}

TensorBuffer loadPositionalEmbedding(const SafeTensors& file,
                                     const EncoderShape& shape)
{
    const auto tensors = encoderTensors(file);
    const auto name = encoderName("embed_positions.weight");
    const auto& tensor = tensors.require(name);

    // The one tensor whose first axis is a capacity rather than an extent: a
    // model carrying 1500 positions runs a 32-frame input perfectly well, and
    // only the shortfall is an error.
    tensors.checkPositionalWidth(tensor, shape.width);

    if (tensor.dimension(0) < shape.positions())
        throw ModelError {
            "tensor '" + name + "' carries " + std::to_string(tensor.dimension(0))
            + " positions, and this run needs " + std::to_string(shape.positions())};

    return file.makeFloatBuffer(name);
}

std::string encoderLayerPrefix(int index)
{
    return encoderName("layers.") + std::to_string(index) + ".";
}
} // namespace

EncoderLayerWeights::EncoderLayerWeights(const SafeTensors& file,
                                         const EncoderShape& shape,
                                         int index,
                                         WeightPacking packing)
    : attentionNormWeight(encoderTensors(file).loadFloatTensor(
          encoderLayerPrefix(index) + "self_attn_layer_norm.weight", {shape.width}))
    , attentionNormBias(encoderTensors(file).loadFloatTensor(
          encoderLayerPrefix(index) + "self_attn_layer_norm.bias", {shape.width}))
    , queryWeight(encoderTensors(file).loadProjectionWeight(
          encoderLayerPrefix(index) + "self_attn.q_proj.weight",
          {shape.width, shape.width},
          packing))
    , queryBias(encoderTensors(file).loadFloatTensor(
          encoderLayerPrefix(index) + "self_attn.q_proj.bias", {shape.width}))
    , keyWeight(encoderTensors(file).loadProjectionWeight(
          encoderLayerPrefix(index) + "self_attn.k_proj.weight",
          {shape.width, shape.width},
          packing))
    , valueWeight(encoderTensors(file).loadProjectionWeight(
          encoderLayerPrefix(index) + "self_attn.v_proj.weight",
          {shape.width, shape.width},
          packing))
    , valueBias(encoderTensors(file).loadFloatTensor(
          encoderLayerPrefix(index) + "self_attn.v_proj.bias", {shape.width}))
    , attentionOutputWeight(encoderTensors(file).loadProjectionWeight(
          encoderLayerPrefix(index) + "self_attn.out_proj.weight",
          {shape.width, shape.width},
          packing))
    , attentionOutputBias(encoderTensors(file).loadFloatTensor(
          encoderLayerPrefix(index) + "self_attn.out_proj.bias", {shape.width}))
    , finalNormWeight(encoderTensors(file).loadFloatTensor(
          encoderLayerPrefix(index) + "final_layer_norm.weight", {shape.width}))
    , finalNormBias(encoderTensors(file).loadFloatTensor(
          encoderLayerPrefix(index) + "final_layer_norm.bias", {shape.width}))
    , feedForwardWeight(encoderTensors(file).loadProjectionWeight(
          encoderLayerPrefix(index) + "fc1.weight",
          {shape.feedForwardWidth, shape.width},
          packing))
    , feedForwardBias(encoderTensors(file).loadFloatTensor(
          encoderLayerPrefix(index) + "fc1.bias", {shape.feedForwardWidth}))
    , feedForwardOutputWeight(encoderTensors(file).loadProjectionWeight(
          encoderLayerPrefix(index) + "fc2.weight",
          {shape.width, shape.feedForwardWidth},
          packing))
    , feedForwardOutputBias(encoderTensors(file).loadFloatTensor(
          encoderLayerPrefix(index) + "fc2.bias", {shape.width}))
{
}

EncoderWeights::EncoderWeights(const SafeTensors& file,
                               const EncoderShape& shapeToUse,
                               WeightPacking packingToUse)
    : shape(shapeToUse)
    , firstConvolutionWeight(encoderTensors(file).loadFloatTensor(
          encoderName("conv1.weight"),
          {shapeToUse.width,
           shapeToUse.melBins,
           EncoderShape::convolutionKernelSize}))
    , firstConvolutionBias(encoderTensors(file).loadFloatTensor(
          encoderName("conv1.bias"), {shapeToUse.width}))
    , secondConvolutionWeight(encoderTensors(file).loadFloatTensor(
          encoderName("conv2.weight"),
          {shapeToUse.width, shapeToUse.width, EncoderShape::convolutionKernelSize}))
    , secondConvolutionBias(encoderTensors(file).loadFloatTensor(
          encoderName("conv2.bias"), {shapeToUse.width}))
    , positionalEmbedding(loadPositionalEmbedding(file, shapeToUse))
    , finalNormWeight(encoderTensors(file).loadFloatTensor(
          encoderName("layer_norm.weight"), {shapeToUse.width}))
    , finalNormBias(encoderTensors(file).loadFloatTensor(
          encoderName("layer_norm.bias"), {shapeToUse.width}))
{
    layers.reserve(shapeToUse.layers);

    for (auto index = 0; index < shapeToUse.layers; ++index)
        layers.emplace_back(file, shapeToUse, index, packingToUse);
}
} // namespace WSP
