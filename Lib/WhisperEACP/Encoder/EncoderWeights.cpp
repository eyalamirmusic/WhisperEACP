#include "EncoderWeights.h"

#include <initializer_list>
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

std::string shapeText(const TensorInfo& tensor)
{
    auto text = std::string {"["};

    for (auto axis = 0; axis < tensor.rank(); ++axis)
        text += (axis == 0 ? "" : ", ") + std::to_string(tensor.dimension(axis));

    return text + "]";
}

const TensorInfo& require(const SafeTensors& file, const std::string& name)
{
    if (const auto* found = file.find(name))
        return *found;

    throw ModelError {"the encoder needs a tensor named '" + name
                      + "', and the file has none"};
}

void checkShape(const TensorInfo& tensor, std::initializer_list<int> expected)
{
    auto wrong = [&]
    {
        auto text = std::string {"["};
        auto axis = 0;

        for (auto extent: expected)
            text += (axis++ == 0 ? "" : ", ") + std::to_string(extent);

        throw ModelError {"tensor '" + tensor.name + "' is " + shapeText(tensor)
                          + ", and the encoder was built for " + text + "]"};
    };

    if (tensor.rank() != (int) expected.size())
        wrong();

    auto axis = 0;

    for (auto extent: expected)
        if (tensor.dimension(axis++) != extent)
            wrong();
}

// Bound to a program that subscripts a float buffer, so a packed one would be
// read at half the stride it was written at — silently, on both backends.
TensorBuffer loadFloatTensor(const SafeTensors& file,
                             const std::string& name,
                             std::initializer_list<int> expected,
                             std::string_view reader)
{
    checkShape(require(file, name), expected);
    auto loaded = file.makeBuffer(name);

    if (loaded.isPackedHalf())
        throw ModelError {"tensor '" + name
                          + "' is fp16, and this encoder binds it to "
                          + std::string {reader}
                          + ", which has no packed-half read: ship the tensor "
                            "as F32"};

    return loaded;
}

// A projection weight, which is the one operand that may stay packed: Linear
// has a half-reading variant and the Encoder picks it by storage.
TensorBuffer loadProjectionWeight(const SafeTensors& file,
                                  const std::string& name,
                                  std::initializer_list<int> expected)
{
    checkShape(require(file, name), expected);
    return file.makeBuffer(name);
}

TensorBuffer loadPositionalEmbedding(const SafeTensors& file,
                                     const EncoderShape& shape)
{
    const auto name = encoderName("embed_positions.weight");
    const auto& tensor = require(file, name);

    // The one tensor whose first axis is a capacity rather than an extent: a
    // model carrying 1500 positions runs a 32-frame input perfectly well, and
    // only the shortfall is an error.
    if (tensor.rank() != 2 || tensor.dimension(1) != shape.width)
        throw ModelError {"tensor '" + name + "' is " + shapeText(tensor)
                          + ", and the encoder was built for [positions, "
                          + std::to_string(shape.width) + "]"};

    if (tensor.dimension(0) < shape.positions())
        throw ModelError {
            "tensor '" + name + "' carries " + std::to_string(tensor.dimension(0))
            + " positions, and this run needs " + std::to_string(shape.positions())};

    auto loaded = file.makeBuffer(name);

    if (loaded.isPackedHalf())
        throw ModelError {"tensor '" + name
                          + "' is fp16, and this encoder binds it to Add, which "
                            "has no packed-half read: ship the tensor as F32"};

    return loaded;
}

std::string encoderLayerPrefix(int index)
{
    return encoderName("layers.") + std::to_string(index) + ".";
}
} // namespace

EncoderLayerWeights::EncoderLayerWeights(const SafeTensors& file,
                                         const EncoderShape& shape,
                                         int index)
    : attentionNormWeight(
          loadFloatTensor(file,
                          encoderLayerPrefix(index) + "self_attn_layer_norm.weight",
                          {shape.width},
                          "LayerNorm"))
    , attentionNormBias(
          loadFloatTensor(file,
                          encoderLayerPrefix(index) + "self_attn_layer_norm.bias",
                          {shape.width},
                          "LayerNorm"))
    , queryWeight(
          loadProjectionWeight(file,
                               encoderLayerPrefix(index) + "self_attn.q_proj.weight",
                               {shape.width, shape.width}))
    , queryBias(loadFloatTensor(file,
                                encoderLayerPrefix(index) + "self_attn.q_proj.bias",
                                {shape.width},
                                "Linear"))
    , keyWeight(
          loadProjectionWeight(file,
                               encoderLayerPrefix(index) + "self_attn.k_proj.weight",
                               {shape.width, shape.width}))
    , valueWeight(
          loadProjectionWeight(file,
                               encoderLayerPrefix(index) + "self_attn.v_proj.weight",
                               {shape.width, shape.width}))
    , valueBias(loadFloatTensor(file,
                                encoderLayerPrefix(index) + "self_attn.v_proj.bias",
                                {shape.width},
                                "Linear"))
    , attentionOutputWeight(loadProjectionWeight(file,
                                                 encoderLayerPrefix(index)
                                                     + "self_attn.out_proj.weight",
                                                 {shape.width, shape.width}))
    , attentionOutputBias(
          loadFloatTensor(file,
                          encoderLayerPrefix(index) + "self_attn.out_proj.bias",
                          {shape.width},
                          "Linear"))
    , finalNormWeight(
          loadFloatTensor(file,
                          encoderLayerPrefix(index) + "final_layer_norm.weight",
                          {shape.width},
                          "LayerNorm"))
    , finalNormBias(
          loadFloatTensor(file,
                          encoderLayerPrefix(index) + "final_layer_norm.bias",
                          {shape.width},
                          "LayerNorm"))
    , feedForwardWeight(
          loadProjectionWeight(file,
                               encoderLayerPrefix(index) + "fc1.weight",
                               {shape.feedForwardWidth, shape.width}))
    , feedForwardBias(loadFloatTensor(file,
                                      encoderLayerPrefix(index) + "fc1.bias",
                                      {shape.feedForwardWidth},
                                      "Linear"))
    , feedForwardOutputWeight(
          loadProjectionWeight(file,
                               encoderLayerPrefix(index) + "fc2.weight",
                               {shape.width, shape.feedForwardWidth}))
    , feedForwardOutputBias(loadFloatTensor(
          file, encoderLayerPrefix(index) + "fc2.bias", {shape.width}, "Linear"))
{
}

EncoderWeights::EncoderWeights(const SafeTensors& file,
                               const EncoderShape& shapeToUse)
    : shape(shapeToUse)
    , firstConvolutionWeight(loadFloatTensor(file,
                                             encoderName("conv1.weight"),
                                             {shapeToUse.width,
                                              shapeToUse.melBins,
                                              EncoderShape::convolutionKernelSize},
                                             "Conv1d"))
    , firstConvolutionBias(loadFloatTensor(
          file, encoderName("conv1.bias"), {shapeToUse.width}, "Conv1d"))
    , secondConvolutionWeight(loadFloatTensor(
          file,
          encoderName("conv2.weight"),
          {shapeToUse.width, shapeToUse.width, EncoderShape::convolutionKernelSize},
          "Conv1d"))
    , secondConvolutionBias(loadFloatTensor(
          file, encoderName("conv2.bias"), {shapeToUse.width}, "Conv1d"))
    , positionalEmbedding(loadPositionalEmbedding(file, shapeToUse))
    , finalNormWeight(loadFloatTensor(
          file, encoderName("layer_norm.weight"), {shapeToUse.width}, "LayerNorm"))
    , finalNormBias(loadFloatTensor(
          file, encoderName("layer_norm.bias"), {shapeToUse.width}, "LayerNorm"))
{
    layers.reserve(shapeToUse.layers);

    for (auto index = 0; index < shapeToUse.layers; ++index)
        layers.emplace_back(file, shapeToUse, index);
}
} // namespace WSP
