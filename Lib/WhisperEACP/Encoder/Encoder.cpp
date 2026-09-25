#include "Encoder.h"

#include <algorithm>
#include <string>

namespace WSP
{
using eacp::GPU::Buffer;
using eacp::GPU::BufferRange;
using eacp::GPU::ComputePass;
using eacp::GPU::Device;

namespace
{
// The three projections read the same normalised rows and write three values
// nothing else has touched, so they are the one place in a layer where a
// concurrent pass has anything to overlap.
Tensor selfAttention(Net& net,
                     const EncoderShape& shape,
                     const EncoderLayerWeights& weights,
                     const Tensor& normalised)
{
    const auto queries = net.linear(
        normalised, weights.queryWeight, &weights.queryBias, Activation::none);
    const auto keys =
        net.linear(normalised, weights.keyWeight, nullptr, Activation::none);
    const auto values = net.linear(
        normalised, weights.valueWeight, &weights.valueBias, Activation::none);

    return net.attention(queries, keys, values, shape.heads, false);
}

// Pre-norm, which is what Whisper is: the block normalises what it reads and
// adds what it computed to what it was given, so the residual is the
// unnormalised input.
Tensor encoderLayer(Net& net,
                    const EncoderShape& shape,
                    const EncoderLayerWeights& weights,
                    const Tensor& hidden)
{
    const auto attended =
        net.linearAdd(selfAttention(net,
                                    shape,
                                    weights,
                                    net.layerNorm(hidden,
                                                  weights.attentionNormWeight,
                                                  weights.attentionNormBias)),
                      weights.attentionOutputWeight,
                      &weights.attentionOutputBias,
                      hidden);

    return net.linearAdd(
        net.linear(
            net.layerNorm(attended, weights.finalNormWeight, weights.finalNormBias),
            weights.feedForwardWeight,
            &weights.feedForwardBias,
            Activation::gelu),
        weights.feedForwardOutputWeight,
        &weights.feedForwardOutputBias,
        attended);
}
} // namespace

// Each convolution carries its padding and the GELU on its store, and reads its
// input frame-major: the mel is turned from band-major before the first, and
// the first writes frame-major, which is HuggingFace's permute after the
// second convolution done by the first one's store.
void recordEncoder(Net& net,
                   const EncoderShape& shape,
                   const EncoderWeights& weights,
                   const Binding& mel,
                   const Binding& rows,
                   int inputFrames)
{
    const auto frames =
        net.transpose(net.input(mel, {shape.melBins, inputFrames}, DType::float32));

    auto hidden = net.conv1d(net.conv1d(frames,
                                        weights.firstConvolutionWeight,
                                        weights.firstConvolutionBias,
                                        EncoderShape::firstConvolutionStride,
                                        EncoderShape::convolutionPadding,
                                        Activation::gelu),
                             weights.secondConvolutionWeight,
                             weights.secondConvolutionBias,
                             EncoderShape::secondConvolutionStride,
                             EncoderShape::convolutionPadding,
                             Activation::gelu);

    hidden = net.add(
        hidden,
        net.rows(weights.positionalEmbedding, 0, shape.positions(inputFrames)));

    for (auto index = 0; index < shape.layers; ++index)
        hidden = encoderLayer(net, shape, weights.layers[index], hidden);

    net.output(net.layerNorm(hidden, weights.finalNormWeight, weights.finalNormBias),
               rows);
}

Encoder::Encoder(const EncoderShape& shapeToUse)
    : encoderShape(shapeToUse)
{
}

void Encoder::prepare(Device& device)
{
    constexpr auto taps = EncoderShape::convolutionKernelSize;
    const auto positions = encoderShape.positions();
    const auto width = encoderShape.width;

    auto layout = KernelNetLayout {};
    layout.heads = encoderShape.heads;
    layout.headWidth = encoderShape.headWidth();
    layout.widestLayerOutput = encoderShape.feedForwardWidth;
    layout.columnElements =
        std::max(encoderShape.convolutionFrames() * encoderShape.melBins * taps,
                 positions * width * taps);

    layout.scratch.add(
        ScratchReservation {encoderShape.convolutionElementCount(), 1});
    layout.scratch.add(ScratchReservation {encoderShape.elementCount(), 6});
    layout.scratch.add(
        ScratchReservation {encoderShape.feedForwardElementCount(), 1});
    layout.attention.add(AttentionReservation {positions, positions});
    layout.zeroBiasLengths.add(
        std::max({width, encoderShape.feedForwardWidth, positions}));

    net.prepare(device, layout);
}

void Encoder::prepare()
{
    prepare(Device::shared());
}

void Encoder::encode(ComputePass& pass,
                     const Buffer& mel,
                     const EncoderWeights& weights,
                     const Buffer& output,
                     int positionCount)
{
    if (!(weights.shape == encoderShape))
        throw ModelError {"the weights were loaded against a different encoder "
                          "shape than this encoder was built for"};

    if (positionCount < 0 || positionCount > encoderShape.positions())
        throw ModelError {
            "an encoder built for " + std::to_string(encoderShape.positions())
            + " positions was asked to run over " + std::to_string(positionCount)};

    const auto inputFrames = positionCount == 0
                                 ? encoderShape.inputFrames
                                 : encoderShape.framesForPositions(positionCount);

    const auto melBinding =
        Binding {BufferRange::of(mel),
                 Shape {encoderShape.melBins, encoderShape.inputFrames}};

    net.begin(pass);
    recordEncoder(net,
                  encoderShape,
                  weights,
                  melBinding,
                  Binding {BufferRange::of(output), {}},
                  inputFrames);
    net.end();
}
} // namespace WSP
