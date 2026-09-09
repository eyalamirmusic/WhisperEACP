#include "Encoder.h"

#include <algorithm>
#include <cstdint>

namespace WSP
{
using eacp::GPU::Buffer;
using eacp::GPU::BufferUsage;
using eacp::GPU::ComputePass;
using eacp::GPU::Device;

namespace
{
constexpr int floatBytes(int elementCount)
{
    return elementCount * (int) sizeof(float);
}

Buffer allocate(Device& device, int elementCount)
{
    return device.makeBuffer(floatBytes(elementCount), BufferUsage::Storage);
}

Buffer allocateZeroed(Device& device, int elementCount)
{
    auto zeroes = Vector<float> {};
    zeroes.resize(elementCount);

    for (auto index = 0; index < elementCount; ++index)
        zeroes[index] = 0.f;

    return device.makeBuffer(
        zeroes.data(), floatBytes(elementCount), BufferUsage::Storage);
}

// The projection programs differ only in how they read the weight and in which
// tiling they compute the product with, so the bind is written once against
// whichever of them the caller picked: every one of them takes a
// TiledMatMulShape.
template <typename Program>
void dispatchLinear(Program& program,
                    ComputePass& pass,
                    const Buffer& input,
                    const Buffer& weight,
                    const Buffer& bias,
                    const Buffer& target,
                    int innerCount,
                    int outputWidth,
                    int rowCount,
                    bool gelu,
                    bool residual)
{
    program.a = input;
    program.b = weight;
    program.bias = bias;
    program.output = target;

    auto shape = TiledMatMulShape::forLinear(rowCount, innerCount, outputWidth);
    shape.gelu = gelu;
    shape.residual = residual;

    program.dispatch(pass, shape);
}
} // namespace

Encoder::Encoder(const EncoderShape& shapeToUse)
    : encoderShape(shapeToUse)
{
}

void Encoder::prepare(Device& device)
{
    unfold.prepare(device);
    sum.prepare(device);
    normalisation.prepare(device);
    projection.prepare(device);
    packedProjection.prepare(device);
    softmax.prepare(device);
    scores.prepare(device);
    attention.prepare(device);

    const auto convolutionElements = encoderShape.convolutionElementCount();
    const auto elements = encoderShape.elementCount();
    constexpr auto taps = EncoderShape::convolutionKernelSize;

    columns.emplace(allocate(
        device,
        std::max(encoderShape.convolutionFrames() * encoderShape.melBins * taps,
                 encoderShape.positions() * encoderShape.width * taps)));
    convolved.emplace(allocate(device, convolutionElements));

    hidden.emplace(allocate(device, elements));
    normalised.emplace(allocate(device, elements));
    queries.emplace(allocate(device, elements));
    keys.emplace(allocate(device, elements));
    values.emplace(allocate(device, elements));
    attended.emplace(allocate(device, elements));

    attentionScores.emplace(allocate(device, encoderShape.scoreElementCount()));
    feedForward.emplace(allocate(device, encoderShape.feedForwardElementCount()));

    zeroBias.emplace(allocateZeroed(device,
                                    std::max({encoderShape.width,
                                              encoderShape.feedForwardWidth,
                                              encoderShape.positions()})));
}

void Encoder::prepare()
{
    prepare(Device::shared());
}

void Encoder::encodeSum(ComputePass& pass,
                        const Buffer& stream,
                        const Buffer& addend)
{
    sum.output = stream;
    sum.addend = addend;

    pass.dispatch(sum, encoderShape.elementCount());
}

void Encoder::encodeLayerNorm(ComputePass& pass,
                              const Buffer& input,
                              const TensorBuffer& weight,
                              const TensorBuffer& bias,
                              const Buffer& target)
{
    normalisation.input = input;
    normalisation.weight = weight.buffer;
    normalisation.bias = bias.buffer;
    normalisation.output = target;
    normalisation.rowLength = (std::uint32_t) encoderShape.width;

    normalisation.dispatchRows(pass, encoderShape.positions());
}

// The one place a weight's storage decides anything: a tensor the loader left
// packed goes to the program that reads packed halves, and one it uploaded as
// floats to the program that subscripts floats. Neither can read the other's
// buffer, which is why the choice is made from the buffer rather than from a
// build-time switch.
void Encoder::encodeLinear(ComputePass& pass,
                           const Buffer& input,
                           const TensorBuffer& weight,
                           const Buffer& bias,
                           const Buffer& target,
                           int innerCount,
                           int outputWidth,
                           int rowCount,
                           bool gelu,
                           bool residual)
{
    if (weight.isPackedHalf())
        dispatchLinear(packedProjection,
                       pass,
                       input,
                       weight.buffer,
                       bias,
                       target,
                       innerCount,
                       outputWidth,
                       rowCount,
                       gelu,
                       residual);
    else
        dispatchLinear(projection,
                       pass,
                       input,
                       weight.buffer,
                       bias,
                       target,
                       innerCount,
                       outputWidth,
                       rowCount,
                       gelu,
                       residual);
}

void Encoder::encodeUnfold(ComputePass& pass,
                           const Buffer& input,
                           int inputChannelCount,
                           int inputLength,
                           int stride,
                           int inputChannelStride,
                           int inputFrameStride,
                           int outputLength)
{
    constexpr auto taps = EncoderShape::convolutionKernelSize;

    unfold.input = input;
    unfold.columns = *columns;
    unfold.inputChannelCount = (std::uint32_t) inputChannelCount;
    unfold.inputLength = (std::uint32_t) inputLength;
    unfold.kernelSize = (std::uint32_t) taps;
    unfold.stride = (std::uint32_t) stride;
    unfold.padding = (std::uint32_t) EncoderShape::convolutionPadding;
    unfold.inputChannelStride = (std::uint32_t) inputChannelStride;
    unfold.inputFrameStride = (std::uint32_t) inputFrameStride;

    pass.dispatch(unfold, inputChannelCount * taps, outputLength);
}

// Each convolution is its windows unfolded into rows and one tiled product of
// them against the weight as the file lays it out, [out, in, k] being
// [out, in * k], with the GELU on the store. conv1 reads the mel band-major
// and writes frame-major, which is what conv2 unfolds at a stride of two and
// what the transformer reads — HuggingFace's permute after the second
// convolution, done by the first one's store.
void Encoder::encodeFrontEnd(ComputePass& pass,
                             const Buffer& mel,
                             const EncoderWeights& weights)
{
    const auto convolutionFrames = encoderShape.convolutionFrames();
    const auto positions = encoderShape.positions();
    const auto width = encoderShape.width;
    constexpr auto taps = EncoderShape::convolutionKernelSize;

    encodeUnfold(pass,
                 mel,
                 encoderShape.melBins,
                 encoderShape.inputFrames,
                 EncoderShape::firstConvolutionStride,
                 encoderShape.inputFrames,
                 1,
                 convolutionFrames);

    pass.barrier();

    encodeLinear(pass,
                 *columns,
                 weights.firstConvolutionWeight,
                 weights.firstConvolutionBias.buffer,
                 *convolved,
                 encoderShape.melBins * taps,
                 width,
                 convolutionFrames,
                 true,
                 false);

    pass.barrier();

    encodeUnfold(pass,
                 *convolved,
                 width,
                 convolutionFrames,
                 EncoderShape::secondConvolutionStride,
                 1,
                 width,
                 positions);

    pass.barrier();

    encodeLinear(pass,
                 *columns,
                 weights.secondConvolutionWeight,
                 weights.secondConvolutionBias.buffer,
                 *hidden,
                 width * taps,
                 width,
                 positions,
                 true,
                 false);

    pass.barrier();

    encodeSum(pass, *hidden, weights.positionalEmbedding.buffer);
}

void Encoder::encodeAttention(ComputePass& pass, const EncoderLayerWeights& weights)
{
    const auto width = encoderShape.width;
    const auto positions = encoderShape.positions();

    encodeLinear(pass,
                 *normalised,
                 weights.queryWeight,
                 weights.queryBias.buffer,
                 *queries,
                 width,
                 width,
                 positions);

    encodeLinear(pass,
                 *normalised,
                 weights.keyWeight,
                 *zeroBias,
                 *keys,
                 width,
                 width,
                 positions);

    encodeLinear(pass,
                 *normalised,
                 weights.valueWeight,
                 weights.valueBias.buffer,
                 *values,
                 width,
                 width,
                 positions);

    // The three projections above read the same normalised rows and write three
    // buffers nothing else has touched, so they are the one place in a layer
    // where a concurrent pass has anything to overlap. Everything from here on
    // reads what the dispatch before it wrote.
    pass.barrier();

    // The scores are a product of the queries against the keys, one batch per
    // head over the head's columns, in the same layout as the projections: a
    // key row is contiguous along the head dimension exactly as a weight row
    // is along its inputs.
    scores.a = *queries;
    scores.b = *keys;
    scores.bias = *zeroBias;
    scores.output = *attentionScores;

    scores.dispatch(
        pass,
        TiledMatMulShape::forAttentionScores(positions,
                                             positions,
                                             encoderShape.heads,
                                             encoderShape.headWidth(),
                                             width,
                                             encoderShape.attentionScale(),
                                             false));

    pass.barrier();

    softmax.values = *attentionScores;
    softmax.rowLength = (std::uint32_t) positions;

    softmax.dispatchRows(pass, encoderShape.scoreRowCount());

    pass.barrier();

    attention.a = *attentionScores;
    attention.b = *values;
    attention.bias = *zeroBias;
    attention.output = *attended;

    attention.dispatch(pass,
                       TiledMatMulShape::forAttentionApply(positions,
                                                           positions,
                                                           encoderShape.heads,
                                                           encoderShape.headWidth(),
                                                           width));
}

// Pre-norm, which is what Whisper is: the block normalises what it reads and
// adds what it computed to what it was given, so the residual is the
// unnormalised input, and hidden is added to in place by each sublayer's last
// projection.
void Encoder::encodeLayer(ComputePass& pass, const EncoderLayerWeights& weights)
{
    const auto width = encoderShape.width;
    const auto positions = encoderShape.positions();

    encodeLayerNorm(pass,
                    *hidden,
                    weights.attentionNormWeight,
                    weights.attentionNormBias,
                    *normalised);

    pass.barrier();

    encodeAttention(pass, weights);

    pass.barrier();

    encodeLinear(pass,
                 *attended,
                 weights.attentionOutputWeight,
                 weights.attentionOutputBias.buffer,
                 *hidden,
                 width,
                 width,
                 positions,
                 false,
                 true);

    pass.barrier();

    encodeLayerNorm(
        pass, *hidden, weights.finalNormWeight, weights.finalNormBias, *normalised);

    pass.barrier();

    encodeLinear(pass,
                 *normalised,
                 weights.feedForwardWeight,
                 weights.feedForwardBias.buffer,
                 *feedForward,
                 width,
                 encoderShape.feedForwardWidth,
                 positions,
                 true,
                 false);

    pass.barrier();

    encodeLinear(pass,
                 *feedForward,
                 weights.feedForwardOutputWeight,
                 weights.feedForwardOutputBias.buffer,
                 *hidden,
                 encoderShape.feedForwardWidth,
                 width,
                 positions,
                 false,
                 true);
}

void Encoder::encode(ComputePass& pass,
                     const Buffer& mel,
                     const EncoderWeights& weights,
                     const Buffer& output)
{
    if (!(weights.shape == encoderShape))
        throw ModelError {"the weights were loaded against a different encoder "
                          "shape than this encoder was built for"};

    encodeFrontEnd(pass, mel, weights);

    for (auto index = 0; index < encoderShape.layers; ++index)
    {
        pass.barrier();
        encodeLayer(pass, weights.layers[index]);
    }

    pass.barrier();

    encodeLayerNorm(
        pass, *hidden, weights.finalNormWeight, weights.finalNormBias, output);
}
} // namespace WSP
