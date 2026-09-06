#include "Encoder.h"

#include <cstdint>

namespace WSP
{
using eacp::GPU::Buffer;
using eacp::GPU::BufferUsage;
using eacp::GPU::CommandBuffer;
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

// The two Linear programs differ only in how they read the weight, so the bind
// is written once against whichever of them the caller picked.
template <WeightStorage weightStorage>
void dispatchLinear(LinearProgram<weightStorage>& program,
                    CommandBuffer& commands,
                    const Buffer& input,
                    const Buffer& weight,
                    const Buffer& bias,
                    const Buffer& target,
                    int innerCount,
                    int outputWidth,
                    int rowCount)
{
    program.input = input;
    program.weights = weight;
    program.bias = bias;
    program.output = target;
    program.innerCount = (std::uint32_t) innerCount;
    program.outputWidth = (std::uint32_t) outputWidth;

    auto pass = commands.beginCompute();
    pass.dispatch(program, outputWidth, rowCount);
}
} // namespace

Encoder::Encoder(const EncoderShape& shapeToUse)
    : encoderShape(shapeToUse)
{
}

void Encoder::prepare(Device& device)
{
    convolution.prepare(device);
    activation.prepare(device);
    sum.prepare(device);
    normalisation.prepare(device);
    projection.prepare(device);
    packedProjection.prepare(device);
    scores.prepare(device);
    softmax.prepare(device);
    attention.prepare(device);

    const auto convolutionElements = encoderShape.convolutionElementCount();
    const auto elements = encoderShape.elementCount();

    convolved.emplace(allocate(device, convolutionElements));
    activated.emplace(allocate(device, convolutionElements));

    projected.emplace(allocate(device, elements));
    activatedProjection.emplace(allocate(device, elements));
    hidden.emplace(allocate(device, elements));
    residual.emplace(allocate(device, elements));
    normalised.emplace(allocate(device, elements));
    queries.emplace(allocate(device, elements));
    keys.emplace(allocate(device, elements));
    values.emplace(allocate(device, elements));
    attended.emplace(allocate(device, elements));
    attentionOutput.emplace(allocate(device, elements));
    feedForwardOutput.emplace(allocate(device, elements));

    attentionScores.emplace(allocate(device, encoderShape.scoreElementCount()));
    attentionWeights.emplace(allocate(device, encoderShape.scoreElementCount()));

    const auto feedForwardElements = encoderShape.feedForwardElementCount();
    feedForward.emplace(allocate(device, feedForwardElements));
    activatedFeedForward.emplace(allocate(device, feedForwardElements));

    zeroBias.emplace(allocateZeroed(device, encoderShape.width));
}

void Encoder::prepare()
{
    prepare(Device::shared());
}

void Encoder::encodeGelu(CommandBuffer& commands,
                         const Buffer& input,
                         const Buffer& target,
                         int elementCount)
{
    activation.input = input;
    activation.output = target;

    auto pass = commands.beginCompute();
    pass.dispatch(activation, elementCount);
}

void Encoder::encodeSum(CommandBuffer& commands,
                        const Buffer& left,
                        const Buffer& right,
                        const Buffer& target)
{
    sum.a = left;
    sum.b = right;
    sum.output = target;

    auto pass = commands.beginCompute();
    pass.dispatch(sum, encoderShape.elementCount());
}

void Encoder::encodeLayerNorm(CommandBuffer& commands,
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

    auto pass = commands.beginCompute();
    pass.dispatch(normalisation, encoderShape.positions());
}

// The one place a weight's storage decides anything: a tensor the loader left
// packed goes to the program that reads packed halves, and one it uploaded as
// floats to the program that subscripts floats. Neither can read the other's
// buffer, which is why the choice is made from the buffer rather than from a
// build-time switch.
void Encoder::encodeLinear(CommandBuffer& commands,
                           const Buffer& input,
                           const TensorBuffer& weight,
                           const Buffer& bias,
                           const Buffer& target,
                           int innerCount,
                           int outputWidth)
{
    const auto rowCount = encoderShape.positions();

    if (weight.isPackedHalf())
        dispatchLinear(packedProjection,
                       commands,
                       input,
                       weight.buffer,
                       bias,
                       target,
                       innerCount,
                       outputWidth,
                       rowCount);
    else
        dispatchLinear(projection,
                       commands,
                       input,
                       weight.buffer,
                       bias,
                       target,
                       innerCount,
                       outputWidth,
                       rowCount);
}

// conv1 keeps the mel's band-major layout, so its output strides are the ones
// conv2 reads at. conv2's turn it frame-major in the store — HuggingFace's
// permute after the second convolution, done by the write rather than by a pass
// of its own.
void Encoder::encodeFrontEnd(CommandBuffer& commands,
                             const Buffer& mel,
                             const EncoderWeights& weights)
{
    const auto convolutionFrames = encoderShape.convolutionFrames();
    const auto positions = encoderShape.positions();

    convolution.input = mel;
    convolution.weight = weights.firstConvolutionWeight.buffer;
    convolution.bias = weights.firstConvolutionBias.buffer;
    convolution.output = *convolved;
    convolution.inputChannelCount = (std::uint32_t) encoderShape.melBins;
    convolution.inputLength = (std::uint32_t) encoderShape.inputFrames;
    convolution.kernelSize = (std::uint32_t) EncoderShape::convolutionKernelSize;
    convolution.stride = (std::uint32_t) EncoderShape::firstConvolutionStride;
    convolution.padding = (std::uint32_t) EncoderShape::convolutionPadding;
    convolution.outputChannelStride = (std::uint32_t) convolutionFrames;
    convolution.outputFrameStride = 1u;

    {
        auto pass = commands.beginCompute();
        pass.dispatch(convolution, convolutionFrames, encoderShape.width);
    }

    encodeGelu(
        commands, *convolved, *activated, encoderShape.convolutionElementCount());

    convolution.input = *activated;
    convolution.weight = weights.secondConvolutionWeight.buffer;
    convolution.bias = weights.secondConvolutionBias.buffer;
    convolution.output = *projected;
    convolution.inputChannelCount = (std::uint32_t) encoderShape.width;
    convolution.inputLength = (std::uint32_t) convolutionFrames;
    convolution.stride = (std::uint32_t) EncoderShape::secondConvolutionStride;
    convolution.outputChannelStride = 1u;
    convolution.outputFrameStride = (std::uint32_t) encoderShape.width;

    {
        auto pass = commands.beginCompute();
        pass.dispatch(convolution, positions, encoderShape.width);
    }

    encodeGelu(
        commands, *projected, *activatedProjection, encoderShape.elementCount());

    encodeSum(
        commands, *activatedProjection, weights.positionalEmbedding.buffer, *hidden);
}

void Encoder::encodeAttention(CommandBuffer& commands,
                              const EncoderLayerWeights& weights)
{
    const auto width = encoderShape.width;
    const auto positions = encoderShape.positions();

    encodeLinear(commands,
                 *normalised,
                 weights.queryWeight,
                 weights.queryBias.buffer,
                 *queries,
                 width,
                 width);

    encodeLinear(
        commands, *normalised, weights.keyWeight, *zeroBias, *keys, width, width);

    encodeLinear(commands,
                 *normalised,
                 weights.valueWeight,
                 weights.valueBias.buffer,
                 *values,
                 width,
                 width);

    scores.queries = *queries;
    scores.keys = *keys;
    scores.scores = *attentionScores;
    scores.modelWidth = (std::uint32_t) width;
    scores.headWidth = (std::uint32_t) encoderShape.headWidth();
    scores.queryCount = (std::uint32_t) positions;
    scores.keyCount = (std::uint32_t) positions;
    scores.scale = encoderShape.attentionScale();

    {
        auto pass = commands.beginCompute();
        pass.dispatch(scores, positions, encoderShape.scoreRowCount());
    }

    softmax.input = *attentionScores;
    softmax.output = *attentionWeights;
    softmax.rowLength = (std::uint32_t) positions;

    {
        auto pass = commands.beginCompute();
        pass.dispatch(softmax, encoderShape.scoreRowCount());
    }

    attention.probabilities = *attentionWeights;
    attention.values = *values;
    attention.output = *attended;
    attention.modelWidth = (std::uint32_t) width;
    attention.headWidth = (std::uint32_t) encoderShape.headWidth();
    attention.queryCount = (std::uint32_t) positions;
    attention.keyCount = (std::uint32_t) positions;

    {
        auto pass = commands.beginCompute();
        pass.dispatch(attention, width, positions);
    }
}

// Pre-norm, which is what Whisper is: the block normalises what it reads and
// adds what it computed to what it was given, so the residual is the
// unnormalised input. hidden and residual take turns holding it rather than one
// buffer being read and written by the same dispatch.
void Encoder::encodeLayer(CommandBuffer& commands,
                          const EncoderLayerWeights& weights)
{
    const auto width = encoderShape.width;

    encodeLayerNorm(commands,
                    *hidden,
                    weights.attentionNormWeight,
                    weights.attentionNormBias,
                    *normalised);

    encodeAttention(commands, weights);

    encodeLinear(commands,
                 *attended,
                 weights.attentionOutputWeight,
                 weights.attentionOutputBias.buffer,
                 *attentionOutput,
                 width,
                 width);

    encodeSum(commands, *hidden, *attentionOutput, *residual);

    encodeLayerNorm(commands,
                    *residual,
                    weights.finalNormWeight,
                    weights.finalNormBias,
                    *normalised);

    encodeLinear(commands,
                 *normalised,
                 weights.feedForwardWeight,
                 weights.feedForwardBias.buffer,
                 *feedForward,
                 width,
                 encoderShape.feedForwardWidth);

    encodeGelu(commands,
               *feedForward,
               *activatedFeedForward,
               encoderShape.feedForwardElementCount());

    encodeLinear(commands,
                 *activatedFeedForward,
                 weights.feedForwardOutputWeight,
                 weights.feedForwardOutputBias.buffer,
                 *feedForwardOutput,
                 encoderShape.feedForwardWidth,
                 width);

    encodeSum(commands, *residual, *feedForwardOutput, *hidden);
}

void Encoder::encode(CommandBuffer& commands,
                     const Buffer& mel,
                     const EncoderWeights& weights,
                     const Buffer& output)
{
    if (!(weights.shape == encoderShape))
        throw ModelError {"the weights were loaded against a different encoder "
                          "shape than this encoder was built for"};

    encodeFrontEnd(commands, mel, weights);

    for (auto index = 0; index < encoderShape.layers; ++index)
        encodeLayer(commands, weights.layers[index]);

    encodeLayerNorm(
        commands, *hidden, weights.finalNormWeight, weights.finalNormBias, output);
}
} // namespace WSP
