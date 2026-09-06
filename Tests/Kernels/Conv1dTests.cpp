#include "Common.h"

using namespace nano;
using namespace WSP;
using namespace eacp::GPU;

namespace
{
struct Conv1dShape
{
    int inputChannelCount;
    int inputLength;
    int outputChannelCount;
    int kernelSize;
    int stride;
    int padding;
};

int outputLengthOf(const Conv1dShape& shape)
{
    return conv1dOutputLength(
        shape.inputLength, shape.kernelSize, shape.stride, shape.padding);
}

// PyTorch's formula spelled out, with the padded taps skipped rather than read
// from a zero border: out[co, t] = bias[co] + sum over ci, k of
// w[co, ci, k] * in[ci, t * stride + k - padding]. The output stride pair is
// the same one the kernel takes, so a layout the kernel writes and a layout the
// reference expects cannot drift apart.
Vector<float> conv1dReference(const Vector<float>& input,
                              const Vector<float>& weight,
                              const Vector<float>& bias,
                              const Conv1dShape& shape,
                              int outputChannelStride,
                              int outputFrameStride)
{
    auto outputLength = outputLengthOf(shape);
    auto expected = sized(shape.outputChannelCount * outputLength);

    for (auto channel = 0; channel < shape.outputChannelCount; ++channel)
    {
        for (auto frame = 0; frame < outputLength; ++frame)
        {
            auto total = (double) bias[channel];

            for (auto source = 0; source < shape.inputChannelCount; ++source)
            {
                for (auto tap = 0; tap < shape.kernelSize; ++tap)
                {
                    auto at = frame * shape.stride + tap - shape.padding;

                    if (at < 0 || at >= shape.inputLength)
                        continue;

                    auto weightAt = (channel * shape.inputChannelCount + source)
                                        * shape.kernelSize
                                    + tap;

                    total += (double) weight[weightAt]
                             * input[source * shape.inputLength + at];
                }
            }

            expected[channel * outputChannelStride + frame * outputFrameStride] =
                (float) total;
        }
    }

    return expected;
}

Vector<float> runConv1d(const Vector<float>& input,
                        const Vector<float>& weight,
                        const Vector<float>& bias,
                        const Conv1dShape& shape,
                        int outputChannelStride,
                        int outputFrameStride)
{
    auto outputLength = outputLengthOf(shape);

    auto inputBuffer = storageOf(input);
    auto weightBuffer = storageOf(weight);
    auto biasBuffer = storageOf(bias);
    auto output = outputFor(shape.outputChannelCount * outputLength);

    auto kernel = Conv1d {};
    kernel.input = inputBuffer;
    kernel.weight = weightBuffer;
    kernel.bias = biasBuffer;
    kernel.output = output;
    kernel.inputChannelCount = (unsigned) shape.inputChannelCount;
    kernel.inputLength = (unsigned) shape.inputLength;
    kernel.kernelSize = (unsigned) shape.kernelSize;
    kernel.stride = (unsigned) shape.stride;
    kernel.padding = (unsigned) shape.padding;
    kernel.outputChannelStride = (unsigned) outputChannelStride;
    kernel.outputFrameStride = (unsigned) outputFrameStride;

    return runOverGrid(kernel, output, outputLength, shape.outputChannelCount);
}

void checkMatches(const Vector<float>& result, const Vector<float>& expected)
{
    check(result.size() == expected.size());

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
}

// Whisper's conv1 in miniature: nothing here divides by the 8 x 8 dispatch
// group, and no two of the counts are equal, so a thread that confused the
// weight's channel stride for its tap stride, or leaned on the grid landing
// exactly on the output, fails here rather than on the first real model.
constexpr auto conv1Shape = Conv1dShape {5, 11, 7, 3, 1, 1};
} // namespace

auto tConv1dStrideOneMatchesCpu = test("Kernels/conv1dStrideOneMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    auto outputLength = outputLengthOf(conv1Shape);
    check(outputLength == conv1Shape.inputLength);

    auto input = spreadValues(
        conv1Shape.inputChannelCount * conv1Shape.inputLength, 90210u, 2.f);

    auto weight =
        spreadValues(conv1Shape.outputChannelCount * conv1Shape.inputChannelCount
                         * conv1Shape.kernelSize,
                     13579u,
                     1.f);

    auto bias = spreadValues(conv1Shape.outputChannelCount, 24680u, 0.5f);

    auto result = runConv1d(input, weight, bias, conv1Shape, outputLength, 1);
    auto expected =
        conv1dReference(input, weight, bias, conv1Shape, outputLength, 1);

    checkMatches(result, expected);
};

// Stride 2 with padding 1, which is conv2's shape and the one that decides how
// long the encoder's sequence is: an input length that is not a multiple of the
// stride, so an output length rounded the wrong way is off by one rather than
// exact by luck.
auto tConv1dStrideTwoMatchesCpu = test("Kernels/conv1dStrideTwoMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto shape = Conv1dShape {3, 11, 5, 3, 2, 1};

    auto outputLength = outputLengthOf(shape);
    check(outputLength == 6);

    auto input =
        spreadValues(shape.inputChannelCount * shape.inputLength, 11111u, 2.f);

    auto weight = spreadValues(shape.outputChannelCount * shape.inputChannelCount
                                   * shape.kernelSize,
                               22222u,
                               1.f);

    auto bias = spreadValues(shape.outputChannelCount, 33333u, 0.5f);

    auto result = runConv1d(input, weight, bias, shape, outputLength, 1);
    auto expected = conv1dReference(input, weight, bias, shape, outputLength, 1);

    checkMatches(result, expected);
};

// The same convolution written out both ways. Frame-major is the [T, C] the
// transformer reads, and the only difference between the two dispatches is the
// stride pair — so the two results have to be transposes of one another, which
// is asserted here directly rather than only through the reference.
auto tConv1dWritesEitherLayout = test("Kernels/conv1dWritesEitherLayout") = []
{
    if (!Device::shared().isValid())
        return;

    auto outputLength = outputLengthOf(conv1Shape);

    auto input = spreadValues(
        conv1Shape.inputChannelCount * conv1Shape.inputLength, 4711u, 2.f);

    auto weight =
        spreadValues(conv1Shape.outputChannelCount * conv1Shape.inputChannelCount
                         * conv1Shape.kernelSize,
                     1471u,
                     1.f);

    auto bias = spreadValues(conv1Shape.outputChannelCount, 7114u, 0.5f);

    auto channelMajor = runConv1d(input, weight, bias, conv1Shape, outputLength, 1);
    auto frameMajor =
        runConv1d(input, weight, bias, conv1Shape, 1, conv1Shape.outputChannelCount);

    checkMatches(channelMajor,
                 conv1dReference(input, weight, bias, conv1Shape, outputLength, 1));

    checkMatches(
        frameMajor,
        conv1dReference(
            input, weight, bias, conv1Shape, 1, conv1Shape.outputChannelCount));

    for (auto channel = 0; channel < conv1Shape.outputChannelCount; ++channel)
        for (auto frame = 0; frame < outputLength; ++frame)
            check(frameMajor[frame * conv1Shape.outputChannelCount + channel]
                  == channelMajor[channel * outputLength + frame]);
};

// The encoder's front-end at small shapes: mel in, conv1 channel-major so
// conv2 can read it, conv2 frame-major so the transformer can. Both dispatches
// go into one pass, which is how the encoder will run them — conv1's output
// buffer is bound as conv2's input rather than making the round trip through
// the host. GELU sits between the two in the model and is deliberately absent
// here: it is the caller's elementwise kernel, not this one's business.
auto tConv1dChainMatchesCpu = test("Kernels/conv1dChainMatchesCpu") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto first = Conv1dShape {5, 9, 6, 3, 1, 1};
    constexpr auto second = Conv1dShape {6, 9, 6, 3, 2, 1};

    auto firstLength = outputLengthOf(first);
    auto secondLength = outputLengthOf(second);

    check(firstLength == second.inputLength);
    check(secondLength == 5);

    auto input =
        spreadValues(first.inputChannelCount * first.inputLength, 606u, 2.f);

    auto firstWeight = spreadValues(first.outputChannelCount
                                        * first.inputChannelCount * first.kernelSize,
                                    607u,
                                    1.f);

    auto firstBias = spreadValues(first.outputChannelCount, 608u, 0.5f);

    auto secondWeight = spreadValues(
        second.outputChannelCount * second.inputChannelCount * second.kernelSize,
        609u,
        1.f);

    auto secondBias = spreadValues(second.outputChannelCount, 610u, 0.5f);

    auto inputBuffer = storageOf(input);
    auto firstWeightBuffer = storageOf(firstWeight);
    auto firstBiasBuffer = storageOf(firstBias);
    auto secondWeightBuffer = storageOf(secondWeight);
    auto secondBiasBuffer = storageOf(secondBias);

    auto firstOutput = outputFor(first.outputChannelCount * firstLength);
    auto secondOutput = outputFor(second.outputChannelCount * secondLength);

    auto kernel = Conv1d {};
    kernel.prepare();

    auto commands = device.makeCommandBuffer();

    {
        auto pass = commands.beginCompute();

        kernel.input = inputBuffer;
        kernel.weight = firstWeightBuffer;
        kernel.bias = firstBiasBuffer;
        kernel.output = firstOutput;
        kernel.inputChannelCount = (unsigned) first.inputChannelCount;
        kernel.inputLength = (unsigned) first.inputLength;
        kernel.kernelSize = (unsigned) first.kernelSize;
        kernel.stride = (unsigned) first.stride;
        kernel.padding = (unsigned) first.padding;
        kernel.outputChannelStride = (unsigned) firstLength;
        kernel.outputFrameStride = 1u;
        pass.dispatch(kernel, firstLength, first.outputChannelCount);

        kernel.input = firstOutput;
        kernel.weight = secondWeightBuffer;
        kernel.bias = secondBiasBuffer;
        kernel.output = secondOutput;
        kernel.inputChannelCount = (unsigned) second.inputChannelCount;
        kernel.inputLength = (unsigned) second.inputLength;
        kernel.kernelSize = (unsigned) second.kernelSize;
        kernel.stride = (unsigned) second.stride;
        kernel.padding = (unsigned) second.padding;
        kernel.outputChannelStride = 1u;
        kernel.outputFrameStride = (unsigned) second.outputChannelCount;
        pass.dispatch(kernel, secondLength, second.outputChannelCount);
    }

    commands.commit();

    auto intermediate =
        conv1dReference(input, firstWeight, firstBias, first, firstLength, 1);

    auto expected = conv1dReference(intermediate,
                                    secondWeight,
                                    secondBias,
                                    second,
                                    1,
                                    second.outputChannelCount);

    checkMatches(readBack(firstOutput, first.outputChannelCount * firstLength),
                 intermediate);

    checkMatches(readBack(secondOutput, second.outputChannelCount * secondLength),
                 expected);
};
