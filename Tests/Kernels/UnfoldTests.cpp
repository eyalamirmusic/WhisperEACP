#include "Common.h"

using namespace nano;
using namespace WSP;
using namespace eacp::GPU;

namespace
{
struct UnfoldShape
{
    int inputChannelCount;
    int inputLength;
    int kernelSize;
    int stride;
    int padding;
    int inputChannelStride;
    int inputFrameStride;

    int outputLength() const
    {
        return conv1dOutputLength(inputLength, kernelSize, stride, padding);
    }

    int columnCount() const { return inputChannelCount * kernelSize; }
};

// The definition, with the taps outside the input written as zero.
Vector<float> unfoldReference(const Vector<float>& input, const UnfoldShape& shape)
{
    auto expected = sized(shape.outputLength() * shape.columnCount());

    for (auto frame = 0; frame < shape.outputLength(); ++frame)
        for (auto channel = 0; channel < shape.inputChannelCount; ++channel)
            for (auto tap = 0; tap < shape.kernelSize; ++tap)
            {
                auto at = frame * shape.stride + tap - shape.padding;
                auto inside = at >= 0 && at < shape.inputLength;
                auto value = inside ? input[channel * shape.inputChannelStride
                                            + at * shape.inputFrameStride]
                                    : 0.f;

                expected[frame * shape.columnCount() + channel * shape.kernelSize
                         + tap] = value;
            }

    return expected;
}

Vector<float> runUnfold(const Vector<float>& input, const UnfoldShape& shape)
{
    auto inputBuffer = storageOf(input);
    auto columns = outputFor(shape.outputLength() * shape.columnCount());

    auto kernel = Unfold {};
    kernel.input = inputBuffer;
    kernel.columns = columns;
    kernel.inputChannelCount = (unsigned) shape.inputChannelCount;
    kernel.inputLength = (unsigned) shape.inputLength;
    kernel.kernelSize = (unsigned) shape.kernelSize;
    kernel.stride = (unsigned) shape.stride;
    kernel.padding = (unsigned) shape.padding;
    kernel.inputChannelStride = (unsigned) shape.inputChannelStride;
    kernel.inputFrameStride = (unsigned) shape.inputFrameStride;

    return runOverGrid(kernel, columns, shape.columnCount(), shape.outputLength());
}

// PyTorch's convolution from its definition, frame-major: out[t, co].
Vector<float> convolutionReference(const Vector<float>& input,
                                   const Vector<float>& weight,
                                   const Vector<float>& bias,
                                   const UnfoldShape& shape,
                                   int outputChannelCount)
{
    auto outputLength = shape.outputLength();
    auto expected = sized(outputLength * outputChannelCount);

    for (auto frame = 0; frame < outputLength; ++frame)
        for (auto channel = 0; channel < outputChannelCount; ++channel)
        {
            auto total = (double) bias[channel];

            for (auto source = 0; source < shape.inputChannelCount; ++source)
                for (auto tap = 0; tap < shape.kernelSize; ++tap)
                {
                    auto at = frame * shape.stride + tap - shape.padding;

                    if (at < 0 || at >= shape.inputLength)
                        continue;

                    auto weightAt = (channel * shape.inputChannelCount + source)
                                        * shape.kernelSize
                                    + tap;

                    total += (double) weight[weightAt]
                             * input[source * shape.inputChannelStride
                                     + at * shape.inputFrameStride];
                }

            expected[frame * outputChannelCount + channel] = (float) total;
        }

    return expected;
}

// Whisper's two convolutions in miniature: kernel 3, padding 1, stride one
// over a channel-major input and stride two over a frame-major one.
constexpr auto channelMajor = UnfoldShape {5, 23, 3, 1, 1, 23, 1};
constexpr auto frameMajor = UnfoldShape {7, 23, 3, 2, 1, 1, 7};
} // namespace

auto tUnfoldMatchesDefinition = test("Kernels/unfoldMatchesDefinition") = []
{
    if (!Device::shared().isValid())
        return;

    for (const auto& shape: {channelMajor, frameMajor})
    {
        auto input =
            spreadValues(shape.inputChannelCount * shape.inputLength, 910u, 2.f);
        auto result = runUnfold(input, shape);
        auto expected = unfoldReference(input, shape);

        check(result.size() == expected.size());

        for (auto i = 0; i < result.size(); ++i)
            check(result[i] == expected[i]);
    }
};

// The two together are the convolution: unfolded rows against the weight in
// its file layout, through the tiled product, land on PyTorch's definition
// frame-major — which is the layout the transformer reads, so the permute
// after the second convolution is the store's.
auto tUnfoldAndTiledProductAreTheConvolution =
    test("Kernels/unfoldAndTiledProductAreTheConvolution") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto outputChannels = 6;

    for (const auto& shape: {channelMajor, frameMajor})
    {
        auto input =
            spreadValues(shape.inputChannelCount * shape.inputLength, 920u, 2.f);
        auto weight = spreadValues(outputChannels * shape.columnCount(), 921u, 1.f);
        auto bias = spreadValues(outputChannels, 922u, 1.f);

        auto columns = storageOf(runUnfold(input, shape));
        auto weightBuffer = storageOf(weight);
        auto biasBuffer = storageOf(bias);
        auto output = outputFor(shape.outputLength() * outputChannels);

        auto product = TiledLinear {};
        product.a = columns;
        product.b = weightBuffer;
        product.bias = biasBuffer;
        product.output = output;
        product.prepare();

        auto commands = Device::shared().makeCommandBuffer();

        {
            auto pass = commands.beginCompute();
            product.dispatch(pass,
                             TiledMatMulShape::forLinear(shape.outputLength(),
                                                         shape.columnCount(),
                                                         outputChannels));
        }

        commands.commit();

        auto result = readBack(output, shape.outputLength() * outputChannels);
        auto expected =
            convolutionReference(input, weight, bias, shape, outputChannels);

        for (auto i = 0; i < result.size(); ++i)
            check(isClose(result[i], expected[i], 1e-5));
    }
};
