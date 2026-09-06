#pragma once

#include "KernelTypes.h"

namespace WSP
{
// PyTorch's nn.Conv1d, which is what Whisper's encoder opens with:
//
//   out[co, t] = bias[co]
//              + sum over ci, k of w[co, ci, k] * in[ci, t * stride + k - padding]
//
// reading zero outside [0, inputLength). The output is
// (inputLength + 2 * padding - kernelSize) / stride + 1 frames long, which over
// Whisper's 3000 mel frames is 3000 after conv1 (kernel 3, stride 1, padding 1)
// and 1500 after conv2 (kernel 3, stride 2, padding 1).
//
// The layouts, all row-major:
//
//   input   channel-major, [inputChannelCount, inputLength], read at
//           ci * inputLength + t. That is what Mel/ writes — 80 bands x 3000
//           frames, band-major — and it is PyTorch's own [C, T].
//   weight  [outputChannelCount, inputChannelCount, kernelSize], read at
//           (co * inputChannelCount + ci) * kernelSize + k. That is the shape
//           model.encoder.conv1.weight already has in the safetensors file
//           (384 x 80 x 3 for tiny.en), so the loader hands the bytes over
//           untransposed.
//   bias    one value per output channel.
//   output  co * outputChannelStride + t * outputFrameStride. The pair of
//           strides is how one kernel writes either layout without a branch:
//           (outputLength, 1) is channel-major, which is what the next
//           convolution reads, and (1, outputChannelCount) is frame-major, the
//           [T, C] rows the transformer wants — HuggingFace's permute after
//           conv2, done here by the store rather than by a pass of its own.
//
// One thread per output element over a 2D grid, accumulating serially:
// dispatch(kernel, outputLength, outputChannelCount). Neither extent is a
// uniform, since the generated bounds guard already has both and the strides
// the kernel walks its buffers at are given outright.
//
// The tap index stays unsigned. t * stride + k counts up from zero, so testing
// the tap against the shifted window — inside when it lands in
// [padding, inputLength + padding) — reaches the same taps as subtracting the
// padding first would, without ever forming the negative index there is no
// unsigned way back from.
//
// GELU follows each convolution, applied by the caller: the elementwise kernels
// stay separate.
struct Conv1d final : ComputeProgram
{
    Conv1d() { compile(); }

    void define() override
    {
        auto position = threadPosition();
        auto frame = position.x;
        auto channel = position.y;

        auto total = var(0.f);
        auto tap = var(0u);

        loop(tap < kernelSize,
             [&]
             {
                 auto source = frame * stride + tap;

                 ifThen(source >= padding && source < inputLength + padding,
                        [&]
                        {
                            auto at = source - padding;
                            auto taps =
                                channel * inputChannelCount * kernelSize + tap;
                            auto sourceChannel = var(0u);

                            loop(sourceChannel < inputChannelCount,
                                 [&]
                                 {
                                     total +=
                                         weight[taps + sourceChannel * kernelSize]
                                         * input[sourceChannel * inputLength + at];

                                     sourceChannel += 1u;
                                 });
                        });

                 tap += 1u;
             });

        write(output,
              channel * outputChannelStride + frame * outputFrameStride,
              total.get() + bias[channel]);
    }

    Uniform<InputBuffer> input;
    Uniform<InputBuffer> weight;
    Uniform<InputBuffer> bias;
    Uniform<OutputBuffer> output;
    Uniform<UInt> inputChannelCount;
    Uniform<UInt> inputLength;
    Uniform<UInt> kernelSize;
    Uniform<UInt> stride;
    Uniform<UInt> padding;
    Uniform<UInt> outputChannelStride;
    Uniform<UInt> outputFrameStride;

    EACP_SHADER(input,
                weight,
                bias,
                output,
                inputChannelCount,
                inputLength,
                kernelSize,
                stride,
                padding,
                outputChannelStride,
                outputFrameStride)
};

// The output length rule, spelled once so a caller sizing a buffer and a test
// checking one cannot disagree about it.
constexpr int
    conv1dOutputLength(int inputLength, int kernelSize, int stride, int padding)
{
    return (inputLength + 2 * padding - kernelSize) / stride + 1;
}
} // namespace WSP
