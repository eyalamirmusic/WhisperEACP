#pragma once

#include "KernelTypes.h"

namespace WSP
{
// PyTorch's nn.Unfold over one dimension: every window a convolution slides
// its kernel across, gathered into a row, so that the convolution itself is
// one product of those rows against the weight in the [out, in, k] layout the
// file ships it in — Conv1d next door as a tiled matrix multiply rather than
// a thread per output element walking every tap.
//
//   columns[t, ci * kernelSize + k] = input[ci, t * stride + k - padding]
//
// reading zero outside [0, inputLength), over the outputLength frames Conv1d's
// rule gives. The input is addressed through a channel stride and a frame
// stride, so it is the mel's band-major [C, T] and the first convolution's
// frame-major [T, C] output alike; the columns come out [outputLength,
// inputChannelCount * kernelSize] row-major, which is what the tiled product
// takes as A with the weight as B.
//
// One thread per (column, frame) over a 2D grid: dispatch(kernel,
// inputChannelCount * kernelSize, outputLength). The tap index stays
// unsigned, for Conv1d's reason: the window is tested against the padded
// range rather than shifted below zero, and a tap outside it reads element
// zero and stores nothing of it.
struct Unfold final : ComputeProgram
{
    Unfold() { compile(); }

    void define() override
    {
        auto position = threadPosition();
        auto column = position.x;
        auto frame = position.y;
        auto channel = column / kernelSize;
        auto tap = column % kernelSize;

        auto source = frame * stride + tap;
        auto inside = source >= padding && source < inputLength + padding;
        auto offset = select(inside, source - padding, 0u);
        auto at = channel * inputChannelStride + offset * inputFrameStride;

        write(columns,
              frame * (inputChannelCount * kernelSize) + column,
              select(inside, input[at], 0.f));
    }

    Uniform<InputBuffer> input;
    Uniform<OutputBuffer> columns;
    Uniform<UInt> inputChannelCount;
    Uniform<UInt> inputLength;
    Uniform<UInt> kernelSize;
    Uniform<UInt> stride;
    Uniform<UInt> padding;
    Uniform<UInt> inputChannelStride;
    Uniform<UInt> inputFrameStride;

    EACP_SHADER(input,
                columns,
                inputChannelCount,
                inputLength,
                kernelSize,
                stride,
                padding,
                inputChannelStride,
                inputFrameStride)
};
} // namespace WSP
