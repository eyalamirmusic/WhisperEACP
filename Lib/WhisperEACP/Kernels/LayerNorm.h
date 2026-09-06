#pragma once

#include "KernelTypes.h"

namespace WSP
{
// y = (x - mean) / sqrt(variance + epsilon) * weight + bias, over rows of
// rowLength elements. weight and bias are one row long and shared by every row.
//
// One thread per row, reducing serially: dispatch(kernel, rowCount). The row
// length is a uniform rather than a constant because this is what the encoder
// and decoder get written out of — a layernorm that only knows tiny.en's width
// is one that gets rewritten for base.
struct LayerNorm final : ComputeProgram
{
    static constexpr auto whisperEpsilon = 1e-5f;

    LayerNorm()
    {
        epsilon = whisperEpsilon;
        compile();
    }

    void define() override
    {
        auto base = threadId() * rowLength;
        auto width = toFloat(rowLength);

        auto total = var(0.f);
        auto summing = var(0u);

        loop(summing < rowLength,
             [&]
             {
                 total += input[base + summing];
                 summing += 1u;
             });

        auto mean = total.get() / width;

        auto squares = var(0.f);
        auto spreading = var(0u);

        loop(spreading < rowLength,
             [&]
             {
                 auto centred = input[base + spreading] - mean;
                 squares += centred * centred;
                 spreading += 1u;
             });

        auto scale = rsqrt(squares.get() / width + epsilon);
        auto writing = var(0u);

        loop(writing < rowLength,
             [&]
             {
                 auto at = base + writing;

                 write(output,
                       at,
                       (input[at] - mean) * scale * weight[writing] + bias[writing]);

                 writing += 1u;
             });
    }

    Uniform<InputBuffer> input;
    Uniform<InputBuffer> weight;
    Uniform<InputBuffer> bias;
    Uniform<OutputBuffer> output;
    Uniform<UInt> rowLength;
    Uniform<Float> epsilon;

    EACP_SHADER(input, weight, bias, output, rowLength, epsilon)
};
} // namespace WSP
