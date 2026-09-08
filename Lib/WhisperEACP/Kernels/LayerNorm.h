#pragma once

#include "Reduce.h"

namespace WSP
{
// y = (x - mean) / sqrt(variance + epsilon) * weight + bias, over rows of
// rowLength elements. weight and bias are one row long and shared by every row.
//
// One group per row: dispatchRows(pass, rowCount). The mean and the variance
// are each a strided walk and a fold, the variance from the centred values so
// the arithmetic is the reference's rather than a rearrangement of it. The row
// length is a uniform rather than a constant because this is what the encoder
// and decoder get written out of — a layernorm that only knows tiny.en's width
// is one that gets rewritten for base.
struct LayerNorm final : ReducingProgram
{
    static constexpr auto whisperEpsilon = 1e-5f;

    LayerNorm()
    {
        epsilon = whisperEpsilon;
        compile();
    }

    void define() override
    {
        auto lane = localId();
        auto base = groupId() * rowLength;
        auto width = toFloat(rowLength);
        auto tile = shared<Float>(groupWidth);

        auto total = var(0.f);
        auto summing = var(lane);

        loop(summing.get() < rowLength,
             [&]
             {
                 total += input[base + summing.get()];
                 summing += lanes;
             });

        write(tile, lane, total.get());
        barrier();
        foldSum(tile, lane);

        auto mean = var(tile[0u] / width);
        barrier();

        auto squares = var(0.f);
        auto spreading = var(lane);

        loop(spreading.get() < rowLength,
             [&]
             {
                 auto centred = input[base + spreading.get()] - mean.get();
                 squares += centred * centred;
                 spreading += lanes;
             });

        write(tile, lane, squares.get());
        barrier();
        foldSum(tile, lane);

        auto scale = var(rsqrt(tile[0u] / width + epsilon));
        auto writing = var(lane);

        loop(writing.get() < rowLength,
             [&]
             {
                 auto at = base + writing.get();
                 auto column = writing.get();

                 write(output,
                       at,
                       (input[at] - mean.get()) * scale.get() * weight[column]
                           + bias[column]);

                 writing += lanes;
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
