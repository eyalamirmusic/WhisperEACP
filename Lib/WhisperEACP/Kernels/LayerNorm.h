#pragma once

#include "Reduce.h"

namespace WSP
{
// y = (x - mean) / sqrt(variance + epsilon) * weight + bias, over rows of
// rowLength elements. weight and bias are one row long and shared by every row.
//
// One group per row: dispatchRows(pass, rowCount). The mean and the variance
// are each a strided walk and a group reduction, the variance from the centred
// values so the arithmetic is the reference's rather than a rearrangement of
// it. The row length is a uniform rather than a constant because this is what
// the encoder and decoder get written out of — a layernorm that only knows
// tiny.en's width is one that gets rewritten for base.
//
// The lane count is the caller's, because the two halves dispatch this at
// opposite shapes and they measure to opposite answers.
struct LayerNorm final : ReducingProgram
{
    static constexpr auto whisperEpsilon = 1e-5f;

    // A decode step's thirteen dispatches are one row of 384 each, with
    // nothing else on the machine, so what is wanted is the widest group that
    // still has work for every lane: 192 over a 384-wide row is two elements a
    // lane, and 4.6 us of hand tree measured down to 3.1.
    static constexpr auto singleRowLanes = 192;

    // The encoder's nine are 1500 rows, which fill the machine on their own,
    // and there a wider group only adds barriers: 128 lanes measured 1.4x
    // slower than the stock 64 and 256 lanes 2.4x. (32 came out 3% under 64,
    // consistently but on 0.8% of an encode, which is not worth holding a
    // group to half a wave on a GPU whose waves are 64 wide.)
    static constexpr auto manyRowLanes = groupWidth;

    explicit LayerNorm(int laneCount = manyRowLanes)
        : ReducingProgram(laneCount)
    {
        epsilon = whisperEpsilon;
        compile();
    }

    void define() override
    {
        auto lane = localId();
        auto base = groupId() * rowLength;
        auto width = toFloat(rowLength);

        auto total = var(0.f);
        auto summing = var(lane);

        loop(summing.get() < rowLength,
             [&]
             {
                 total += input[base + summing.get()];
                 summing += lanes;
             });

        auto mean = var(groupSum(total.get()) / width);

        auto squares = var(0.f);
        auto spreading = var(lane);

        loop(spreading.get() < rowLength,
             [&]
             {
                 auto centred = input[base + spreading.get()] - mean.get();
                 squares += centred * centred;
                 spreading += lanes;
             });

        auto scale = var(rsqrt(groupSum(squares.get()) / width + epsilon));
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
