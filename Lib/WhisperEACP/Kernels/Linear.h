#pragma once

#include "Gelu.h"
#include "KernelTypes.h"
#include "MatMul.h"

#include <algorithm>

namespace WSP
{
// y = x W^T + b, which is every projection in a transformer, with W stored the
// way PyTorch's nn.Linear stores it: [outputWidth, innerCount] row-major, one
// output's whole input row contiguous.
//
//   output[row, o] = bias[o]
//                  + sum over k of input[row * innerCount + k]
//                                 * weight[o * innerCount + k]
//
// That is the shape model.encoder.layers.0.self_attn.q_proj.weight already has
// in the safetensors file, so the loader hands the bytes over untransposed and
// nothing in the model transposes a weight. MatMul next door is the same
// product with the operand stored the other way round, which is the shape a
// matrix multiplication is written in rather than the shape a weight is shipped
// in; both exist because both occur.
//
// One thread per output element over a 2D grid, accumulating serially:
// dispatch(kernel, outputWidth, rowCount). The row count is the dispatch height
// and never reaches the kernel body, which is why it is not a uniform; the
// other two are, because they are the strides the input, the weight and the
// output are walked at.
//
// The bias buffer is always bound, for MatMul's reason: a projection that has
// none — Whisper's attention k_proj — binds a zero buffer of outputWidth
// floats, since neither backend defines what a shader reading a buffer nothing
// was bound to gets.
//
// The weight is the one operand that can be fp16, and for MatMul's reason
// again: a repo ships its weights packed, while the input, the bias and the
// output are activations this stack computes in float32 throughout.
template <WeightStorage weightStorage>
struct LinearProgram final : ComputeProgram
{
    LinearProgram() { compile(); }

    void define() override
    {
        auto position = threadPosition();
        auto column = position.x;
        auto row = position.y;

        auto inputBase = row * innerCount;
        auto weightBase = column * innerCount;

        auto total = var(0.f);
        auto step = var(0u);

        loop(step < innerCount,
             [&]
             {
                 total += input[inputBase + step] * weight(weightBase + step);
                 step += 1u;
             });

        write(output, row * outputWidth + column, total.get() + bias[column]);
    }

    // Exactly representable either way round, so the packed form is the same
    // number the widened one is rather than the same number to a tolerance.
    Float weight(const UInt& index)
    {
        if constexpr (weightStorage == WeightStorage::PackedHalf)
            return weights.readHalf(index);
        else
            return weights[index];
    }

    Uniform<InputBuffer> input;
    Uniform<InputBuffer> weights;
    Uniform<InputBuffer> bias;
    Uniform<OutputBuffer> output;
    Uniform<UInt> innerCount;
    Uniform<UInt> outputWidth;

    EACP_SHADER(input, weights, bias, output, innerCount, outputWidth)
};

using Linear = LinearProgram<WeightStorage::Float>;
using HalfWeightLinear = LinearProgram<WeightStorage::PackedHalf>;

// The same product for a handful of rows — a decode step's one token, or the
// prompt's two — where a thread per output is a thread per 384 or 1536 serial
// multiply-adds and the GPU is nearly idle. The inner sum is split across
// splitCount lanes instead: the grid is [splitCount, rowCount * outputWidth],
// each lane of a group row takes every splitCount'th run of four inputs of one
// output, and the group folds the partial sums before the one store. Adjacent
// lanes read adjacent words of the same weight row, which is the access the
// memory system serves whole.
//
// **The split count is the shape's, and it is what makes this kernel fast.**
// It sets both how much of the inner sum a lane walks and how many groups the
// dispatch has, and the second is what a decode step is short of: a 384-wide
// projection of one row is 590 kB of weight read by 48 groups at the stock 8,
// which is a few thousand threads on a device sized for tens of thousands.
// The caller names the count its shapes want — Decoder::stepSplitCount and
// logitsSplitCount are the two this model uses, both measured — the way
// ReducingProgram takes its lane count.
//
// A group is splitCount lanes by however many outputs 64 threads then hold,
// and one output at or past 64. At one output the fold is groupSum(), which is
// a SIMD reduction rather than the shared array and the serial walk of it a
// group holding several outputs still needs.
//
// dispatch(pass, outputWidth, rowCount), which rounds the height up to whole
// groups since the fold is a barrier; a thread past the last output computes
// against the last row and stores nothing. The runs of four are read4 when
// innerCount divides by four and single elements otherwise, so the kernel is
// correct at every shape and fast at the model's.
//
// Two flags fold the stages either side of a projection into its store, each
// a dispatch of its own otherwise and a few microseconds of GPU whatever its
// size: gelu applies the activation to the result, which is what follows fc1,
// and residual adds the result to what the output already holds, which is
// what follows every out_proj and fc2 — in place on the residual stream, the
// element read and stored by the one lane that stores it. That one lane
// matters: the residual is a read-modify-write, so every path here stores from
// a single lane of the group rather than from all of them holding the same
// folded value.
template <WeightStorage weightStorage>
struct SplitLinearProgram final : ComputeProgram
{
    explicit SplitLinearProgram(int splitCount = groupSize2D)
        : ComputeProgram({splitCount, std::max(1, groupWidth / splitCount)})
        , splits((unsigned) groupShape().x)
        , outputsPerGroup(groupShape().y)
    {
        compile();
    }

    void dispatch(ComputePass& pass, int outputWidth, int rowCount)
    {
        const auto outputs = outputWidth * rowCount;
        const auto height =
            (outputs + outputsPerGroup - 1) / outputsPerGroup * outputsPerGroup;

        pass.dispatch(*this, (int) splits, height);
    }

    void define() override
    {
        auto position = threadPosition();
        auto split = position.x;
        auto element = position.y;
        auto row = element / outputWidth;
        auto column = element % outputWidth;

        auto inputBase = min(row, rowCount - 1u) * innerCount;
        auto weightBase = column * innerCount;

        auto total = var(0.f);

        ifThen(
            innerCount % 4u == 0u,
            [&]
            {
                auto step = var(split * 4u);

                loop(step.get() < innerCount,
                     [&]
                     {
                         total += dot(input.read4((inputBase + step.get()) / 4u),
                                      weight4(weightBase + step.get()));
                         step += splits * 4u;
                     });
            },
            [&]
            {
                auto step = var(split);

                loop(step.get() < innerCount,
                     [&]
                     {
                         total += input[inputBase + step.get()]
                                  * weight(weightBase + step.get());
                         step += splits;
                     });
            });

        auto local = localPosition();

        if (outputsPerGroup == 1)
        {
            auto sum = var(groupSum(total.get()));

            ifThen(local.x == 0u, [&] { store(sum.get(), element, column, row); });
            return;
        }

        auto tile = shared<Float>((int) splits * outputsPerGroup);

        write(tile, local.y * splits + local.x, total.get());
        barrier();

        auto sum = var(0.f);

        for (auto part = 0u; part < splits; ++part)
            sum += tile[local.y * splits + part];

        ifThen(local.x == 0u, [&] { store(sum.get(), element, column, row); });
    }

    // The one store, behind the row test a rounded-up dispatch height needs:
    // the bias, then the two stages folded into it.
    void store(const Float& sum,
               const UInt& element,
               const UInt& column,
               const UInt& row)
    {
        ifThen(row < rowCount,
               [&]
               {
                   auto value = sum + bias[column];
                   auto activated = select(gelu != 0u, exactGelu(value), value);
                   auto carried = select(residual != 0u, output[element], 0.f);

                   write(output, element, activated + carried);
               });
    }

    Float weight(const UInt& index)
    {
        if constexpr (weightStorage == WeightStorage::PackedHalf)
            return weights.readHalf(index);
        else
            return weights[index];
    }

    // Four consecutive weights from a four-aligned index: one record of the
    // float buffer, or the two words that hold four halves.
    Float4 weight4(const UInt& index)
    {
        if constexpr (weightStorage == WeightStorage::PackedHalf)
            return float4(weights.readHalf2(index / 2u),
                          weights.readHalf2(index / 2u + 1u));
        else
            return weights.read4(index / 4u);
    }

    Uniform<InputBuffer> input;
    Uniform<InputBuffer> weights;
    Uniform<InputBuffer> bias;
    Uniform<OutputBuffer> output;
    Uniform<UInt> innerCount;
    Uniform<UInt> outputWidth;
    Uniform<UInt> rowCount;
    Uniform<UInt> gelu;
    Uniform<UInt> residual;

    // How many lanes share one output's inner sum, and how many outputs one
    // group therefore holds: the group is [splits, groupWidth / splits] below
    // the group width, so it stays 64 threads, and [splits, 1] at or above it.
    const unsigned splits;
    const int outputsPerGroup;

    EACP_SHADER(input,
                weights,
                bias,
                output,
                innerCount,
                outputWidth,
                rowCount,
                gelu,
                residual)
};

using SplitLinear = SplitLinearProgram<WeightStorage::Float>;
using HalfWeightSplitLinear = SplitLinearProgram<WeightStorage::PackedHalf>;
} // namespace WSP
