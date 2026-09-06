#pragma once

#include "KernelTypes.h"
#include "MatMul.h"

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
} // namespace WSP
