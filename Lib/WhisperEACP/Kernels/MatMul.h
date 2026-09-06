#pragma once

#include "KernelTypes.h"

namespace WSP
{
// C = A * B + bias, all row-major: A is rowCount x innerCount, B is
// innerCount x columnCount, C is rowCount x columnCount, and bias is one value
// per output column, broadcast down every row — the shape a linear layer's
// bias has.
//
// One thread per output element over a 2D grid, accumulating serially:
// dispatch(kernel, columnCount, rowCount). The row count is the dispatch
// height and never reaches the kernel body, which is why it is not a uniform;
// the other two are, because they are the strides A, B and C are walked at.
//
// The bias buffer is always bound. A projection that has none — Whisper's
// attention k_proj, for one — binds a zero buffer of columnCount floats, which
// is the cheapest way to say it: neither backend defines what a shader reading
// a buffer nothing was bound to gets, so there is no unbound slot to branch
// around and a flag would only guard a read that must not happen at all.
struct MatMul final : ComputeProgram
{
    MatMul() { compile(); }

    void define() override
    {
        auto position = threadPosition();
        auto column = position.x;
        auto row = position.y;

        auto total = var(0.f);
        auto step = var(0u);

        loop(step < innerCount,
             [&]
             {
                 total +=
                     a[row * innerCount + step] * b[step * columnCount + column];
                 step += 1u;
             });

        write(output, row * columnCount + column, total.get() + bias[column]);
    }

    Uniform<InputBuffer> a;
    Uniform<InputBuffer> b;
    Uniform<InputBuffer> bias;
    Uniform<OutputBuffer> output;
    Uniform<UInt> innerCount;
    Uniform<UInt> columnCount;

    EACP_SHADER(a, b, bias, output, innerCount, columnCount)
};
} // namespace WSP
