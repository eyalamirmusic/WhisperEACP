#pragma once

#include "KernelTypes.h"

namespace WSP
{
// output[i] = a[i] + b[i], one thread per element: dispatch(kernel,
// elementCount). No shape and no stride — the residual connection around every
// attention and every feed-forward, and the positional embedding added to the
// encoder's input, are all the same elementwise sum over two buffers of equal
// length, and none of them cares how those elements are laid out in rows.
struct Add final : ComputeProgram
{
    Add() { compile(); }

    void define() override
    {
        auto at = threadId();
        write(output, at, a[at] + b[at]);
    }

    Uniform<InputBuffer> a;
    Uniform<InputBuffer> b;
    Uniform<OutputBuffer> output;

    EACP_SHADER(a, b, output)
};
} // namespace WSP
