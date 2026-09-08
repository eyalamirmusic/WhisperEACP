#pragma once

#include "KernelTypes.h"

namespace WSP
{
// output[i] += addend[i], one thread per element: dispatch(kernel,
// elementCount). No shape and no stride — the residual connection around every
// attention and every feed-forward, and the positional embedding added to the
// encoder's input, are all the same elementwise sum into a stream that goes on
// being the stream, and none of them cares how the elements are laid out in
// rows.
//
// In place on eacp's terms for a 1:1 stage: the element is read and stored by
// the one thread that owns it, so the residual stream is one buffer added to
// rather than three taking turns.
struct Add final : ComputeProgram
{
    Add() { compile(); }

    void define() override
    {
        auto at = threadId();
        write(output, at, output[at] + addend[at]);
    }

    Uniform<InputBuffer> addend;
    Uniform<OutputBuffer> output;

    EACP_SHADER(addend, output)
};
} // namespace WSP
