#pragma once

#include "KernelTypes.h"

namespace WSP
{
// Row-wise softmax over rows of rowLength elements, one thread per row:
// dispatch(kernel, rowCount).
//
// The row maximum is subtracted before the exponential, which is what keeps an
// attention row of large logits finite: exp overflows to infinity somewhere
// past 88 in float32, and the ratio it is heading for is unchanged by the
// shift. The exponential is evaluated twice per element rather than kept in a
// buffer between the two passes — an OutputBuffer is write-only, and a scratch
// allocation the size of the attention matrix costs more than the second exp.
struct Softmax final : ComputeProgram
{
    Softmax() { compile(); }

    void define() override
    {
        auto base = threadId() * rowLength;

        auto largest = var(input[base]);
        auto scanning = var(1u);

        loop(scanning < rowLength,
             [&]
             {
                 largest = max(largest.get(), input[base + scanning]);
                 scanning += 1u;
             });

        auto rowMaximum = largest.get();

        auto total = var(0.f);
        auto summing = var(0u);

        loop(summing < rowLength,
             [&]
             {
                 total += exp(input[base + summing] - rowMaximum);
                 summing += 1u;
             });

        auto normaliser = 1.f / total.get();
        auto writing = var(0u);

        loop(writing < rowLength,
             [&]
             {
                 auto at = base + writing;
                 write(output, at, exp(input[at] - rowMaximum) * normaliser);
                 writing += 1u;
             });
    }

    Uniform<InputBuffer> input;
    Uniform<OutputBuffer> output;
    Uniform<UInt> rowLength;

    EACP_SHADER(input, output, rowLength)
};
} // namespace WSP
