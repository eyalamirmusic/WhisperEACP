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
// shift. Each exponential is evaluated once and parked in output as it is
// summed, then read back and scaled in place — a thread owns its whole row, so
// every read-after-write here is of its own store.
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
                 auto at = base + summing;
                 auto weight = exp(input[at] - rowMaximum);

                 write(output, at, weight);
                 total += weight;
                 summing += 1u;
             });

        auto normaliser = 1.f / total.get();
        auto scaling = var(0u);

        loop(scaling < rowLength,
             [&]
             {
                 auto at = base + scaling;
                 write(output, at, output[at] * normaliser);
                 scaling += 1u;
             });
    }

    Uniform<InputBuffer> input;
    Uniform<OutputBuffer> output;
    Uniform<UInt> rowLength;

    EACP_SHADER(input, output, rowLength)
};
} // namespace WSP
