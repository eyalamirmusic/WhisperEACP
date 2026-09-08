#pragma once

#include "Reduce.h"

#include <limits>

namespace WSP
{
// Row-wise softmax over rows of rowLength elements, in place, one group per
// row: dispatchRows(pass, rowCount).
//
// The row maximum is subtracted before the exponential, which is what keeps an
// attention row of large logits finite: exp overflows to infinity somewhere
// past 88 in float32, and the ratio it is heading for is unchanged by the
// shift. Each exponential is evaluated once and parked in the row as it is
// summed, then read back and scaled — and every element is read back by the
// lane that stored it, since a lane walks the same strided share of the row in
// all three passes, which is the read-after-write eacp promises for an output.
//
// The exponential is held in a var before it is stored. A buffer read in eacp
// names the element rather than the value — a read handle re-materialises
// after a store to its slot, which is what makes the in-place scaling pass
// read what was stored — so an expression built on the read and used after
// the store would be recomputed from the stored value: exp of the exp.
//
// In place because the row is never needed unnormalised: an attention's score
// buffer becomes its probability buffer, and the encoder's 54 MB of scores are
// written once and read twice rather than copied.
struct Softmax final : ReducingProgram
{
    Softmax() { compile(); }

    void define() override
    {
        auto lane = localId();
        auto base = groupId() * rowLength;
        auto tile = shared<Float>(groupWidth);

        auto largest = var(std::numeric_limits<float>::lowest());
        auto scanning = var(lane);

        loop(scanning.get() < rowLength,
             [&]
             {
                 largest = max(largest.get(), values[base + scanning.get()]);
                 scanning += lanes;
             });

        write(tile, lane, largest.get());
        barrier();
        foldMax(tile, lane);

        auto rowMaximum = var(tile[0u]);
        barrier();

        auto total = var(0.f);
        auto summing = var(lane);

        loop(summing.get() < rowLength,
             [&]
             {
                 auto at = base + summing.get();
                 auto weight = var(exp(values[at] - rowMaximum.get()));

                 write(values, at, weight.get());
                 total += weight.get();
                 summing += lanes;
             });

        write(tile, lane, total.get());
        barrier();
        foldSum(tile, lane);

        auto normaliser = var(1.f / tile[0u]);
        auto scaling = var(lane);

        loop(scaling.get() < rowLength,
             [&]
             {
                 auto at = base + scaling.get();
                 write(values, at, values[at] * normaliser.get());
                 scaling += lanes;
             });
    }

    Uniform<OutputBuffer> values;
    Uniform<UInt> rowLength;

    EACP_SHADER(values, rowLength)
};
} // namespace WSP
