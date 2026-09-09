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
// buffer becomes its probability buffer rather than being copied.
//
// The encoder does not run this. Its score matrix is 54 MB a layer, and three
// reads and two writes of that is more than the product either side of it
// costs, so the scores product reports each row's maxima and the apply
// normalises what it stages — TiledMatMul.h's AFold and RowMaxima. What is
// left here is the decoder's prompt step, whose rows are few.
//
// Unlike LayerNorm's, one lane count serves both halves here, because a row
// this long has work for every lane whichever half is asking: 256 measured
// best over 9000 rows of 1500 (207 us at the stock 64, 176 at 256) and best
// again over the twelve rows the decoder's prompt step normalises (9.6 us to
// 5.1). 512 loses at both.
struct Softmax final : ReducingProgram
{
    static constexpr auto preferredLanes = 256;

    explicit Softmax(int laneCount = preferredLanes)
        : ReducingProgram(laneCount)
    {
        compile();
    }

    void define() override
    {
        auto lane = localId();
        auto base = groupId() * rowLength;

        auto largest = var(std::numeric_limits<float>::lowest());
        auto scanning = var(lane);

        loop(scanning.get() < rowLength,
             [&]
             {
                 largest = max(largest.get(), values[base + scanning.get()]);
                 scanning += lanes;
             });

        auto rowMaximum = var(groupMax(largest.get()));

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

        auto normaliser = var(1.f / groupSum(total.get()));
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
