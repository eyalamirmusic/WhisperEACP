#pragma once

#include "KernelTypes.h"

namespace WSP
{
// The base of every kernel here that gives a whole group of threads to one row
// rather than one thread: each thread walks a strided share of the row, parks
// what it holds in a shared tile, and the tile is folded to element zero by
// halving spans with a barrier between each. A row of 1500 keys or 51864
// logits walked by a single thread was the whole cost of a decode step.
//
// A group is the unit, so dispatchRows() dispatches rowCount groups and lanes
// is the stride a thread walks its row at. The group shape is eacp's fixed one
// (ComputeProgram::groupWidth), which is why it is a constant and not a
// uniform.
//
// A barrier removes the generated bounds guard, so a kernel built on this is
// always dispatched over whole groups and bounds itself: every walk stops at
// the row length and every store is behind a lane test or a row test.
struct ReducingProgram : ComputeProgram
{
    static constexpr auto lanes = (unsigned) groupWidth;

    void dispatchRows(ComputePass& pass, int rowCount)
    {
        pass.dispatch(*this, rowCount * groupWidth);
    }

protected:
    template <typename Combine>
    void fold(const Shared<Float>& tile, const UInt& lane, Combine&& combine)
    {
        for (auto span = lanes / 2u; span > 0u; span /= 2u)
        {
            ifThen(lane < span,
                   [&]
                   { write(tile, lane, combine(tile[lane], tile[lane + span])); });

            barrier();
        }
    }

    void foldSum(const Shared<Float>& tile, const UInt& lane)
    {
        fold(tile, lane, [](const Float& a, const Float& b) { return a + b; });
    }

    void foldMax(const Shared<Float>& tile, const UInt& lane)
    {
        fold(tile, lane, [](const Float& a, const Float& b) { return max(a, b); });
    }
};
} // namespace WSP
