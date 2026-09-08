#pragma once

#include "KernelTypes.h"

namespace WSP
{
// The base of every kernel here that gives a whole group of threads to one row
// rather than one thread: each thread walks a strided share of the row, and
// the group folds what the threads are left holding through groupSum() or
// groupMax() — every thread contributes one value and every thread gets the
// whole group's fold back. A row of 1500 keys or 51864 logits walked by a
// single thread was the whole cost of a decode step.
//
// A group is the unit, so dispatchRows() dispatches rowCount groups and lanes
// is the stride a thread walks its row at. The group is this program's own
// rather than the one width every kernel shares: the constructor names it and
// lanes reads it back from groupShape(), so a kernel dispatched over one row
// can take the wider group that hides its latency while a kernel dispatched
// over 1500 keeps the stock 64, which already saturates memory bandwidth.
//
// A group reduction is a barrier, and a barrier removes the generated bounds
// guard, so a kernel built on this is always dispatched over whole groups and
// bounds itself: every walk stops at the row length, every store is behind a
// lane test or a row test, and every reduction is reached by every thread of
// the group.
struct ReducingProgram : ComputeProgram
{
    explicit ReducingProgram(int laneCount = groupWidth)
        : ComputeProgram({laneCount})
        , lanes((unsigned) groupShape().x)
    {
    }

    void dispatchRows(ComputePass& pass, int rowCount)
    {
        pass.dispatch(*this, rowCount * (int) lanes);
    }

    // The threads one group holds, which is both what a strided walk steps by
    // and how many rows' worth of threads a dispatch asks for.
    const unsigned lanes;
};
} // namespace WSP
