#pragma once

#include "KernelTypes.h"

namespace WSP
{
// What greedy sampling ends a decoder step with: the index of the largest
// logit in a row, with the tokens the caller suppressed taken out of the
// running before the comparison rather than after it.
//
//   indices[row] = the first j maximising logits[row * rowLength + j]
//                  over the j with mask[j] == 0
//
// The layouts:
//
//   logits   [rowCount, rowLength]   row-major
//   mask     [rowLength]             nonzero at a suppressed token
//   indices  [rowCount]              unsigned integers
//
// One thread per row, reducing serially: dispatch(kernel, rowCount). The row
// count is the dispatch count and never reaches the body; rowLength is a
// uniform, since it is the stride the logits are walked at and the bound the
// scan runs to.
//
// Lowest index on ties, which is what torch.argmax documents and what the
// reference implementations therefore produce. A strict > against the best so
// far is the whole of it: an equal value never displaces the earlier index.
//
// The mask is one buffer and the caller decides which one. Whisper ships two
// lists — suppress_tokens, which applies at every step, and
// begin_suppress_tokens, which applies only at the first — and folding them
// into a rule here would put the generation config inside a kernel. Binding
// the union at the first step and suppress_tokens after it keeps that where it
// belongs, and costs a buffer swap between dispatches.
//
// A row whose every token is suppressed has no answer, and yields index zero.
// That case does not arise from Whisper's lists, which cover a few hundred of
// a fifty-thousand-token vocabulary; it is stated so the value is documented
// rather than incidental.
//
// The output is an AtomicBuffer because that is the only buffer kind eacp's
// write() takes a UInt into: its elements are unsigned on both backends
// (`device atomic_uint*` and RWStructuredBuffer<uint>), where an OutputBuffer's
// are floats. Nothing here is atomic — one thread owns one element and no two
// threads meet — so what the type is doing is naming the element type, and the
// buffer is read back on the host as uint32.
struct Argmax final : ComputeProgram
{
    Argmax() { compile(); }

    void define() override
    {
        auto row = threadId();
        auto base = row * rowLength;

        auto bestIndex = var(0u);
        auto bestValue = var(0.f);
        auto chosen = var(false);
        auto scanning = var(0u);

        loop(scanning < rowLength,
             [&]
             {
                 auto index = scanning.get();
                 auto value = logits[base + index];
                 auto allowed = mask[index] == 0.f;
                 auto better = !chosen.get() || value > bestValue.get();

                 ifThen(allowed && better,
                        [&]
                        {
                            bestIndex = index;
                            bestValue = value;
                            chosen = true;
                        });

                 scanning += 1u;
             });

        write(indices, row, bestIndex.get());
    }

    Uniform<InputBuffer> logits;
    Uniform<InputBuffer> mask;
    Uniform<AtomicBuffer> indices;
    Uniform<UInt> rowLength;

    EACP_SHADER(logits, mask, indices, rowLength)
};
} // namespace WSP
