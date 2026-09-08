#pragma once

#include "Reduce.h"

#include <limits>
#include <optional>

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
// Two dispatches, because a 51864-wide row scanned by one thread was 3.9 ms
// of a 9.4 ms decode step. ArgmaxPartial spreads a row over groupsPerRow
// groups: every thread walks its strided share in ascending order keeping the
// first maximum it meets, the group folds its candidates in shared memory, and
// one (value, index) pair per group comes out. ArgmaxFinal folds a row's
// pairs the same way and writes the index.
//
// A candidate is a pair, and merging two prefers the larger value and, on
// equal values, the lower index, so the fold ends on exactly the index a
// left-to-right scan would have stopped at — lowest index on ties, which is
// what torch.argmax documents. A thread or a group that met no allowed
// element carries the sentinel index and loses to any that did; a row whose
// every token is suppressed comes out as index zero, a case Whisper's lists
// never produce and which is stated so the value is documented rather than
// incidental.
//
// The mask is one buffer and the caller decides which one. Whisper ships two
// lists — suppress_tokens, which applies at every step, and
// begin_suppress_tokens, which applies only at the first — and folding them
// into a rule here would put the generation config inside a kernel.
//
// The indices come out through an integer buffer, so the step that follows
// can read them as its tokens without the host in between.
inline constexpr auto unchosenIndex = 0xFFFFFFFFu;

struct CandidateFold : ReducingProgram
{
protected:
    void foldCandidates(const Shared<Float>& values,
                        const Shared<UInt>& indices,
                        const UInt& lane)
    {
        for (auto span = lanes / 2u; span > 0u; span /= 2u)
        {
            ifThen(lane < span,
                   [&]
                   {
                       auto value = values[lane];
                       auto index = indices[lane];
                       auto otherValue = values[lane + span];
                       auto otherIndex = indices[lane + span];

                       auto larger = otherValue > value
                                     || (otherValue == value && otherIndex < index);
                       auto takeOther = otherIndex != unchosenIndex
                                        && (index == unchosenIndex || larger);

                       ifThen(takeOther,
                              [&]
                              {
                                  write(values, lane, otherValue);
                                  write(indices, lane, otherIndex);
                              });
                   });

            barrier();
        }
    }
};

// One candidate per group over rowCount * groupsPerRow groups, dispatched as
// that many whole groups of threads.
struct ArgmaxPartial final : CandidateFold
{
    ArgmaxPartial() { compile(); }

    void define() override
    {
        auto lane = localId();
        auto group = groupId();
        auto row = group / groupsPerRow;
        auto part = group % groupsPerRow;
        auto base = row * rowLength;
        auto stride = groupsPerRow * lanes;

        auto values = shared<Float>(groupWidth);
        auto indices = shared<UInt>(groupWidth);

        auto bestValue = var(std::numeric_limits<float>::lowest());
        auto bestIndex = var(unchosenIndex);
        auto scanning = var(part * lanes + lane);

        loop(scanning.get() < rowLength,
             [&]
             {
                 auto index = scanning.get();
                 auto value = logits[base + index];
                 auto allowed = mask[index] == 0.f;
                 auto better =
                     bestIndex.get() == unchosenIndex || value > bestValue.get();

                 ifThen(allowed && better,
                        [&]
                        {
                            bestValue = value;
                            bestIndex = index;
                        });

                 scanning += stride;
             });

        write(values, lane, bestValue.get());
        write(indices, lane, bestIndex.get());
        barrier();
        foldCandidates(values, indices, lane);

        ifThen(lane == 0u,
               [&]
               {
                   write(partialValues, group, values[0u]);
                   write(partialIndices, group, indices[0u]);
               });
    }

    Uniform<InputBuffer> logits;
    Uniform<InputBuffer> mask;
    Uniform<OutputBuffer> partialValues;
    Uniform<UIntOutputBuffer> partialIndices;
    Uniform<UInt> rowLength;
    Uniform<UInt> groupsPerRow;

    EACP_SHADER(logits, mask, partialValues, partialIndices, rowLength, groupsPerRow)
};

// One group per row over its partialsPerRow candidates: dispatchRows(pass,
// rowCount).
struct ArgmaxFinal final : CandidateFold
{
    ArgmaxFinal() { compile(); }

    void define() override
    {
        auto lane = localId();
        auto row = groupId();
        auto base = row * partialsPerRow;

        auto values = shared<Float>(groupWidth);
        auto indices = shared<UInt>(groupWidth);

        auto bestValue = var(std::numeric_limits<float>::lowest());
        auto bestIndex = var(unchosenIndex);
        auto scanning = var(lane);

        loop(scanning.get() < partialsPerRow,
             [&]
             {
                 auto value = partialValues[base + scanning.get()];
                 auto index = partialIndices[base + scanning.get()];

                 auto larger =
                     value > bestValue.get()
                     || (value == bestValue.get() && index < bestIndex.get());
                 auto take = index != unchosenIndex
                             && (bestIndex.get() == unchosenIndex || larger);

                 ifThen(take,
                        [&]
                        {
                            bestValue = value;
                            bestIndex = index;
                        });

                 scanning += lanes;
             });

        write(values, lane, bestValue.get());
        write(indices, lane, bestIndex.get());
        barrier();
        foldCandidates(values, indices, lane);

        ifThen(lane == 0u,
               [&]
               {
                   auto chosen = indices[0u];
                   write(result, row, select(chosen == unchosenIndex, 0u, chosen));
               });
    }

    Uniform<InputBuffer> partialValues;
    Uniform<UIntInputBuffer> partialIndices;
    Uniform<UIntOutputBuffer> result;
    Uniform<UInt> partialsPerRow;

    EACP_SHADER(partialValues, partialIndices, result, partialsPerRow)
};

// The two stages and the candidates between them. prepare() sizes the
// candidate buffers for the largest shape encode() will be given, the way the
// decoder sizes its intermediates once; a larger one is an error before
// anything is recorded.
class Argmax
{
public:
    void prepare(eacp::GPU::Device& device, int rowCount, int rowLength);
    void prepare(int rowCount, int rowLength);

    // Two dispatches into the caller's pass: the candidates, then the answer.
    // logits is a range so a caller holding several rows can sample the last
    // of them without scanning the rest, and indices is one so the answer
    // lands in the slot of a sequence the next step reads.
    void encode(eacp::GPU::ComputePass& pass,
                const eacp::GPU::BufferRange& logits,
                const eacp::GPU::Buffer& mask,
                const eacp::GPU::BufferRange& indices,
                int rowCount,
                int rowLength);

    void encode(eacp::GPU::ComputePass& pass,
                const eacp::GPU::BufferRange& logits,
                const eacp::GPU::Buffer& mask,
                const eacp::GPU::Buffer& indices,
                int rowCount,
                int rowLength)
    {
        encode(pass,
               logits,
               mask,
               eacp::GPU::BufferRange::of(indices),
               rowCount,
               rowLength);
    }

    // How many groups a row is spread over: about eight elements per thread,
    // and never fewer than one group.
    static int groupsPerRow(int rowLength);

private:
    ArgmaxPartial partialStage;
    ArgmaxFinal finalStage;

    std::optional<eacp::GPU::Buffer> partialValues;
    std::optional<eacp::GPU::Buffer> partialIndices;
    int rowCapacity = 0;
    int lengthCapacity = 0;
};
} // namespace WSP
