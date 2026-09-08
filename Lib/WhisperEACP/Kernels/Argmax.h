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
    using ReducingProgram::ReducingProgram;

protected:
    // What a group's fold leaves: the largest value any of its lanes held, and
    // the lowest index holding it.
    struct Candidate
    {
        Float value;
        UInt index;
    };

    // The pair fold as two group reductions rather than a tree over a pair of
    // shared arrays: the largest value over the whole group, then the lowest
    // index among the lanes holding exactly it. The equality is against the
    // fold's own bits, so a lane that holds the maximum recognises itself.
    //
    // The sentinel needs no case of its own. A lane that met no allowed
    // element carries the lowest float and unchosenIndex, so it never raises
    // the maximum, and when a real value ties its lowest float it still offers
    // unchosenIndex — the largest unsigned there is, and so the one groupMin
    // discards against any lane that chose. A group whose every lane went
    // unchosen is the one case it survives, which is exactly when the answer
    // is that nothing was chosen.
    Candidate foldCandidates(const Float& value, const UInt& index)
    {
        auto best = groupMax(value);

        return {best, groupMin(select(value == best, index, unchosenIndex))};
    }
};

// One candidate per group over rowCount * groupsPerRow groups, dispatched as
// that many whole groups of threads.
struct ArgmaxPartial final : CandidateFold
{
    explicit ArgmaxPartial(int laneCount = groupWidth)
        : CandidateFold(laneCount)
    {
        compile();
    }

    void define() override
    {
        auto lane = localId();
        auto group = groupId();
        auto row = group / groupsPerRow;
        auto part = group % groupsPerRow;
        auto base = row * rowLength;
        auto stride = groupsPerRow * lanes;

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

        auto candidate = foldCandidates(bestValue.get(), bestIndex.get());

        ifThen(lane == 0u,
               [&]
               {
                   write(partialValues, group, candidate.value);
                   write(partialIndices, group, candidate.index);
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
    explicit ArgmaxFinal(int laneCount = groupWidth)
        : CandidateFold(laneCount)
    {
        compile();
    }

    void define() override
    {
        auto lane = localId();
        auto row = groupId();
        auto base = row * partialsPerRow;

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

        auto candidate = foldCandidates(bestValue.get(), bestIndex.get());

        ifThen(lane == 0u,
               [&]
               {
                   auto chosen = candidate.index;
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
    // The group both stages are dispatched in, and what groupsPerRow spreads a
    // row over. One count serves both because there is only ever one shape
    // here: the single row of 51864 logits a decode step ends on.
    //
    // The stock 64, which is what that shape measured at — a row this wide is
    // already spread over a hundred groups, so it fills the machine at any
    // lane count, and 128 and 256 only fold deeper for it. What the two
    // reductions bought was the fold itself: 8.1 us of shared-memory pair tree
    // down to 6.4.
    static constexpr auto lanes = ComputeProgram::groupWidth;

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
    ArgmaxPartial partialStage {lanes};
    ArgmaxFinal finalStage {lanes};

    std::optional<eacp::GPU::Buffer> partialValues;
    std::optional<eacp::GPU::Buffer> partialIndices;
    int rowCapacity = 0;
    int lengthCapacity = 0;
};
} // namespace WSP
