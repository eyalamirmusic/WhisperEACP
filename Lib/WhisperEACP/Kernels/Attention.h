#pragma once

#include "KernelTypes.h"

namespace WSP
{
// Multi-head attention as its two halves, with the softmax between them run by
// the Softmax kernel next door: scores, then normalise, then apply. Splitting
// it here rather than fusing it is what lets the three be checked against a
// scalar reference one at a time, and what lets the softmax stay the one
// numerically careful kernel instead of being written out twice.
//
// The layouts, over a model width W split into H heads of D = W / H:
//
//   queries       [queryCount, W]              row-major
//   keys, values  [keyCount,   W]              row-major
//   scores        [H, queryCount, keyCount]    row-major
//   output        [queryCount, W]              row-major
//
// Head h owns columns [h * D, (h + 1) * D) of every W-wide row, so the heads
// are strided slices of one buffer rather than buffers of their own — which is
// what the projections produce, and what the output projection consumes.
//
// queryCount and keyCount are separately uniform because they differ wherever
// it matters: cross-attention runs the decoder's queries against 1500 encoder
// keys, and a KV cache runs one query against every key cached so far. Nothing
// below assumes the score matrix is square.
//
// The scores buffer is exactly what Softmax normalises with no reshaping:
// [H, queryCount, keyCount] row-major is H * queryCount rows of keyCount, so
// dispatch(softmax, H * queryCount) with rowLength = keyCount is the softmax
// over j the definition asks for.

// The score a causally masked key gets, chosen so that Softmax next door turns
// it into a weight of exactly zero and nothing else.
//
// The softmax subtracts the row maximum before exponentiating, so what has to
// underflow is sentinel - rowMaximum, and exp of anything below about -104 is
// zero in float32 — this leaves twenty-eight orders of magnitude of margin over
// that, whatever the real scores are. It is finite rather than -infinity
// because -inf - -inf is a NaN, and it stops eight orders of magnitude short of
// -FLT_MAX so that the subtraction cannot overflow to -infinity either.
//
// It is never the row maximum: query i sits at absolute position
// keyCount - queryCount + i, which is a key index in range, so every row has
// its own diagonal unmasked and therefore at least one real score in it.
inline constexpr auto causalMaskScore = -1e30f;

// scores[h, i, j] = scale * dot(queries[i, head h], keys[j, head h]), with
// scale = D^-0.5 supplied by the caller rather than derived here, since the
// caller is the one that knows D as a number.
//
// With causal nonzero, a key later than its query is masked instead: query i
// of queryCount stands at absolute position keyCount - queryCount + i, so the
// queries are the last queryCount positions of the keyCount keys, and key j is
// suppressed when j is past that. That is a KV cache's own shape — n keys
// cached, one query appended, the query at position n — and it is the encoder's
// shape too, where queryCount == keyCount makes it the plain lower triangle.
// The comparison is spelled j + queryCount > i + keyCount so that unsigned
// arithmetic never has to go below zero.
//
// With causal zero nothing is masked and the kernel is what it was: the same
// product, the same store, the mask only choosing between two values already
// computed.
//
// One thread per (j, (h, i)) over a 2D grid: dispatch(kernel, keyCount,
// H * queryCount). eacp dispatches 1D or 2D and no further, so the head is
// folded into the row index — position.y is h * queryCount + i, and the body
// takes it apart again by queryCount. That the fold is exactly the scores
// buffer's own row index is why the write below needs no arithmetic beyond it.
//
// keyCount is a uniform despite also being the dispatch width, for MatMul's
// reason: it is the stride the scores buffer is walked at, and a stride the
// body needs is a uniform whether or not the grid happens to share it.
struct AttentionScores final : ComputeProgram
{
    AttentionScores() { compile(); }

    void define() override
    {
        auto position = threadPosition();
        auto key = position.x;
        auto head = position.y / queryCount;
        auto query = position.y % queryCount;

        auto headColumn = head * headWidth;
        auto queryBase = query * modelWidth + headColumn;
        auto keyBase = key * modelWidth + headColumn;

        auto total = var(0.f);
        auto channel = var(0u);

        loop(channel < headWidth,
             [&]
             {
                 total += queries[queryBase + channel] * keys[keyBase + channel];
                 channel += 1u;
             });

        auto laterThanQuery = key + queryCount > query + keyCount;
        auto suppressed = causal != 0u && laterThanQuery;

        write(scores,
              position.y * keyCount + key,
              select(suppressed, causalMaskScore, scale * total.get()));
    }

    Uniform<InputBuffer> queries;
    Uniform<InputBuffer> keys;
    Uniform<OutputBuffer> scores;
    Uniform<UInt> modelWidth;
    Uniform<UInt> headWidth;
    Uniform<UInt> queryCount;
    Uniform<UInt> keyCount;
    Uniform<UInt> causal;
    Uniform<Float> scale;

    EACP_SHADER(queries,
                keys,
                scores,
                modelWidth,
                headWidth,
                queryCount,
                keyCount,
                causal,
                scale)
};

// output[i, h * D + c] = sum over j of probabilities[h, i, j] * values[j,
// h * D + c], the softmaxed scores read back over V. What comes out is
// [queryCount, W] — the concatenation of the heads, which is what the output
// projection is a plain matmul over.
//
// One thread per (h * D + c, i) over a 2D grid: dispatch(kernel, W,
// queryCount). The head is recovered from the column rather than passed in,
// since h * D + c divided by D is h whatever c is, and c itself is never
// needed as a number: the column indexes V and the output directly.
struct AttentionApply final : ComputeProgram
{
    AttentionApply() { compile(); }

    void define() override
    {
        auto position = threadPosition();
        auto column = position.x;
        auto query = position.y;
        auto head = column / headWidth;

        auto probabilityBase = (head * queryCount + query) * keyCount;

        auto total = var(0.f);
        auto key = var(0u);

        loop(key < keyCount,
             [&]
             {
                 total += probabilities[probabilityBase + key]
                          * values[key * modelWidth + column];
                 key += 1u;
             });

        write(output, query * modelWidth + column, total.get());
    }

    Uniform<InputBuffer> probabilities;
    Uniform<InputBuffer> values;
    Uniform<OutputBuffer> output;
    Uniform<UInt> modelWidth;
    Uniform<UInt> headWidth;
    Uniform<UInt> queryCount;
    Uniform<UInt> keyCount;

    EACP_SHADER(
        probabilities, values, output, modelWidth, headWidth, queryCount, keyCount)
};
} // namespace WSP
