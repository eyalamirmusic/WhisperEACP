#pragma once

#include "Attention.h"
#include "Reduce.h"

#include <limits>
#include <optional>

namespace WSP
{
// Multi-head attention for the one query a decode step has, in two dispatches
// per attention rather than the three AttentionScores, Softmax and
// AttentionApply are, and with the keys split across groups so that no loop
// is more than a few keys deep: six of a step's twelve attentions run over
// 1500 keys, and the three kernels over them were a third of the step.
//
// The layouts are Attention.h's with queryCount = 1: the query is a [W] row,
// keys and values are [keyCount, W], the output is a [W] row, and head h
// owns columns [h * D, (h + 1) * D). The query stands at the last position,
// which is a KV cache's shape and the reason no causal mask is needed: no key
// is later than it.
//
// Each group of SingleQueryAttentionPartial takes one head and one chunk of
// its keys: the lanes score their keys against the query slice held in
// registers, the group takes the chunk's maximum and the exponentials'
// sum in shared memory, and each lane accumulates the exponentials over its
// keys' value rows into a partial output row, the sixty-four rows folded to
// one. A chunk leaves its maximum, its sum and its unnormalised row behind.
// SingleQueryAttentionCombine, a group per head, joins the chunks the way a
// softmax split in two joins: every chunk rescaled by exp of its maximum
// less the largest, then the rows summed and divided by the sums. That is
// exact, and it is what lets eight groups share a head's 1500 keys.
//
// The query slice and each key and value row are read four channels at a
// time, so a head is a whole number of quads wide, and the shared arrays are
// sized for the widest head this is dispatched with and the longest chunk;
// the facade below checks all three before anything is recorded. Whisper's
// heads are 64 wide in every size.
struct SingleQueryAttentionPartial final : ReducingProgram
{
    static constexpr auto maxChunk = 256;
    static constexpr auto maxHeadWidth = 64;

    SingleQueryAttentionPartial() { compile(); }

    void define() override
    {
        constexpr auto quads = maxHeadWidth / 4;
        constexpr auto rowPitch = (unsigned) maxHeadWidth;

        auto lane = localId();
        auto group = groupId();
        auto head = group / chunkCount;
        auto chunk = group % chunkCount;
        auto headColumn = head * headWidth;

        auto firstKey = chunk * chunkLength;
        auto lastKey = min(firstKey + chunkLength, keyCount);

        auto scores = shared<Float>(maxChunk);
        auto partials = shared<Float>(groupWidth * maxHeadWidth);
        auto tile = shared<Float>(groupWidth);

        auto zero = constant(0.f);
        auto zero4 = float4(zero, zero, zero, zero);
        auto q0 = var(zero4);
        auto q1 = var(zero4);
        auto q2 = var(zero4);
        auto q3 = var(zero4);
        auto q4 = var(zero4);
        auto q5 = var(zero4);
        auto q6 = var(zero4);
        auto q7 = var(zero4);
        auto q8 = var(zero4);
        auto q9 = var(zero4);
        auto q10 = var(zero4);
        auto q11 = var(zero4);
        auto q12 = var(zero4);
        auto q13 = var(zero4);
        auto q14 = var(zero4);
        auto q15 = var(zero4);
        Var<Float4>* query[quads] = {&q0,
                                     &q1,
                                     &q2,
                                     &q3,
                                     &q4,
                                     &q5,
                                     &q6,
                                     &q7,
                                     &q8,
                                     &q9,
                                     &q10,
                                     &q11,
                                     &q12,
                                     &q13,
                                     &q14,
                                     &q15};

        for (auto quad = 0u; quad < (unsigned) quads; ++quad)
        {
            ifThen(quad * 4u < headWidth,
                   [&]
                   {
                       auto at = headColumn + quad * 4u;
                       *query[quad] = float4(queries[at],
                                             queries[at + 1u],
                                             queries[at + 2u],
                                             queries[at + 3u]);
                   });
        }

        // Scores for the chunk's keys, lane l taking keys l, l + 64, ...
        auto scoring = var(firstKey + lane);

        loop(scoring.get() < lastKey,
             [&]
             {
                 auto keyBase = scoring.get() * modelWidth + headColumn;
                 auto total = var(0.f);

                 for (auto quad = 0u; quad < (unsigned) quads; ++quad)
                 {
                     ifThen(quad * 4u < headWidth,
                            [&]
                            {
                                auto at = keyBase + quad * 4u;
                                auto key = float4(keys[at],
                                                  keys[at + 1u],
                                                  keys[at + 2u],
                                                  keys[at + 3u]);

                                total += dot(query[quad]->get(), key);
                            });
                 }

                 write(scores, scoring.get() - firstKey, scale * total.get());
                 scoring += lanes;
             });

        barrier();

        auto largest = var(std::numeric_limits<float>::lowest());
        auto scanning = var(firstKey + lane);

        loop(scanning.get() < lastKey,
             [&]
             {
                 largest = max(largest.get(), scores[scanning.get() - firstKey]);
                 scanning += lanes;
             });

        write(tile, lane, largest.get());
        barrier();
        foldMax(tile, lane);

        auto chunkMaximum = var(tile[0u]);
        barrier();

        auto acc0 = var(zero4);
        auto acc1 = var(zero4);
        auto acc2 = var(zero4);
        auto acc3 = var(zero4);
        auto acc4 = var(zero4);
        auto acc5 = var(zero4);
        auto acc6 = var(zero4);
        auto acc7 = var(zero4);
        auto acc8 = var(zero4);
        auto acc9 = var(zero4);
        auto acc10 = var(zero4);
        auto acc11 = var(zero4);
        auto acc12 = var(zero4);
        auto acc13 = var(zero4);
        auto acc14 = var(zero4);
        auto acc15 = var(zero4);
        Var<Float4>* accumulators[quads] = {&acc0,
                                            &acc1,
                                            &acc2,
                                            &acc3,
                                            &acc4,
                                            &acc5,
                                            &acc6,
                                            &acc7,
                                            &acc8,
                                            &acc9,
                                            &acc10,
                                            &acc11,
                                            &acc12,
                                            &acc13,
                                            &acc14,
                                            &acc15};

        // The exponentials, summed and applied to the value rows as they are
        // taken, so the chunk's scores are read once more and never stored
        // normalised.
        auto total = var(0.f);
        auto applying = var(firstKey + lane);

        loop(applying.get() < lastKey,
             [&]
             {
                 auto weight = var(
                     exp(scores[applying.get() - firstKey] - chunkMaximum.get()));
                 auto valueBase = applying.get() * modelWidth + headColumn;

                 total += weight.get();

                 for (auto quad = 0u; quad < (unsigned) quads; ++quad)
                 {
                     ifThen(quad * 4u < headWidth,
                            [&]
                            {
                                auto at = valueBase + quad * 4u;
                                auto row = float4(values[at],
                                                  values[at + 1u],
                                                  values[at + 2u],
                                                  values[at + 3u]);

                                *accumulators[quad] += row * weight.get();
                            });
                 }

                 applying += lanes;
             });

        write(tile, lane, total.get());

        for (auto quad = 0u; quad < (unsigned) quads; ++quad)
        {
            auto row = accumulators[quad]->get();
            auto at = lane * rowPitch + quad * 4u;

            write(partials, at, row.x());
            write(partials, at + 1u, row.y());
            write(partials, at + 2u, row.z());
            write(partials, at + 3u, row.w());
        }

        barrier();
        foldSum(tile, lane);

        ifThen(lane == 0u,
               [&]
               {
                   write(chunkMaxima, group, chunkMaximum.get());
                   write(chunkSums, group, tile[0u]);
               });

        ifThen(lane < headWidth,
               [&]
               {
                   auto sum = var(0.f);

                   for (auto other = 0u; other < lanes; ++other)
                       sum += partials[other * rowPitch + lane];

                   write(chunkRows, group * rowPitch + lane, sum.get());
               });
    }

    Uniform<InputBuffer> queries;
    Uniform<InputBuffer> keys;
    Uniform<InputBuffer> values;
    Uniform<OutputBuffer> chunkMaxima;
    Uniform<OutputBuffer> chunkSums;
    Uniform<OutputBuffer> chunkRows;
    Uniform<UInt> modelWidth;
    Uniform<UInt> headWidth;
    Uniform<UInt> keyCount;
    Uniform<UInt> chunkCount;
    Uniform<UInt> chunkLength;
    Uniform<Float> scale;

    EACP_SHADER(queries,
                keys,
                values,
                chunkMaxima,
                chunkSums,
                chunkRows,
                modelWidth,
                headWidth,
                keyCount,
                chunkCount,
                chunkLength,
                scale)
};

// A group per head, a lane per column: the chunks rescaled to the head's
// largest maximum and joined. A chunk that held no keys left the lowest float
// and a zero sum, and exp of that less any real maximum is zero.
struct SingleQueryAttentionCombine final : ReducingProgram
{
    SingleQueryAttentionCombine() { compile(); }

    void define() override
    {
        constexpr auto rowPitch =
            (unsigned) SingleQueryAttentionPartial::maxHeadWidth;

        auto lane = localId();
        auto head = groupId();
        auto firstChunk = head * chunkCount;

        auto largest = var(std::numeric_limits<float>::lowest());
        auto scanning = var(0u);

        loop(scanning.get() < chunkCount,
             [&]
             {
                 largest =
                     max(largest.get(), chunkMaxima[firstChunk + scanning.get()]);
                 scanning += 1u;
             });

        auto numerator = var(0.f);
        auto denominator = var(0.f);
        auto joining = var(0u);

        loop(joining.get() < chunkCount,
             [&]
             {
                 auto chunk = firstChunk + joining.get();
                 auto weight = exp(chunkMaxima[chunk] - largest.get());

                 denominator += chunkSums[chunk] * weight;
                 numerator += chunkRows[chunk * rowPitch + min(lane, headWidth - 1u)]
                              * weight;
                 joining += 1u;
             });

        ifThen(lane < headWidth,
               [&]
               {
                   write(output,
                         head * headWidth + lane,
                         numerator.get() / denominator.get());
               });
    }

    Uniform<InputBuffer> chunkMaxima;
    Uniform<InputBuffer> chunkSums;
    Uniform<InputBuffer> chunkRows;
    Uniform<OutputBuffer> output;
    Uniform<UInt> headWidth;
    Uniform<UInt> chunkCount;

    EACP_SHADER(chunkMaxima, chunkSums, chunkRows, output, headWidth, chunkCount)
};

// The two stages and the chunk partials between them, sized once by
// prepare() for the head count and the longest key count encode() will see.
class SingleQueryAttention
{
public:
    static constexpr auto chunksPerHead = 8;

    void prepare(eacp::GPU::Device& device,
                 int headCount,
                 int headWidth,
                 int maxKeys);
    void prepare(int headCount, int headWidth, int maxKeys);

    // Two dispatches into the caller's pass. queries is a [W] row, keys and
    // values are [keyCount, W], output receives the [W] row.
    void encode(eacp::GPU::ComputePass& pass,
                const eacp::GPU::Buffer& queries,
                const eacp::GPU::Buffer& keys,
                const eacp::GPU::Buffer& values,
                const eacp::GPU::Buffer& output,
                int modelWidth,
                int headWidth,
                int keyCount,
                float scale);

private:
    SingleQueryAttentionPartial partialStage;
    SingleQueryAttentionCombine combineStage;

    std::optional<eacp::GPU::Buffer> chunkMaxima;
    std::optional<eacp::GPU::Buffer> chunkSums;
    std::optional<eacp::GPU::Buffer> chunkRows;
    int headCapacity = 0;
    int keyCapacity = 0;
};
} // namespace WSP
