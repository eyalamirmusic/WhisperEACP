#pragma once

#include "Attention.h"
#include "Reduce.h"

#include <limits>
#include <optional>

namespace WSP
{
// Multi-head attention for the one query a decode step has, in one or two
// dispatches per attention rather than the three AttentionScores, Softmax and
// AttentionApply are: six of a step's twelve attentions run over 1500 keys,
// and the three kernels over them were a third of the step.
//
// Two kernels, and which one runs is the key count. Over a long cache the keys
// are split across groups so that no loop is more than a few keys deep, and
// the partials are joined; over a short one a single group takes the whole row
// with a lane per output column. See SingleQueryAttentionRow.
//
// The layouts are Attention.h's with queryCount = 1: the query is a [W] row,
// keys and values are [keyCount, W], the output is a [W] row, and head h
// owns columns [h * D, (h + 1) * D). The query stands at the last position,
// which is a KV cache's shape and the reason no causal mask is needed: no key
// is later than it.
//
// Each group of SingleQueryAttentionPartial takes one head and one chunk of
// its keys: the lanes score their keys against the query slice held in
// registers, groupMax and groupSum take the chunk's maximum and the
// exponentials' sum, and each lane accumulates the exponentials over its keys'
// value rows into a partial output row, one such row per lane folded to one. A
// chunk leaves its maximum, its sum and its unnormalised row behind.
// SingleQueryAttentionCombine, a group per head, joins the chunks the way a
// softmax split in two joins: every chunk rescaled by exp of its maximum
// less the largest, then the rows summed and divided by the sums. That is
// exact, and it is what lets a couple of dozen groups share a head's 1500
// keys.
//
// That fold of the value rows is per column rather than group-wide — column c
// wants the sum over lanes of each lane's row, which is as many reductions as
// a head is wide — so it stays a walk over the shared rows. Those rows are
// also what holds the lane count at the stock 64: a row per lane at the widest
// head is lanes * 64 floats of threadgroup memory, and 64 lanes already spends
// half a Metal group's 32 kB on them, so 128 is a pipeline the backend
// refuses. Fewer buys the occupancy that costs and loses more than it buys —
// over jfk.wav's decode at the chunk counts each prefers, 16 lanes 0.01207 s,
// 32 lanes 0.01138, **64 lanes 0.01104**.
//
// A cache short enough for one group takes one dispatch instead of two, and
// takes it through SingleQueryAttentionRow below rather than through this
// kernel at chunkCount one — the two are written the other way round from each
// other and each is fastest at one end of the key count. That short cache is
// what a step's self attention is: it runs over the tokens decoded so far, two
// at the prompt and a couple of dozen by the end of a sentence.
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

    explicit SingleQueryAttentionPartial(int laneCount = groupWidth)
        : ReducingProgram(laneCount)
    {
        compile();
    }

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
        auto partials = shared<Float>((int) lanes * maxHeadWidth);

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

        auto chunkMaximum = var(groupMax(largest.get()));

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

        auto chunkTotal = var(groupSum(total.get()));

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

        ifThen(lane == 0u,
               [&]
               {
                   write(chunkMaxima, group, chunkMaximum.get());
                   write(chunkSums, group, chunkTotal.get());
               });

        auto column = var(lane);

        loop(column.get() < headWidth,
             [&]
             {
                 auto sum = var(0.f);

                 for (auto other = 0u; other < lanes; ++other)
                     sum += partials[other * rowPitch + column.get()];

                 write(chunkRows, group * rowPitch + column.get(), sum.get());
                 column += lanes;
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
    explicit SingleQueryAttentionCombine(int laneCount = groupWidth)
        : ReducingProgram(laneCount)
    {
        compile();
    }

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

// The same attention for a cache one group can take whole — a group per head,
// nothing to rescale and nothing to join — written the other way round: **a
// lane per output column rather than a lane per key**.
//
// That is what makes it a kernel of its own rather than the partial with
// chunkCount at one. A lane there carries the head's whole width in
// accumulators and leaves a row of them behind for the group to fold, which is
// 64 registers a thread and 16 kB of threadgroup memory a group whatever the
// cache holds; a lane here carries one accumulator and walks the value rows in
// order, the group reading one row's columns at a time in one coalesced read,
// for a chunk of scores and a handful of registers. The query is read from the
// buffer inside the scoring loop rather than staged: every lane wants the same
// 256 bytes, which is a cache hit, and not staging it is one barrier and one
// shared array fewer.
//
// What it costs is that the walk is as deep as the cache rather than as deep
// as a lane's share of it, so this is the right shape only while the cache is
// short. Both were measured on jfk.wav at 384 wide over six 64-wide heads:
// this form takes a step's self attention over a couple of dozen keys from
// 17.7 us to 9.2, and its cross attention over 1500 from 21.3 to 36.0. Hence
// two kernels, and singleChunkKeyLimit between them.
struct SingleQueryAttentionRow final : ReducingProgram
{
    static constexpr auto maxKeys = 256;
    static constexpr auto maxHeadWidth = SingleQueryAttentionPartial::maxHeadWidth;

    explicit SingleQueryAttentionRow(int laneCount = groupWidth)
        : ReducingProgram(laneCount)
    {
        compile();
    }

    void define() override
    {
        auto lane = localId();
        auto head = groupId();
        auto headColumn = head * headWidth;

        auto scores = shared<Float>(maxKeys);

        auto largest = var(std::numeric_limits<float>::lowest());
        auto scoring = var(lane);

        loop(scoring.get() < keyCount,
             [&]
             {
                 auto keyBase = scoring.get() * modelWidth + headColumn;
                 auto total = var(0.f);
                 auto channel = var(0u);

                 loop(channel.get() < headWidth,
                      [&]
                      {
                          auto row = keys.read4((keyBase + channel.get()) / 4u);
                          auto slice =
                              queries.read4((headColumn + channel.get()) / 4u);

                          total += dot(slice, row);
                          channel += 4u;
                      });

                 auto score = var(scale * total.get());

                 write(scores, scoring.get(), score.get());
                 largest = max(largest.get(), score.get());
                 scoring += lanes;
             });

        auto rowMaximum = var(groupMax(largest.get()));

        // The exponentials, left where the scores that made them were. A lane
        // reads back only what it wrote, so nothing here has to be visible
        // across the group until the walk below.
        auto total = var(0.f);
        auto exponentiating = var(lane);

        loop(exponentiating.get() < keyCount,
             [&]
             {
                 auto weight =
                     var(exp(scores[exponentiating.get()] - rowMaximum.get()));

                 write(scores, exponentiating.get(), weight.get());
                 total += weight.get();
                 exponentiating += lanes;
             });

        auto denominator = var(groupSum(total.get()));

        barrier();

        auto column = var(lane);

        loop(column.get() < headWidth,
             [&]
             {
                 auto sum = var(0.f);
                 auto key = var(0u);

                 loop(key.get() < keyCount,
                      [&]
                      {
                          sum += scores[key.get()]
                                 * values[key.get() * modelWidth + headColumn
                                          + column.get()];
                          key += 1u;
                      });

                 write(output,
                       headColumn + column.get(),
                       sum.get() / denominator.get());
                 column += lanes;
             });
    }

    Uniform<InputBuffer> queries;
    Uniform<InputBuffer> keys;
    Uniform<InputBuffer> values;
    Uniform<OutputBuffer> output;
    Uniform<UInt> modelWidth;
    Uniform<UInt> headWidth;
    Uniform<UInt> keyCount;
    Uniform<Float> scale;

    EACP_SHADER(
        queries, keys, values, output, modelWidth, headWidth, keyCount, scale)
};

// The three stages and the chunk partials between two of them, sized once by
// prepare() for the head count and the longest key count encode() will see.
class SingleQueryAttention
{
public:
    // How many groups share a head's keys once there are too many for one.
    // With SingleQueryAttentionRow taking everything up to 256 keys, this only
    // ever describes a decode step's cross attention over the encoder's 1500
    // rows, and it is an occupancy number rather than a work one: six heads at
    // eight chunks is 48 groups, which does not fill this machine.
    //
    // The curve has a floor and then a cliff, measured over jfk.wav's decode
    // interleaved three times: 8 chunks 0.01156 s, 16 0.01137, **24 0.01107**,
    // 28 0.01204, 32 0.01205, 48 0.01249. Past 24 the partials the combine has
    // to join cost more than the extra groups buy, and they cost it sharply.
    static constexpr auto chunksPerHead = 24;

    // Up to how many keys one group takes on its own, which is the longest row
    // SingleQueryAttentionRow's shared score array holds. A step's cross
    // attention is always over it; its self attention is under it for the first
    // 256 tokens of a sequence and chunks like the cross attention past them,
    // which a 448-position model reaches on a full window of dense speech.
    static constexpr auto singleChunkKeyLimit = SingleQueryAttentionRow::maxKeys;

    void prepare(eacp::GPU::Device& device,
                 int headCount,
                 int headWidth,
                 int maxKeys);
    void prepare(int headCount, int headWidth, int maxKeys);

    // One dispatch into the caller's pass up to singleChunkKeyLimit keys and
    // two beyond it. queries is a [W] row, keys and values are [keyCount, W],
    // output receives the [W] row.
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
    SingleQueryAttentionRow rowStage;
    SingleQueryAttentionPartial partialStage;
    SingleQueryAttentionCombine combineStage;

    std::optional<eacp::GPU::Buffer> chunkMaxima;
    std::optional<eacp::GPU::Buffer> chunkSums;
    std::optional<eacp::GPU::Buffer> chunkRows;
    int headCapacity = 0;
    int keyCapacity = 0;
};
} // namespace WSP
