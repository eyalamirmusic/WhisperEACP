#pragma once

#include <WhisperEACP/Kernels/Kernels.h>
#include <WhisperEACP/Model/ModelConfig.h>

namespace WSP
{
// Every extent one decoder is built against. All of them reach the kernels as
// uniforms, so this is a description of a model rather than a set of numbers
// compiled in: fromConfig reads tiny.en's out of config.json, and a test writes
// its own small ones.
//
// Two of the seven are capacities rather than extents of a single dispatch.
// maxPositions is max_target_positions — the row count of the positional table
// and, with it, how many tokens the self-attention KV cache can hold before the
// sequence has to end. crossPositions is the encoder output's row count, a
// parameter rather than 1500 for the reason EncoderShape::inputFrames is one:
// the same dispatches over a shorter encoder output are what let the real
// weights be checked against a scalar reference at a size the CPU can afford.
struct DecoderShape
{
    int width = 0;
    int heads = 0;
    int layers = 0;
    int feedForwardWidth = 0;
    int vocabularySize = 0;
    int maxPositions = 0;
    int crossPositions = 0;

    static DecoderShape fromConfig(const ModelConfig& config,
                                   int crossPositionCount);

    // Weights are loaded against one of these and dispatched against another,
    // and nothing about a GPU buffer says which — so the two are compared
    // outright before the first dispatch.
    friend bool operator==(const DecoderShape&, const DecoderShape&) = default;

    int headWidth() const { return width / heads; }

    // One row of each of the three widths a step walks. A step of n tokens is n
    // of them: the prompt a sequence opens with is decoded in one call, so a
    // step is not a single token and none of the counts below is a buffer size.
    int rowElementCount() const { return width; }
    int feedForwardElementCount() const { return feedForwardWidth; }
    int logitElementCount() const { return vocabularySize; }

    // What every per-token intermediate is sized at, once: the largest step is
    // a prompt filling the whole window. The same number as the cache's
    // capacity below, and for the same reason — a step cannot be longer than
    // what it appends to.
    int stepElementCount() const { return maxPositions * width; }

    int stepFeedForwardElementCount() const
    {
        return maxPositions * feedForwardWidth;
    }

    // Both KV caches, sized for the whole run rather than for one step: the
    // self-attention pair grows a row per token up to maxPositions, and the
    // cross-attention pair is projected once from the encoder output and then
    // read by every step.
    int cacheElementCount() const { return maxPositions * width; }
    int crossElementCount() const { return crossPositions * width; }

    // [heads, queries, keys] row-major, which is heads * queries rows of keys —
    // exactly what Softmax normalises with no reshaping. Both are capacities
    // rather than a step's extents: a buffer sized for the longest step a
    // sequence can take serves every shorter one, and the cache is what the key
    // count grows towards.
    int selfScoreElementCount() const { return heads * maxPositions * maxPositions; }

    int crossScoreElementCount() const
    {
        return heads * maxPositions * crossPositions;
    }

    // The head fold the attention kernels dispatch over, which is the score
    // buffer's own row index — a query count rather than a shape constant,
    // since a step decides how many queries it has.
    int scoreRowCount(int queryCount) const { return heads * queryCount; }

    float attentionScale() const;
};
} // namespace WSP
