#include "SingleQueryAttention.h"

#include <stdexcept>
#include <string>

namespace WSP
{
using eacp::GPU::Buffer;
using eacp::GPU::BufferUsage;
using eacp::GPU::ComputePass;
using eacp::GPU::Device;

namespace
{
constexpr int floatBytes(int elementCount)
{
    return elementCount * (int) sizeof(float);
}

int chunkLengthFor(int keyCount)
{
    return (keyCount + SingleQueryAttention::chunksPerHead - 1)
           / SingleQueryAttention::chunksPerHead;
}
} // namespace

void SingleQueryAttention::prepare(Device& device,
                                   int headCount,
                                   int headWidth,
                                   int maxKeys)
{
    if (headWidth > SingleQueryAttentionPartial::maxHeadWidth || headWidth % 4 != 0
        || chunkLengthFor(maxKeys) > SingleQueryAttentionPartial::maxChunk)
        throw std::invalid_argument {
            "SingleQueryAttention holds heads up to "
            + std::to_string(SingleQueryAttentionPartial::maxHeadWidth)
            + " wide in quads, and chunks up to "
            + std::to_string(SingleQueryAttentionPartial::maxChunk)
            + " keys, and was asked for " + std::to_string(headWidth) + " and "
            + std::to_string(chunkLengthFor(maxKeys))};

    partialStage.prepare(device);
    combineStage.prepare(device);

    const auto chunks = headCount * chunksPerHead;

    chunkMaxima.emplace(device.makeBuffer(floatBytes(chunks), BufferUsage::Storage));
    chunkSums.emplace(device.makeBuffer(floatBytes(chunks), BufferUsage::Storage));
    chunkRows.emplace(device.makeBuffer(
        floatBytes(chunks * SingleQueryAttentionPartial::maxHeadWidth),
        BufferUsage::Storage));

    headCapacity = headCount;
    keyCapacity = maxKeys;
}

void SingleQueryAttention::prepare(int headCount, int headWidth, int maxKeys)
{
    prepare(Device::shared(), headCount, headWidth, maxKeys);
}

void SingleQueryAttention::encode(ComputePass& pass,
                                  const Buffer& queries,
                                  const Buffer& keys,
                                  const Buffer& values,
                                  const Buffer& output,
                                  int modelWidth,
                                  int headWidth,
                                  int keyCount,
                                  float scale)
{
    const auto headCount = modelWidth / headWidth;

    if (headCount > headCapacity || keyCount > keyCapacity)
        throw std::invalid_argument {
            "SingleQueryAttention was prepared for " + std::to_string(headCapacity)
            + " heads over " + std::to_string(keyCapacity) + " keys and asked for "
            + std::to_string(headCount) + " over " + std::to_string(keyCount)};

    partialStage.queries = queries;
    partialStage.keys = keys;
    partialStage.values = values;
    partialStage.chunkMaxima = *chunkMaxima;
    partialStage.chunkSums = *chunkSums;
    partialStage.chunkRows = *chunkRows;
    partialStage.modelWidth = (std::uint32_t) modelWidth;
    partialStage.headWidth = (std::uint32_t) headWidth;
    partialStage.keyCount = (std::uint32_t) keyCount;
    partialStage.chunkCount = (std::uint32_t) chunksPerHead;
    partialStage.chunkLength = (std::uint32_t) chunkLengthFor(keyCount);
    partialStage.scale = scale;

    partialStage.dispatchRows(pass, headCount * chunksPerHead);

    // The combine folds what the partials wrote.
    pass.barrier();

    combineStage.chunkMaxima = *chunkMaxima;
    combineStage.chunkSums = *chunkSums;
    combineStage.chunkRows = *chunkRows;
    combineStage.output = output;
    combineStage.headWidth = (std::uint32_t) headWidth;
    combineStage.chunkCount = (std::uint32_t) chunksPerHead;

    combineStage.dispatchRows(pass, headCount);
}
} // namespace WSP
