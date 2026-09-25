#include "Decoder.h"

#include <string>

namespace WSP
{
using eacp::GPU::Buffer;
using eacp::GPU::BufferRange;
using eacp::GPU::BufferUsage;
using eacp::GPU::ComputePass;
using eacp::GPU::Device;

namespace
{
void addCaches(Vector<Cache>& caches, Net& net, int layers, int rows, int width)
{
    caches.clear();

    for (auto index = 0; index < layers; ++index)
        caches.add(net.makeCache(rows, width));
}

// Self-attention over the sequence so far: the queries are this step's, the
// keys and values are every token's, and the mask is what keeps token t out of
// its own future. Query i stands at absolute position keyCount - queryCount + i
// — which is exactly where this step's rows landed in the cache — so a prompt
// decoded in one call is masked the way the same tokens fed one at a time
// would be.
Tensor selfAttention(Net& net,
                     const DecoderShape& shape,
                     const DecoderLayerWeights& weights,
                     DecoderCaches& caches,
                     int layerIndex,
                     const Tensor& normalised)
{
    const auto queries = net.linear(normalised,
                                    weights.selfQueryWeight,
                                    &weights.selfQueryBias,
                                    Activation::none);
    const auto keys = net.appendLinear(
        caches.selfKeys[layerIndex], normalised, weights.selfKeyWeight, nullptr);
    const auto values = net.appendLinear(caches.selfValues[layerIndex],
                                         normalised,
                                         weights.selfValueWeight,
                                         &weights.selfValueBias);

    return net.attention(queries, keys, values, shape.heads, true);
}

// Cross-attention against the encoder's rows: only the queries are projected
// here, since the keys and values were projected once when the sequence
// opened. Nothing is masked — every token attends to the whole utterance.
Tensor crossAttention(Net& net,
                      const DecoderShape& shape,
                      const DecoderLayerWeights& weights,
                      DecoderCaches& caches,
                      int layerIndex,
                      const Tensor& normalised)
{
    const auto queries = net.linear(normalised,
                                    weights.crossQueryWeight,
                                    &weights.crossQueryBias,
                                    Activation::none);
    const auto keys = net.cached(caches.crossKeys[layerIndex]);
    const auto values = net.cached(caches.crossValues[layerIndex]);

    return net.attention(queries, keys, values, shape.heads, false);
}

// Pre-norm, which is what Whisper is: each of the three sublayers normalises
// what it reads and adds what it computed to what it was given, so the residual
// is the unnormalised input.
Tensor decoderLayer(Net& net,
                    const DecoderShape& shape,
                    const DecoderLayerWeights& weights,
                    DecoderCaches& caches,
                    int layerIndex,
                    const Tensor& hidden)
{
    const auto afterSelf =
        net.linearAdd(selfAttention(net,
                                    shape,
                                    weights,
                                    caches,
                                    layerIndex,
                                    net.layerNorm(hidden,
                                                  weights.selfAttentionNormWeight,
                                                  weights.selfAttentionNormBias)),
                      weights.selfAttentionOutputWeight,
                      &weights.selfAttentionOutputBias,
                      hidden);

    const auto afterCross =
        net.linearAdd(crossAttention(net,
                                     shape,
                                     weights,
                                     caches,
                                     layerIndex,
                                     net.layerNorm(afterSelf,
                                                   weights.crossAttentionNormWeight,
                                                   weights.crossAttentionNormBias)),
                      weights.crossAttentionOutputWeight,
                      &weights.crossAttentionOutputBias,
                      afterSelf);

    return net.linearAdd(net.linear(net.layerNorm(afterCross,
                                                  weights.finalNormWeight,
                                                  weights.finalNormBias),
                                    weights.feedForwardWeight,
                                    &weights.feedForwardBias,
                                    Activation::gelu),
                         weights.feedForwardOutputWeight,
                         &weights.feedForwardOutputBias,
                         afterCross);
}
} // namespace

// k_proj has no bias in either attention — that is PyTorch's own bias=False on
// those two Linears. All eight projections read the encoder's output and write
// caches of their own, so nothing here is ordered against anything else here.
void recordSequenceStart(Net& net,
                         const DecoderShape& shape,
                         const DecoderWeights& weights,
                         DecoderCaches& caches,
                         const Binding& encoderRows,
                         int rows)
{
    const auto encoded = net.input(encoderRows, {rows, shape.width}, DType::float32);

    for (auto index = 0; index < shape.layers; ++index)
    {
        const auto& layer = weights.layers[index];

        caches.selfKeys[index].clear();
        caches.selfValues[index].clear();
        caches.crossKeys[index].clear();
        caches.crossValues[index].clear();

        net.appendLinear(
            caches.crossKeys[index], encoded, layer.crossKeyWeight, nullptr);
        net.appendLinear(caches.crossValues[index],
                         encoded,
                         layer.crossValueWeight,
                         &layer.crossValueBias);
    }
}

// The logits projection is embed_tokens itself — there is no proj_out in a
// Whisper safetensors file, and no bias either. logitsWeight() is that matrix,
// or the fp16 copy of it the weights were asked to keep beside it.
void recordDecoderStep(Net& net,
                       const DecoderShape& shape,
                       const DecoderWeights& weights,
                       DecoderCaches& caches,
                       const Binding& tokens,
                       int tokenCount,
                       int firstPosition,
                       const Binding& hiddenStates,
                       const Binding& logits)
{
    auto hidden = net.embed(net.input(tokens, {tokenCount}, DType::int32),
                            weights.tokenEmbedding,
                            weights.positionalEmbedding,
                            firstPosition);

    for (auto index = 0; index < shape.layers; ++index)
        hidden =
            decoderLayer(net, shape, weights.layers[index], caches, index, hidden);

    const auto normalised =
        net.layerNorm(hidden, weights.finalNormWeight, weights.finalNormBias);
    net.output(normalised, hiddenStates);

    net.output(
        net.linear(normalised, weights.logitsWeight(), nullptr, Activation::none),
        logits);
}

Decoder::Decoder(const DecoderShape& shapeToUse)
    : decoderShape(shapeToUse)
{
}

// Every intermediate is sized for the longest step a sequence can take rather
// than for the step in hand, so a prompt of several tokens and a single token
// after it are the same buffers at different dispatch heights.
//
// The two score buffers are what that costs, and they are the only allocations
// here worth a number: [heads, maxPositions, keys] floats each, which for
// tiny.en's 6 heads and 448 target positions is 4.8 MB against the cache and
// 16 MB against a full 1500-row encoder output — the softmax normalises them
// in place, so there is no second pair — beside 5.5 MB of self-attention
// cache and 18 MB of cross-attention keys and values over the four layers.
void Decoder::prepare(Device& device)
{
    const auto width = decoderShape.width;
    const auto layers = decoderShape.layers;

    auto layout = KernelNetLayout {};
    layout.heads = decoderShape.heads;
    layout.headWidth = decoderShape.headWidth();
    layout.widestLayerOutput = decoderShape.feedForwardWidth;

    layout.scratch.add(ScratchReservation {decoderShape.stepElementCount(), 4});
    layout.scratch.add(
        ScratchReservation {decoderShape.stepFeedForwardElementCount(), 1});
    layout.attention.add(
        AttentionReservation {decoderShape.maxPositions, decoderShape.maxPositions});
    layout.attention.add(AttentionReservation {decoderShape.maxPositions,
                                               decoderShape.crossPositions});
    layout.zeroBiasLengths.add(width);
    layout.zeroBiasLengths.add(decoderShape.vocabularySize);

    net.prepare(device, layout);

    addCaches(caches.selfKeys, net, layers, decoderShape.maxPositions, width);
    addCaches(caches.selfValues, net, layers, decoderShape.maxPositions, width);
    addCaches(caches.crossKeys, net, layers, decoderShape.crossPositions, width);
    addCaches(caches.crossValues, net, layers, decoderShape.crossPositions, width);

    normalisedRows.emplace(
        device.makeBuffer(decoderShape.stepElementCount() * (int) sizeof(float),
                          BufferUsage::Storage));

    decodedPositions = 0;
    activeCrossPositions = decoderShape.crossPositions;
}

void Decoder::prepare()
{
    prepare(Device::shared());
}

void Decoder::requireMatchingWeights(const DecoderWeights& weights) const
{
    if (!(weights.shape == decoderShape))
        throw ModelError {"the weights were loaded against a different decoder "
                          "shape than this decoder was built for"};
}

void Decoder::beginSequence(ComputePass& pass,
                            const Buffer& encoderOutput,
                            const DecoderWeights& weights,
                            int crossPositionCount)
{
    requireMatchingWeights(weights);

    if (crossPositionCount < 0 || crossPositionCount > decoderShape.crossPositions)
        throw ModelError {"a decoder built for "
                          + std::to_string(decoderShape.crossPositions)
                          + " encoder rows was asked to attend to "
                          + std::to_string(crossPositionCount)};

    const auto rows =
        crossPositionCount == 0 ? decoderShape.crossPositions : crossPositionCount;

    const auto encoderRows =
        Binding {BufferRange::of(encoderOutput),
                 Shape {decoderShape.crossPositions, decoderShape.width},
                 "encoder_rows"};

    net.begin(pass);
    recordSequenceStart(net, decoderShape, weights, caches, encoderRows, rows);
    net.end();

    decodedPositions = 0;
    activeCrossPositions = rows;
}

void Decoder::step(ComputePass& pass,
                   const BufferRange& tokens,
                   int tokenCount,
                   const DecoderWeights& weights,
                   const Buffer& logits)
{
    requireMatchingWeights(weights);

    if (tokenCount <= 0)
        throw ModelError {"a decoder step needs at least one token"};

    // Before a single dispatch: a step that ran off the end of the window would
    // bind a cache row outside its buffer, and a range past a buffer's end
    // binds nothing at all rather than failing.
    if (decodedPositions + tokenCount > decoderShape.maxPositions)
        throw ModelError {"this step would carry the sequence to "
                          + std::to_string(decodedPositions + tokenCount)
                          + " tokens, past the "
                          + std::to_string(decoderShape.maxPositions)
                          + " this decoder was built for"};

    const auto tokenSlots =
        Binding {tokens, Shape {decoderShape.maxPositions}, "tokens"};

    net.begin(pass);
    recordDecoderStep(
        net,
        decoderShape,
        weights,
        caches,
        tokenSlots,
        tokenCount,
        decodedPositions,
        Binding {BufferRange::of(*normalisedRows), {}, "hidden_states"},
        Binding {BufferRange::of(logits), {}, "logits"});
    net.end();

    decodedPositions += tokenCount;
}
} // namespace WSP
