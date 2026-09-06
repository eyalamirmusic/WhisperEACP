#include "Decoder.h"

#include <cstdint>
#include <string>

namespace WSP
{
using eacp::GPU::Buffer;
using eacp::GPU::BufferRange;
using eacp::GPU::BufferUsage;
using eacp::GPU::CommandBuffer;
using eacp::GPU::Device;

namespace
{
constexpr int floatBytes(int elementCount)
{
    return elementCount * (int) sizeof(float);
}

Buffer allocate(Device& device, int elementCount)
{
    return device.makeBuffer(floatBytes(elementCount), BufferUsage::Storage);
}

Buffer allocateZeroed(Device& device, int elementCount)
{
    auto zeroes = Vector<float> {};
    zeroes.resize(elementCount);

    for (auto index = 0; index < elementCount; ++index)
        zeroes[index] = 0.f;

    return device.makeBuffer(
        zeroes.data(), floatBytes(elementCount), BufferUsage::Storage);
}

void allocatePerLayer(Vector<Buffer>& buffers,
                      Device& device,
                      int layers,
                      int elementCount)
{
    buffers.clear();
    buffers.reserve(layers);

    for (auto index = 0; index < layers; ++index)
        buffers.emplace_back(allocate(device, elementCount));
}

// The two Linear programs differ only in how they read the weight, so the bind
// is written once against whichever of them the caller picked. The target is a
// range rather than a buffer because one of its call sites writes into the
// middle of a KV cache.
template <WeightStorage weightStorage>
void dispatchLinear(LinearProgram<weightStorage>& program,
                    CommandBuffer& commands,
                    const Buffer& input,
                    const Buffer& weight,
                    const Buffer& bias,
                    const BufferRange& target,
                    int innerCount,
                    int outputWidth,
                    int rowCount)
{
    program.input = input;
    program.weights = weight;
    program.bias = bias;
    program.output = target;
    program.innerCount = (std::uint32_t) innerCount;
    program.outputWidth = (std::uint32_t) outputWidth;

    auto pass = commands.beginCompute();
    pass.dispatch(program, outputWidth, rowCount);
}
} // namespace

Decoder::Decoder(const DecoderShape& shapeToUse)
    : decoderShape(shapeToUse)
{
}

// Every intermediate is sized for the longest step a sequence can take rather
// than for the step in hand, so a prompt of several tokens and a single token
// after it are the same buffers at different dispatch heights.
//
// The two score pairs are what that costs, and they are the only allocations
// here worth a number: [heads, maxPositions, keys] floats each, which for
// tiny.en's 6 heads and 448 target positions is 4.8 MB apiece against the cache
// and 16 MB apiece against a full 1500-row encoder output — 42 MB for the four,
// beside 5.5 MB of self-attention cache and 18 MB of cross-attention keys and
// values over the four layers.
void Decoder::prepare(Device& device)
{
    embedding.prepare(device);
    activation.prepare(device);
    sum.prepare(device);
    normalisation.prepare(device);
    projection.prepare(device);
    packedProjection.prepare(device);
    scores.prepare(device);
    softmax.prepare(device);
    attention.prepare(device);

    const auto stepElements = decoderShape.stepElementCount();

    hidden.emplace(allocate(device, stepElements));
    afterSelfAttention.emplace(allocate(device, stepElements));
    afterCrossAttention.emplace(allocate(device, stepElements));
    normalised.emplace(allocate(device, stepElements));
    queries.emplace(allocate(device, stepElements));
    attended.emplace(allocate(device, stepElements));
    attentionOutput.emplace(allocate(device, stepElements));
    feedForwardOutput.emplace(allocate(device, stepElements));
    normalisedRows.emplace(allocate(device, stepElements));

    selfScores.emplace(allocate(device, decoderShape.selfScoreElementCount()));
    selfWeights.emplace(allocate(device, decoderShape.selfScoreElementCount()));
    crossScores.emplace(allocate(device, decoderShape.crossScoreElementCount()));
    crossWeights.emplace(allocate(device, decoderShape.crossScoreElementCount()));

    const auto feedForwardElements = decoderShape.stepFeedForwardElementCount();
    feedForward.emplace(allocate(device, feedForwardElements));
    activatedFeedForward.emplace(allocate(device, feedForwardElements));

    const auto layers = decoderShape.layers;
    allocatePerLayer(selfKeyCache, device, layers, decoderShape.cacheElementCount());
    allocatePerLayer(
        selfValueCache, device, layers, decoderShape.cacheElementCount());
    allocatePerLayer(crossKeys, device, layers, decoderShape.crossElementCount());
    allocatePerLayer(crossValues, device, layers, decoderShape.crossElementCount());

    zeroBias.emplace(allocateZeroed(device, decoderShape.width));
    zeroLogitBias.emplace(allocateZeroed(device, decoderShape.vocabularySize));

    decodedPositions = 0;
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

void Decoder::encodeGelu(CommandBuffer& commands,
                         const Buffer& input,
                         const Buffer& target,
                         int elementCount)
{
    activation.input = input;
    activation.output = target;

    auto pass = commands.beginCompute();
    pass.dispatch(activation, elementCount);
}

void Decoder::encodeSum(CommandBuffer& commands,
                        const Buffer& left,
                        const Buffer& right,
                        const Buffer& target,
                        int elementCount)
{
    sum.a = left;
    sum.b = right;
    sum.output = target;

    auto pass = commands.beginCompute();
    pass.dispatch(sum, elementCount);
}

void Decoder::encodeLayerNorm(CommandBuffer& commands,
                              const Buffer& input,
                              const TensorBuffer& weight,
                              const TensorBuffer& bias,
                              const Buffer& target,
                              int rowCount)
{
    normalisation.input = input;
    normalisation.weight = weight.buffer;
    normalisation.bias = bias.buffer;
    normalisation.output = target;
    normalisation.rowLength = (std::uint32_t) decoderShape.width;

    auto pass = commands.beginCompute();
    pass.dispatch(normalisation, rowCount);
}

// The one place a weight's storage decides anything: a tensor the loader left
// packed goes to the program that reads packed halves, and one it uploaded as
// floats to the program that subscripts floats. Neither can read the other's
// buffer, which is why the choice is made from the buffer rather than from a
// build-time switch.
void Decoder::encodeLinear(CommandBuffer& commands,
                           const Buffer& input,
                           const TensorBuffer& weight,
                           const Buffer& bias,
                           const BufferRange& target,
                           int innerCount,
                           int outputWidth,
                           int rowCount)
{
    if (weight.isPackedHalf())
        dispatchLinear(packedProjection,
                       commands,
                       input,
                       weight.buffer,
                       bias,
                       target,
                       innerCount,
                       outputWidth,
                       rowCount);
    else
        dispatchLinear(projection,
                       commands,
                       input,
                       weight.buffer,
                       bias,
                       target,
                       innerCount,
                       outputWidth,
                       rowCount);
}

// Appending to the cache is a bind, not a copy: the key and value projections
// write straight into the rows this step owns, at position * width floats into
// the layer's buffer, and the attention that follows reads the whole cache from
// row zero.
//
// This is what ComputePass taking a BufferRange bought — the gap plan.md
// recorded against Frame/ComputePass.h and eacp filled. Without it a step would
// have to project into a scratch row and copy it into place, which is a
// dispatch and a pass per layer per token whose only work is a memcpy the GPU
// has no kernel for, or else carry a row-offset uniform through Linear that
// every one of its other call sites would pay for in its inner loop.
//
// The range's byte count is not a bound — neither backend can make it one — so
// the row must be inside the buffer before it is bound, and a range at or past
// the end binds nothing at all, silently. step() is where that is checked, once
// for the whole step, before anything is recorded.
BufferRange Decoder::cacheRowsAt(const Buffer& cache, int tokenCount) const
{
    return {&cache,
            floatBytes(decodedPositions * decoderShape.width),
            floatBytes(tokenCount * decoderShape.width)};
}

// scores, softmax, apply — the three dispatches every attention is, over
// whichever keys and values the caller cached. The queries are already
// projected into *queries; what comes out is [queryCount, width] in *attended,
// the concatenation of the heads the output projection is a plain matmul over.
void Decoder::encodeAttentionOverCache(CommandBuffer& commands,
                                       const Buffer& keys,
                                       const Buffer& values,
                                       const Buffer& scoreTarget,
                                       const Buffer& weightTarget,
                                       int queryCount,
                                       int keyCount,
                                       bool causal)
{
    const auto width = decoderShape.width;
    const auto scoreRows = decoderShape.scoreRowCount(queryCount);

    scores.queries = *queries;
    scores.keys = keys;
    scores.scores = scoreTarget;
    scores.modelWidth = (std::uint32_t) width;
    scores.headWidth = (std::uint32_t) decoderShape.headWidth();
    scores.queryCount = (std::uint32_t) queryCount;
    scores.keyCount = (std::uint32_t) keyCount;
    scores.causal = causal ? 1u : 0u;
    scores.scale = decoderShape.attentionScale();

    {
        auto pass = commands.beginCompute();
        pass.dispatch(scores, keyCount, scoreRows);
    }

    softmax.input = scoreTarget;
    softmax.output = weightTarget;
    softmax.rowLength = (std::uint32_t) keyCount;

    {
        auto pass = commands.beginCompute();
        pass.dispatch(softmax, scoreRows);
    }

    attention.probabilities = weightTarget;
    attention.values = values;
    attention.output = *attended;
    attention.modelWidth = (std::uint32_t) width;
    attention.headWidth = (std::uint32_t) decoderShape.headWidth();
    attention.queryCount = (std::uint32_t) queryCount;
    attention.keyCount = (std::uint32_t) keyCount;

    {
        auto pass = commands.beginCompute();
        pass.dispatch(attention, width, queryCount);
    }
}

// Self-attention over the sequence so far: the queries are this step's, the
// keys and values are every token's, and the mask is what keeps token t out of
// its own future. Query i stands at absolute position keyCount - queryCount + i
// — which is exactly where this step's rows landed in the cache — so a prompt
// decoded in one call is masked the way the same tokens fed one at a time
// would be.
void Decoder::encodeSelfAttention(CommandBuffer& commands,
                                  const DecoderLayerWeights& weights,
                                  int layerIndex,
                                  int tokenCount)
{
    const auto width = decoderShape.width;
    const auto keyCount = decodedPositions + tokenCount;

    encodeLinear(commands,
                 *normalised,
                 weights.selfQueryWeight,
                 weights.selfQueryBias.buffer,
                 BufferRange::of(*queries),
                 width,
                 width,
                 tokenCount);

    encodeLinear(commands,
                 *normalised,
                 weights.selfKeyWeight,
                 *zeroBias,
                 cacheRowsAt(selfKeyCache[layerIndex], tokenCount),
                 width,
                 width,
                 tokenCount);

    encodeLinear(commands,
                 *normalised,
                 weights.selfValueWeight,
                 weights.selfValueBias.buffer,
                 cacheRowsAt(selfValueCache[layerIndex], tokenCount),
                 width,
                 width,
                 tokenCount);

    encodeAttentionOverCache(commands,
                             selfKeyCache[layerIndex],
                             selfValueCache[layerIndex],
                             *selfScores,
                             *selfWeights,
                             tokenCount,
                             keyCount,
                             true);
}

// Cross-attention against the encoder's rows: only the queries are projected
// here, since the keys and values were projected once when the sequence opened.
// Nothing is masked — every token attends to the whole utterance.
void Decoder::encodeCrossAttention(CommandBuffer& commands,
                                   const DecoderLayerWeights& weights,
                                   int layerIndex,
                                   int tokenCount)
{
    const auto width = decoderShape.width;

    encodeLinear(commands,
                 *normalised,
                 weights.crossQueryWeight,
                 weights.crossQueryBias.buffer,
                 BufferRange::of(*queries),
                 width,
                 width,
                 tokenCount);

    encodeAttentionOverCache(commands,
                             crossKeys[layerIndex],
                             crossValues[layerIndex],
                             *crossScores,
                             *crossWeights,
                             tokenCount,
                             decoderShape.crossPositions,
                             false);
}

// Pre-norm, which is what Whisper is: each of the three sublayers normalises
// what it reads and adds what it computed to what it was given, so the residual
// is the unnormalised input. The three buffers take turns holding the stream
// rather than one being read and written by the same dispatch, and the layer
// ends where it started so the next one needs to know nothing about the order.
void Decoder::encodeLayer(CommandBuffer& commands,
                          const DecoderLayerWeights& weights,
                          int layerIndex,
                          int tokenCount)
{
    const auto width = decoderShape.width;
    const auto elements = tokenCount * width;

    encodeLayerNorm(commands,
                    *hidden,
                    weights.selfAttentionNormWeight,
                    weights.selfAttentionNormBias,
                    *normalised,
                    tokenCount);

    encodeSelfAttention(commands, weights, layerIndex, tokenCount);

    encodeLinear(commands,
                 *attended,
                 weights.selfAttentionOutputWeight,
                 weights.selfAttentionOutputBias.buffer,
                 BufferRange::of(*attentionOutput),
                 width,
                 width,
                 tokenCount);

    encodeSum(commands, *hidden, *attentionOutput, *afterSelfAttention, elements);

    encodeLayerNorm(commands,
                    *afterSelfAttention,
                    weights.crossAttentionNormWeight,
                    weights.crossAttentionNormBias,
                    *normalised,
                    tokenCount);

    encodeCrossAttention(commands, weights, layerIndex, tokenCount);

    encodeLinear(commands,
                 *attended,
                 weights.crossAttentionOutputWeight,
                 weights.crossAttentionOutputBias.buffer,
                 BufferRange::of(*attentionOutput),
                 width,
                 width,
                 tokenCount);

    encodeSum(commands,
              *afterSelfAttention,
              *attentionOutput,
              *afterCrossAttention,
              elements);

    encodeLayerNorm(commands,
                    *afterCrossAttention,
                    weights.finalNormWeight,
                    weights.finalNormBias,
                    *normalised,
                    tokenCount);

    encodeLinear(commands,
                 *normalised,
                 weights.feedForwardWeight,
                 weights.feedForwardBias.buffer,
                 BufferRange::of(*feedForward),
                 width,
                 decoderShape.feedForwardWidth,
                 tokenCount);

    encodeGelu(commands,
               *feedForward,
               *activatedFeedForward,
               tokenCount * decoderShape.feedForwardWidth);

    encodeLinear(commands,
                 *activatedFeedForward,
                 weights.feedForwardOutputWeight,
                 weights.feedForwardOutputBias.buffer,
                 BufferRange::of(*feedForwardOutput),
                 decoderShape.feedForwardWidth,
                 width,
                 tokenCount);

    encodeSum(commands, *afterCrossAttention, *feedForwardOutput, *hidden, elements);
}

// The keys and values every step of this sequence will attend to, projected
// once out of the encoder's rows. k_proj has no bias in either attention — that
// is PyTorch's own bias=False on those two Linears — so it binds the zero
// buffer where v_proj binds the model's.
void Decoder::beginSequence(CommandBuffer& commands,
                            const Buffer& encoderOutput,
                            const DecoderWeights& weights)
{
    requireMatchingWeights(weights);

    const auto width = decoderShape.width;
    const auto rows = decoderShape.crossPositions;

    for (auto index = 0; index < decoderShape.layers; ++index)
    {
        const auto& layer = weights.layers[index];

        encodeLinear(commands,
                     encoderOutput,
                     layer.crossKeyWeight,
                     *zeroBias,
                     BufferRange::of(crossKeys[index]),
                     width,
                     width,
                     rows);

        encodeLinear(commands,
                     encoderOutput,
                     layer.crossValueWeight,
                     layer.crossValueBias.buffer,
                     BufferRange::of(crossValues[index]),
                     width,
                     width,
                     rows);
    }

    decodedPositions = 0;
}

void Decoder::step(CommandBuffer& commands,
                   const Buffer& tokens,
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

    embedding.tokens = tokens;
    embedding.tokenTable = weights.tokenEmbedding.buffer;
    embedding.positionTable = weights.positionalEmbedding.buffer;
    embedding.output = *hidden;
    embedding.width = (std::uint32_t) decoderShape.width;
    embedding.firstPosition = (std::uint32_t) decodedPositions;

    {
        auto pass = commands.beginCompute();
        pass.dispatch(embedding, decoderShape.width, tokenCount);
    }

    for (auto index = 0; index < decoderShape.layers; ++index)
        encodeLayer(commands, weights.layers[index], index, tokenCount);

    encodeLayerNorm(commands,
                    *hidden,
                    weights.finalNormWeight,
                    weights.finalNormBias,
                    *normalisedRows,
                    tokenCount);

    // The logits projection is embed_tokens itself — there is no proj_out in a
    // Whisper safetensors file, and no bias either, so the zero buffer this
    // binds is the vocabulary-wide one rather than the width-wide one k_proj
    // takes.
    encodeLinear(commands,
                 *normalisedRows,
                 weights.tokenEmbedding,
                 *zeroLogitBias,
                 BufferRange::of(logits),
                 decoderShape.width,
                 decoderShape.vocabularySize,
                 tokenCount);

    decodedPositions += tokenCount;
}
} // namespace WSP
