#pragma once

#include <WhisperEACP/Decoder/DecoderWeights.h>
#include <WhisperEACP/Net/KernelNet.h>

#include <optional>

namespace WSP
{
// One pair per layer. The self-attention pair grows a row per token and is read
// from row zero to the position every step; the cross-attention pair is written
// once per sequence and read whole by every step after it.
struct DecoderCaches
{
    Vector<Cache> selfKeys;
    Vector<Cache> selfValues;
    Vector<Cache> crossKeys;
    Vector<Cache> crossValues;
};

// HuggingFace's WhisperDecoder.forward with a KV cache:
//
//   h = embed_tokens[token] + embed_positions[position + t]
//   per layer: h += self_attn.out_proj(attention(self_attn_layer_norm(h)))
//                          causal, over every key cached so far and this step's
//              h += encoder_attn.out_proj(attention(
//                          encoder_attn_layer_norm(h), the encoder's rows))
//              h += fc2(gelu(fc1(final_layer_norm(h))))
//   h = layer_norm(h)
//   logits = h · embed_tokensᵀ
//
// A run is a beginSequence and then steps. beginSequence projects the
// cross-attention keys and values out of the encoder's output once for every
// layer — they do not depend on the tokens, so a sequence that projected them
// per step would pay for 1500 encoder rows on every token — and resets the
// position. A step appends however many tokens it is given to the sequence and
// writes their logits, so the prompt a run opens with is one call and each
// token after it is another.
//
// The two recordings are recordSequenceStart and recordDecoderStep, written
// once against Net. This class runs them on the kernel backend, the Encoder's
// way: prepare() compiles every kernel and sizes every intermediate once, and
// the two recording calls only record, into the one compute pass the caller
// opened, the KV cache read in the same pass that appended to it.
//
// Argmax is deliberately not here. Greedy sampling is the layer above: which
// tokens are suppressed at which step is generation config, and a decoder that
// sampled would have to hold it.
class Decoder
{
public:
    explicit Decoder(const DecoderShape& shapeToUse);

    void prepare(eacp::GPU::Device& device);
    void prepare();

    const DecoderShape& shape() const { return decoderShape; }

    // Opens a sequence over an encoder output of shape().crossElementCount()
    // floats, [crossPositions, width] row-major — what Encoder::encode writes.
    // The cross-attention keys and values are projected out of it for every
    // layer, and the position goes back to zero, so the next step is the
    // sequence's first.
    //
    // crossPositionCount is the encoder's own audio_ctx read back: an encoder
    // that ran over a prefix wrote that many rows, so the projections take
    // that many and every step of the sequence attends to that many. Zero, the
    // default, is the whole encoder output. Nothing is allocated here — the
    // caches are the shape's and a shorter sequence fills a prefix of them.
    void beginSequence(eacp::GPU::ComputePass& pass,
                       const eacp::GPU::Buffer& encoderOutput,
                       const DecoderWeights& weights,
                       int crossPositionCount = 0);

    // Appends tokenCount tokens to the sequence and writes their logits.
    //
    // tokens holds tokenCount uint32 ids, a range so that the slot an Argmax
    // wrote at the end of one step is what the next step embeds, with the
    // host nowhere in between; logits receives
    // tokenCount * shape().logitElementCount() floats, [tokenCount,
    // vocabularySize] row-major. The hidden rows behind them stay readable
    // through hiddenStates() until the next step.
    //
    // A step that would carry the sequence past maxPositions is a ModelError
    // thrown before a single dispatch is recorded, since a half-recorded step
    // would leave the cache holding rows the position count does not know
    // about.
    void step(eacp::GPU::ComputePass& pass,
              const eacp::GPU::BufferRange& tokens,
              int tokenCount,
              const DecoderWeights& weights,
              const eacp::GPU::Buffer& logits);

    void step(eacp::GPU::ComputePass& pass,
              const eacp::GPU::Buffer& tokens,
              int tokenCount,
              const DecoderWeights& weights,
              const eacp::GPU::Buffer& logits)
    {
        step(pass, eacp::GPU::BufferRange::of(tokens), tokenCount, weights, logits);
    }

    // How many tokens the sequence holds, which is the position the next one
    // takes and the number of keys already cached.
    int position() const { return decodedPositions; }

    // How many encoder rows this sequence attends to, which is what
    // beginSequence was given.
    int crossPositions() const { return activeCrossPositions; }

    // The last step's rows after the final layer norm and before the tied
    // projection — [tokenCount, width] row-major, valid until the next step
    // overwrites it. Exposed so a comparison that disagrees bisects to before
    // or after the logits rather than only reporting a wrong vocabulary row.
    const eacp::GPU::Buffer& hiddenStates() const { return *normalisedRows; }

private:
    void requireMatchingWeights(const DecoderWeights& weights) const;

    DecoderShape decoderShape;
    int decodedPositions = 0;
    int activeCrossPositions = 0;

    KernelNet net {KernelProfile::decoder};
    DecoderCaches caches;
    std::optional<eacp::GPU::Buffer> normalisedRows;
};

// Empties every cache and projects the cross-attention keys and values out of
// the first rows of the encoder's output.
void recordSequenceStart(Net& net,
                         const DecoderShape& shape,
                         const DecoderWeights& weights,
                         DecoderCaches& caches,
                         const Binding& encoderRows,
                         int rows);

// Appends tokenCount tokens read from tokens to the self-attention caches, and
// writes their final hidden rows into hiddenStates and their logits into logits.
void recordDecoderStep(Net& net,
                       const DecoderShape& shape,
                       const DecoderWeights& weights,
                       DecoderCaches& caches,
                       const Binding& tokens,
                       int tokenCount,
                       int firstPosition,
                       const Binding& hiddenStates,
                       const Binding& logits);
} // namespace WSP
