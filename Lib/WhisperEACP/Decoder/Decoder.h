#pragma once

#include <WhisperEACP/Decoder/DecoderWeights.h>

#include <optional>

namespace WSP
{
// HuggingFace's WhisperDecoder.forward with a KV cache, recorded into a command
// buffer:
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
// The shape of this is the Encoder's: prepare() compiles every kernel and sizes
// every intermediate once, and the two recording calls only record. Each
// dispatch is its own compute pass, because a pass boundary is what orders one
// dispatch's writes against the next one's reads — every stage below reads what
// the stage before it wrote, and the KV cache is read in the same command
// buffer that appended to it.
//
// One program of each kind serves every dispatch of that kind: the shapes are
// uniforms, so the layers, the two attentions and the two feed-forward widths
// are re-bindings of eight pipelines rather than pipelines of their own. The
// exception is Linear, which is held twice — the float-weight program and the
// packed-half one — so a weight that arrived fp16 is dispatched through the
// program that reads it as fp16.
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
    void beginSequence(eacp::GPU::CommandBuffer& commands,
                       const eacp::GPU::Buffer& encoderOutput,
                       const DecoderWeights& weights);

    // Appends tokenCount tokens to the sequence and writes their logits.
    //
    // tokens holds tokenCount uint32 ids — ids are indices, so they travel as
    // integers and Embed reads their bit pattern back — and logits receives
    // tokenCount * shape().logitElementCount() floats, [tokenCount,
    // vocabularySize] row-major. The hidden rows behind them stay readable
    // through hiddenStates() until the next step.
    //
    // A step that would carry the sequence past maxPositions is a ModelError
    // thrown before a single dispatch is recorded, since a half-recorded step
    // would leave the cache holding rows the position count does not know
    // about.
    void step(eacp::GPU::CommandBuffer& commands,
              const eacp::GPU::Buffer& tokens,
              int tokenCount,
              const DecoderWeights& weights,
              const eacp::GPU::Buffer& logits);

    // How many tokens the sequence holds, which is the position the next one
    // takes and the number of keys already cached.
    int position() const { return decodedPositions; }

    // The last step's rows after the final layer norm and before the tied
    // projection — [tokenCount, width] row-major, valid until the next step
    // overwrites it. Exposed so a comparison that disagrees bisects to before
    // or after the logits rather than only reporting a wrong vocabulary row.
    const eacp::GPU::Buffer& hiddenStates() const { return *normalisedRows; }

private:
    void requireMatchingWeights(const DecoderWeights& weights) const;

    void encodeLayer(eacp::GPU::CommandBuffer& commands,
                     const DecoderLayerWeights& weights,
                     int layerIndex,
                     int tokenCount);

    void encodeSelfAttention(eacp::GPU::CommandBuffer& commands,
                             const DecoderLayerWeights& weights,
                             int layerIndex,
                             int tokenCount);

    void encodeCrossAttention(eacp::GPU::CommandBuffer& commands,
                              const DecoderLayerWeights& weights,
                              int layerIndex,
                              int tokenCount);

    void encodeAttentionOverCache(eacp::GPU::CommandBuffer& commands,
                                  const eacp::GPU::Buffer& keys,
                                  const eacp::GPU::Buffer& values,
                                  const eacp::GPU::Buffer& scoreTarget,
                                  const eacp::GPU::Buffer& weightTarget,
                                  int queryCount,
                                  int keyCount,
                                  bool causal);

    void encodeGelu(eacp::GPU::CommandBuffer& commands,
                    const eacp::GPU::Buffer& input,
                    const eacp::GPU::Buffer& target,
                    int elementCount);

    void encodeSum(eacp::GPU::CommandBuffer& commands,
                   const eacp::GPU::Buffer& left,
                   const eacp::GPU::Buffer& right,
                   const eacp::GPU::Buffer& target,
                   int elementCount);

    void encodeLayerNorm(eacp::GPU::CommandBuffer& commands,
                         const eacp::GPU::Buffer& input,
                         const TensorBuffer& weight,
                         const TensorBuffer& bias,
                         const eacp::GPU::Buffer& target,
                         int rowCount);

    void encodeLinear(eacp::GPU::CommandBuffer& commands,
                      const eacp::GPU::Buffer& input,
                      const TensorBuffer& weight,
                      const eacp::GPU::Buffer& bias,
                      const eacp::GPU::BufferRange& target,
                      int innerCount,
                      int outputWidth,
                      int rowCount);

    // Where this step's keys and values are written into the layer's cache:
    // the row the sequence has reached, as a byte offset into the buffer.
    eacp::GPU::BufferRange cacheRowsAt(const eacp::GPU::Buffer& cache,
                                       int tokenCount) const;

    DecoderShape decoderShape;
    int decodedPositions = 0;

    Embed embedding;
    Gelu activation;
    Add sum;
    LayerNorm normalisation;
    Linear projection;
    HalfWeightLinear packedProjection;
    AttentionScores scores;
    Softmax softmax;
    AttentionApply attention;

    // The residual stream, which three sublayers add to in turn. Three buffers
    // rather than two because there are three additions and no dispatch here
    // reads and writes one buffer: a layer enters and leaves in hidden, so the
    // next layer reads what this one wrote without a swap the call site would
    // have to keep track of.
    std::optional<eacp::GPU::Buffer> hidden;
    std::optional<eacp::GPU::Buffer> afterSelfAttention;
    std::optional<eacp::GPU::Buffer> afterCrossAttention;

    std::optional<eacp::GPU::Buffer> normalised;
    std::optional<eacp::GPU::Buffer> queries;
    std::optional<eacp::GPU::Buffer> attended;
    std::optional<eacp::GPU::Buffer> attentionOutput;
    std::optional<eacp::GPU::Buffer> selfScores;
    std::optional<eacp::GPU::Buffer> selfWeights;
    std::optional<eacp::GPU::Buffer> crossScores;
    std::optional<eacp::GPU::Buffer> crossWeights;
    std::optional<eacp::GPU::Buffer> feedForward;
    std::optional<eacp::GPU::Buffer> activatedFeedForward;
    std::optional<eacp::GPU::Buffer> feedForwardOutput;
    std::optional<eacp::GPU::Buffer> normalisedRows;

    // One pair per layer. The self-attention pair grows a row per token and is
    // read from row zero to the position every step; the cross-attention pair
    // is written once per sequence and read whole by every step after it.
    Vector<eacp::GPU::Buffer> selfKeyCache;
    Vector<eacp::GPU::Buffer> selfValueCache;
    Vector<eacp::GPU::Buffer> crossKeys;
    Vector<eacp::GPU::Buffer> crossValues;

    // What k_proj binds where another projection binds its bias, and what the
    // tied logits projection binds where a proj_out would have had one. Two
    // buffers rather than one because Linear reads outputWidth of them and the
    // two projections have different output widths. Filled once, because
    // neither backend defines what a shader reading an unbound buffer gets and
    // a flag would only guard a read that must not happen at all.
    std::optional<eacp::GPU::Buffer> zeroBias;
    std::optional<eacp::GPU::Buffer> zeroLogitBias;
};
} // namespace WSP
