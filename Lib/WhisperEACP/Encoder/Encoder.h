#pragma once

#include <WhisperEACP/Encoder/EncoderWeights.h>

#include <optional>

namespace WSP
{
// HuggingFace's WhisperEncoder.forward, recorded into one command buffer:
//
//   x = gelu(conv1(mel))          mel is [melBins, inputFrames], band-major,
//   x = gelu(conv2(x))            exactly as Mel/ writes it; each convolution
//                                 is an unfold and a tiled product with the
//                                 GELU on its store, frame-major from the
//                                 first, and conv2's stride of two halves the
//                                 frames
//   h = x + embed_positions
//   per layer: h += out_proj(attention(layerNorm(h)))
//              h += fc2(gelu(fc1(layerNorm(h))))
//   h = layerNorm(h)
//
// [positions, width] rows come out — 1500 x 384 for tiny.en over a full 30
// second window.
//
// The shape of this mirrors MelSpectrogram: prepare() compiles every kernel and
// sizes every intermediate once, and encode() only records, into the compute
// pass the caller opened. The recording spells its own ordering out: a
// pass.barrier() sits at every boundary where a stage reads what the stage
// before it wrote, which is a no-op in a serial pass and the whole ordering in
// a concurrent one. So the caller decides which the pass is, and the only
// dispatches without a barrier between them are the three a layer's attention
// opens with — see encodeAttention.
//
// One program of each kind serves every dispatch of that kind: the shapes are
// uniforms, so the four layers, the two convolutions and the two feed-forward
// widths are re-bindings of a few pipelines rather than pipelines of their
// own. The products are all the tiled kernel: the six projections and the
// attention scores read their second operand along k, so they share one
// program — held twice, the float-weight form and the packed-half one, so a
// weight that arrived fp16 is dispatched through the program that reads it as
// fp16 — and the attention apply reads its values along n through the other.
class Encoder
{
public:
    explicit Encoder(const EncoderShape& shapeToUse);

    void prepare(eacp::GPU::Device& device);
    void prepare();

    const EncoderShape& shape() const { return encoderShape; }

    // mel holds shape().melElementCount() floats band-major, and output
    // receives shape().elementCount() of them, [positions, width] row-major.
    // The weights must have been loaded against the same shape; a mismatch is a
    // ModelError rather than a dispatch at the wrong stride.
    void encode(eacp::GPU::ComputePass& pass,
                const eacp::GPU::Buffer& mel,
                const EncoderWeights& weights,
                const eacp::GPU::Buffer& output);

private:
    void encodeFrontEnd(eacp::GPU::ComputePass& pass,
                        const eacp::GPU::Buffer& mel,
                        const EncoderWeights& weights);

    void encodeLayer(eacp::GPU::ComputePass& pass,
                     const EncoderLayerWeights& weights);

    void encodeAttention(eacp::GPU::ComputePass& pass,
                         const EncoderLayerWeights& weights);

    void encodeSum(eacp::GPU::ComputePass& pass,
                   const eacp::GPU::Buffer& stream,
                   const eacp::GPU::Buffer& addend);

    void encodeLayerNorm(eacp::GPU::ComputePass& pass,
                         const eacp::GPU::Buffer& input,
                         const TensorBuffer& weight,
                         const TensorBuffer& bias,
                         const eacp::GPU::Buffer& target);

    // gelu applies the activation on the store, and residual adds the result
    // into what the target holds: the stages either side of a projection,
    // folded into it rather than dispatched on their own.
    void encodeLinear(eacp::GPU::ComputePass& pass,
                      const eacp::GPU::Buffer& input,
                      const TensorBuffer& weight,
                      const eacp::GPU::Buffer& bias,
                      const eacp::GPU::Buffer& target,
                      int innerCount,
                      int outputWidth,
                      int rowCount,
                      bool gelu = false,
                      bool residual = false);

    // A convolution's windows gathered into rows, over an input addressed by
    // its two strides, so the convolution is a linear over them.
    void encodeUnfold(eacp::GPU::ComputePass& pass,
                      const eacp::GPU::Buffer& input,
                      int inputChannelCount,
                      int inputLength,
                      int stride,
                      int inputChannelStride,
                      int inputFrameStride,
                      int outputLength);

    EncoderShape encoderShape;

    Unfold unfold;
    Add sum;

    // The many-row group: every layer norm here is 1500 rows at once.
    LayerNorm normalisation {LayerNorm::manyRowLanes};
    LinearProduct projection;
    HalfWeightLinearProduct packedProjection;
    Softmax softmax;

    // The scores are the same product shape as a projection and were dispatched
    // through the same program, but the two roles pick their kernel separately
    // — a [1500, 1500] product over a head's 64 columns is not the shape a
    // projection is — so each holds its own.
    AttentionScoresProduct scores;
    AttentionApplyProduct attention;

    // The residual stream is hidden, added to in place by every sublayer's
    // last projection; the GELUs are on the stores that feed them and the
    // softmax rewrites what it reads, so no stage here has an output of its
    // own that a second buffer would only copy. columns is the unfolded input
    // of whichever convolution is running, sized for the larger.
    std::optional<eacp::GPU::Buffer> columns;
    std::optional<eacp::GPU::Buffer> convolved;
    std::optional<eacp::GPU::Buffer> hidden;
    std::optional<eacp::GPU::Buffer> normalised;
    std::optional<eacp::GPU::Buffer> queries;
    std::optional<eacp::GPU::Buffer> keys;
    std::optional<eacp::GPU::Buffer> values;
    std::optional<eacp::GPU::Buffer> attentionScores;
    std::optional<eacp::GPU::Buffer> attended;
    std::optional<eacp::GPU::Buffer> feedForward;

    // What k_proj binds where another projection binds its bias, and what the
    // two attention products bind, having none. One buffer, as long as the
    // widest product's columns, filled once, because neither backend defines
    // what a shader reading an unbound buffer gets and a flag would only guard
    // a read that must not happen at all.
    std::optional<eacp::GPU::Buffer> zeroBias;
};
} // namespace WSP
