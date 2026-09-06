#pragma once

#include <WhisperEACP/Encoder/EncoderWeights.h>

#include <optional>

namespace WSP
{
// HuggingFace's WhisperEncoder.forward, recorded into one command buffer:
//
//   x = gelu(conv1(mel))          mel is [melBins, inputFrames], band-major,
//   x = gelu(conv2(x))            exactly as Mel/ writes it; conv2's stride of
//                                 two halves the frames, and its output strides
//                                 turn the result frame-major in the store
//                                 rather than in a pass of its own
//   h = x + embed_positions
//   per layer: h += out_proj(attention(layerNorm(h)))
//              h += fc2(gelu(fc1(layerNorm(h))))
//   h = layerNorm(h)
//
// [positions, width] rows come out — 1500 x 384 for tiny.en over a full 30
// second window.
//
// The shape of this mirrors MelSpectrogram: prepare() compiles every kernel and
// sizes every intermediate once, and encode() only records. Each dispatch is
// its own compute pass, because a pass boundary is what orders one dispatch's
// writes against the next one's reads — every stage below reads what the stage
// before it wrote.
//
// One program of each kind serves every dispatch of that kind: the shapes are
// uniforms, so the four layers, the two convolutions and the two feed-forward
// widths are re-bindings of eight pipelines rather than pipelines of their own.
// The exception is Linear, which is held twice — the float-weight program and
// the packed-half one — so a weight that arrived fp16 is dispatched through the
// program that reads it as fp16.
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
    void encode(eacp::GPU::CommandBuffer& commands,
                const eacp::GPU::Buffer& mel,
                const EncoderWeights& weights,
                const eacp::GPU::Buffer& output);

private:
    void encodeFrontEnd(eacp::GPU::CommandBuffer& commands,
                        const eacp::GPU::Buffer& mel,
                        const EncoderWeights& weights);

    void encodeLayer(eacp::GPU::CommandBuffer& commands,
                     const EncoderLayerWeights& weights);

    void encodeAttention(eacp::GPU::CommandBuffer& commands,
                         const EncoderLayerWeights& weights);

    void encodeGelu(eacp::GPU::CommandBuffer& commands,
                    const eacp::GPU::Buffer& input,
                    const eacp::GPU::Buffer& target,
                    int elementCount);

    void encodeSum(eacp::GPU::CommandBuffer& commands,
                   const eacp::GPU::Buffer& left,
                   const eacp::GPU::Buffer& right,
                   const eacp::GPU::Buffer& target);

    void encodeLayerNorm(eacp::GPU::CommandBuffer& commands,
                         const eacp::GPU::Buffer& input,
                         const TensorBuffer& weight,
                         const TensorBuffer& bias,
                         const eacp::GPU::Buffer& target);

    void encodeLinear(eacp::GPU::CommandBuffer& commands,
                      const eacp::GPU::Buffer& input,
                      const TensorBuffer& weight,
                      const eacp::GPU::Buffer& bias,
                      const eacp::GPU::Buffer& target,
                      int innerCount,
                      int outputWidth);

    EncoderShape encoderShape;

    Conv1d convolution;
    Gelu activation;
    Add sum;
    LayerNorm normalisation;
    Linear projection;
    HalfWeightLinear packedProjection;
    AttentionScores scores;
    Softmax softmax;
    AttentionApply attention;

    std::optional<eacp::GPU::Buffer> convolved;
    std::optional<eacp::GPU::Buffer> activated;
    std::optional<eacp::GPU::Buffer> projected;
    std::optional<eacp::GPU::Buffer> activatedProjection;
    std::optional<eacp::GPU::Buffer> hidden;
    std::optional<eacp::GPU::Buffer> residual;
    std::optional<eacp::GPU::Buffer> normalised;
    std::optional<eacp::GPU::Buffer> queries;
    std::optional<eacp::GPU::Buffer> keys;
    std::optional<eacp::GPU::Buffer> values;
    std::optional<eacp::GPU::Buffer> attentionScores;
    std::optional<eacp::GPU::Buffer> attentionWeights;
    std::optional<eacp::GPU::Buffer> attended;
    std::optional<eacp::GPU::Buffer> attentionOutput;
    std::optional<eacp::GPU::Buffer> feedForward;
    std::optional<eacp::GPU::Buffer> activatedFeedForward;
    std::optional<eacp::GPU::Buffer> feedForwardOutput;

    // What k_proj binds where another projection binds its bias. One buffer for
    // every layer, filled once, because neither backend defines what a shader
    // reading an unbound buffer gets and a flag would only guard a read that
    // must not happen at all.
    std::optional<eacp::GPU::Buffer> zeroBias;
};
} // namespace WSP
