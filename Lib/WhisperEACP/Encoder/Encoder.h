#pragma once

#include <WhisperEACP/Encoder/EncoderWeights.h>
#include <WhisperEACP/Net/KernelNet.h>

namespace WSP
{
// HuggingFace's WhisperEncoder.forward:
//
//   x = gelu(conv1(mel))          mel is [melBins, inputFrames], band-major,
//   x = gelu(conv2(x))            exactly as Mel/ writes it, turned frame-major
//                                 before the first convolution; conv2's stride
//                                 of two halves the frames
//   h = x + embed_positions
//   per layer: h += out_proj(attention(layerNorm(h)))
//              h += fc2(gelu(fc1(layerNorm(h))))
//   h = layerNorm(h)
//
// [positions, width] rows come out — 1500 x 384 for tiny.en over a full 30
// second window.
//
// The sequence is recordEncoder, written once against Net. This class runs it
// on the kernel backend, and its shape mirrors MelSpectrogram: prepare()
// compiles every kernel and sizes every intermediate once, and encode() only
// records, into the compute pass the caller opened. The caller decides whether
// that pass is serial or concurrent; the kernel net puts a barrier at every
// boundary where a dispatch reads what one before it wrote.
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
    //
    // positionCount is whisper.cpp's audio_ctx: the run computes that many
    // rows out of the first shape().framesForPositions(positionCount) mel
    // frames and writes them into the front of output, the rest of which it
    // neither reads nor touches. Zero, the default, is the whole window.
    // Nothing is allocated here — prepare() sized every intermediate for the
    // window and a shorter run dispatches over a prefix of each — and nothing
    // outside the audio is read: the convolutions see zero past the frames
    // they were given, exactly as the model's own padding does at the end of
    // the window.
    void encode(eacp::GPU::ComputePass& pass,
                const eacp::GPU::Buffer& mel,
                const EncoderWeights& weights,
                const eacp::GPU::Buffer& output,
                int positionCount = 0);

private:
    EncoderShape encoderShape;
    KernelNet net {KernelProfile::encoder};
};

// The encoder against any backend: mel is the [melBins, window frames] input of
// which the run reads the first inputFrames of each band, and rows receives the
// [positions, width] output.
void recordEncoder(Net& net,
                   const EncoderShape& shape,
                   const EncoderWeights& weights,
                   const Binding& mel,
                   const Binding& rows,
                   int inputFrames);
} // namespace WSP
