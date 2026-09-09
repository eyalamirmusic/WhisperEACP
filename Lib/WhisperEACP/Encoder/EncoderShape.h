#pragma once

#include <WhisperEACP/Kernels/Kernels.h>
#include <WhisperEACP/Model/ModelConfig.h>

namespace WSP
{
// Every extent one encoder is built against. All of them reach the kernels as
// uniforms, so this is a description of a model rather than a set of numbers
// compiled in: fromConfig reads tiny.en's out of config.json, and a test writes
// its own small ones.
//
// inputFrames is a parameter rather than Whisper's 3000. The forward pass is
// the same dispatches over a shorter mel, which is what lets the real weights
// be checked against a scalar reference at a size the CPU can afford.
struct EncoderShape
{
    // Both of Whisper's convolutions, which differ only in stride. The second
    // one's is what halves the frame count before the transformer blocks.
    static constexpr auto convolutionKernelSize = 3;
    static constexpr auto convolutionPadding = 1;
    static constexpr auto firstConvolutionStride = 1;
    static constexpr auto secondConvolutionStride = 2;

    int melBins = 0;
    int width = 0;
    int heads = 0;
    int layers = 0;
    int feedForwardWidth = 0;
    int inputFrames = 0;

    static EncoderShape fromConfig(const ModelConfig& config, int frameCount);

    // Weights are loaded against one of these and dispatched against another,
    // and nothing about a GPU buffer says which — so the two are compared
    // outright before the first dispatch.
    friend bool operator==(const EncoderShape&, const EncoderShape&) = default;

    int headWidth() const { return width / heads; }

    // conv1 keeps the frame count (stride 1, and the padding covers the
    // kernel); conv2 halves it. Both go through the rule in Conv1d.h rather
    // than restating it, so a buffer sized here and a dispatch sized there
    // cannot disagree. The overloads take the mel frames a run over a prefix
    // of the window reads — whisper.cpp's audio_ctx — where the no-argument
    // pair takes the shape's own.
    int convolutionFrames() const { return convolutionFrames(inputFrames); }
    int positions() const { return positions(inputFrames); }
    int convolutionFrames(int frameCount) const;
    int positions(int frameCount) const;

    // The inverse, which conv2's stride of two makes exact: positionCount rows
    // come out of the first 2 * positionCount frames, and a count past the
    // window is the window.
    int framesForPositions(int positionCount) const;

    int melElementCount() const { return melBins * inputFrames; }
    int convolutionElementCount() const { return width * convolutionFrames(); }
    int elementCount() const { return positions() * width; }
    int feedForwardElementCount() const { return positions() * feedForwardWidth; }

    // [heads, positions, positions] row-major, which is heads * positions rows
    // of positions — the row a softmax is over, whoever applies it.
    int scoreElementCount() const { return heads * positions() * positions(); }
    int scoreRowCount() const { return heads * positions(); }

    float attentionScale() const;
};
} // namespace WSP
