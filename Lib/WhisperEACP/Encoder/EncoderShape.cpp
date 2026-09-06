#include "EncoderShape.h"

#include <cmath>

namespace WSP
{
EncoderShape EncoderShape::fromConfig(const ModelConfig& config, int frameCount)
{
    return {config.melBins,
            config.modelWidth,
            config.encoderHeads,
            config.encoderLayers,
            config.encoderFeedForwardWidth,
            frameCount};
}

int EncoderShape::convolutionFrames() const
{
    return conv1dOutputLength(inputFrames,
                              convolutionKernelSize,
                              firstConvolutionStride,
                              convolutionPadding);
}

int EncoderShape::positions() const
{
    return conv1dOutputLength(convolutionFrames(),
                              convolutionKernelSize,
                              secondConvolutionStride,
                              convolutionPadding);
}

// D^-0.5, applied to the scores rather than folded into the query projection,
// which is where HuggingFace's WhisperAttention applies it.
float EncoderShape::attentionScale() const
{
    return 1.f / std::sqrt((float) headWidth());
}
} // namespace WSP
