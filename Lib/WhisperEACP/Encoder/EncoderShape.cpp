#include "EncoderShape.h"

#include <algorithm>
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

int EncoderShape::convolutionFrames(int frameCount) const
{
    return conv1dOutputLength(frameCount,
                              convolutionKernelSize,
                              firstConvolutionStride,
                              convolutionPadding);
}

int EncoderShape::positions(int frameCount) const
{
    return conv1dOutputLength(convolutionFrames(frameCount),
                              convolutionKernelSize,
                              secondConvolutionStride,
                              convolutionPadding);
}

int EncoderShape::framesForPositions(int positionCount) const
{
    return std::min(positionCount * secondConvolutionStride, inputFrames);
}

// D^-0.5, applied to the scores rather than folded into the query projection,
// which is where HuggingFace's WhisperAttention applies it.
float EncoderShape::attentionScale() const
{
    return 1.f / std::sqrt((float) headWidth());
}
} // namespace WSP
