#include "DecoderShape.h"

#include <cmath>

namespace WSP
{
DecoderShape DecoderShape::fromConfig(const ModelConfig& config,
                                      int crossPositionCount)
{
    return {config.modelWidth,
            config.decoderHeads,
            config.decoderLayers,
            config.decoderFeedForwardWidth,
            config.vocabularySize,
            config.maxTargetPositions,
            crossPositionCount};
}

// D^-0.5, applied to the scores rather than folded into the query projection,
// which is where HuggingFace's WhisperAttention applies it. Both attentions in
// a block share it: cross-attention changes how many keys there are, not how
// wide a head is.
float DecoderShape::attentionScale() const
{
    return 1.f / std::sqrt((float) headWidth());
}
} // namespace WSP
