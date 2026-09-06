#pragma once

#include <WhisperEACP/Core/Core.h>
#include <WhisperEACP/Model/ModelError.h>

#include <filesystem>
#include <string>
#include <string_view>

namespace WSP
{
// config.json from the HF repo: the shapes the encoder and decoder are built
// against. tiny.en is 384 wide, 4 layers and 6 heads each side, a 1536 feed
// forward, 80 mel bins and a 51864-token vocabulary.
//
// The shape fields are required — a config missing one is an error rather than
// a default that quietly misbuilds the model. The generation fields are
// optional, because a config that only describes the weights is still a usable
// config for loading them.
struct ModelConfig
{
    int melBins = 0;
    int modelWidth = 0;
    int encoderLayers = 0;
    int decoderLayers = 0;
    int encoderHeads = 0;
    int decoderHeads = 0;
    int encoderFeedForwardWidth = 0;
    int decoderFeedForwardWidth = 0;
    int maxSourcePositions = 0;
    int maxTargetPositions = 0;
    int vocabularySize = 0;

    std::string activationFunction = "gelu";
    bool scaleEmbedding = false;

    static constexpr auto noToken = -1;

    int beginningOfSequenceToken = noToken;
    int endOfSequenceToken = noToken;
    int padToken = noToken;
    int decoderStartToken = noToken;

    Vector<int> suppressedTokens;
    Vector<int> initiallySuppressedTokens;

    int encoderHeadWidth() const;
    int decoderHeadWidth() const;

    // The encoder's second convolution has stride 2, so the transformer blocks
    // see half the mel frames. That is what max_source_positions counts, and
    // this is the frame count it implies.
    int encoderInputFrames() const;

    static ModelConfig fromFile(const std::filesystem::path& path);
    static ModelConfig fromJson(std::string_view text);
};
} // namespace WSP
