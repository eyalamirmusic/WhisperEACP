#include "ModelConfig.h"

#include "ModelIO.h"

namespace WSP
{
namespace
{
constexpr auto configName = "config.json";

void requirePositive(int value, std::string_view field)
{
    if (value <= 0)
        throw ModelError {"config.json field '" + std::string {field}
                          + "' must be positive"};
}

void requireDivides(int width, int heads, std::string_view side)
{
    if (width % heads != 0)
        throw ModelError {"config.json d_model does not divide into the "
                          + std::string {side} + " head count"};
}

void requireWhisper(const Miro::Json::Object& object)
{
    const auto type =
        ModelIO::stringFieldOr(object, "model_type", "whisper", configName);

    if (type != "whisper")
        throw ModelError {"config.json describes a '" + type
                          + "' model, not a whisper one"};
}
} // namespace

int ModelConfig::encoderHeadWidth() const
{
    return modelWidth / encoderHeads;
}

int ModelConfig::decoderHeadWidth() const
{
    return modelWidth / decoderHeads;
}

int ModelConfig::encoderInputFrames() const
{
    return maxSourcePositions * 2;
}

ModelConfig ModelConfig::fromFile(const std::filesystem::path& path)
{
    const auto bytes = ModelIO::readFileBytes(path);
    return fromJson(ModelIO::textOf(bytes));
}

ModelConfig ModelConfig::fromJson(std::string_view text)
{
    const auto parsed = ModelIO::parseObject(text, configName);
    const auto& object = parsed.asObject();

    requireWhisper(object);

    auto config = ModelConfig {};

    config.melBins = ModelIO::intField(object, "num_mel_bins", configName);
    config.modelWidth = ModelIO::intField(object, "d_model", configName);
    config.encoderLayers = ModelIO::intField(object, "encoder_layers", configName);
    config.decoderLayers = ModelIO::intField(object, "decoder_layers", configName);
    config.encoderHeads =
        ModelIO::intField(object, "encoder_attention_heads", configName);
    config.decoderHeads =
        ModelIO::intField(object, "decoder_attention_heads", configName);
    config.encoderFeedForwardWidth =
        ModelIO::intField(object, "encoder_ffn_dim", configName);
    config.decoderFeedForwardWidth =
        ModelIO::intField(object, "decoder_ffn_dim", configName);
    config.maxSourcePositions =
        ModelIO::intField(object, "max_source_positions", configName);
    config.maxTargetPositions =
        ModelIO::intField(object, "max_target_positions", configName);
    config.vocabularySize = ModelIO::intField(object, "vocab_size", configName);

    config.activationFunction =
        ModelIO::stringFieldOr(object, "activation_function", "gelu", configName);
    config.scaleEmbedding =
        ModelIO::boolFieldOr(object, "scale_embedding", false, configName);

    config.beginningOfSequenceToken =
        ModelIO::intFieldOr(object, "bos_token_id", noToken, configName);
    config.endOfSequenceToken =
        ModelIO::intFieldOr(object, "eos_token_id", noToken, configName);
    config.padToken =
        ModelIO::intFieldOr(object, "pad_token_id", noToken, configName);
    config.decoderStartToken =
        ModelIO::intFieldOr(object, "decoder_start_token_id", noToken, configName);

    config.suppressedTokens =
        ModelIO::intArrayFieldOr(object, "suppress_tokens", configName);
    config.initiallySuppressedTokens =
        ModelIO::intArrayFieldOr(object, "begin_suppress_tokens", configName);

    requirePositive(config.melBins, "num_mel_bins");
    requirePositive(config.modelWidth, "d_model");
    requirePositive(config.encoderLayers, "encoder_layers");
    requirePositive(config.decoderLayers, "decoder_layers");
    requirePositive(config.encoderHeads, "encoder_attention_heads");
    requirePositive(config.decoderHeads, "decoder_attention_heads");
    requirePositive(config.encoderFeedForwardWidth, "encoder_ffn_dim");
    requirePositive(config.decoderFeedForwardWidth, "decoder_ffn_dim");
    requirePositive(config.maxSourcePositions, "max_source_positions");
    requirePositive(config.maxTargetPositions, "max_target_positions");
    requirePositive(config.vocabularySize, "vocab_size");

    requireDivides(config.modelWidth, config.encoderHeads, "encoder");
    requireDivides(config.modelWidth, config.decoderHeads, "decoder");

    return config;
}
} // namespace WSP
