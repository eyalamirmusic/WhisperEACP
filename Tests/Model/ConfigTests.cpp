#include "Common.h"

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;

namespace
{
// Overrides come first because Miro's parser keeps the first occurrence of a
// duplicate key, so a field named here is the one that wins.
std::string tinyConfigJson(std::string_view overrides = {})
{
    auto text = std::string {"{"};

    if (!overrides.empty())
        text += std::string {overrides} + ",";

    return text
           + R"("model_type":"whisper",)"
             R"("num_mel_bins":80,"d_model":384,)"
             R"("encoder_layers":4,"decoder_layers":4,)"
             R"("encoder_attention_heads":6,)"
             R"("decoder_attention_heads":6,)"
             R"("encoder_ffn_dim":1536,"decoder_ffn_dim":1536,)"
             R"("max_source_positions":1500,)"
             R"("max_target_positions":448,)"
             R"("vocab_size":51864})";
}

std::string configWithout(std::string_view field)
{
    const auto full = tinyConfigJson();
    const auto key = "\"" + std::string {field} + "\":";
    const auto start = full.find(key);
    const auto end = full.find(',', start);

    return full.substr(0, start) + full.substr(end + 1);
}

// A four-bin filterbank over an eight-point transform, which is five columns.
std::string filterbankJson(std::string_view rows)
{
    return R"({"sampling_rate":16000,"n_fft":8,"hop_length":2,)"
           R"("feature_size":4,"chunk_length":30,"n_samples":480000,)"
           R"("nb_max_frames":3000,"mel_filters":)"
           + std::string {rows} + "}";
}

const auto melMajorRows = std::string {"[[1,0,0,0,0],"
                                       "[0,1,0,0,0],"
                                       "[0,0,1,0,0],"
                                       "[0,0,0,1,0]]"};

const auto frequencyMajorRows = std::string {"[[1,0,0,0],"
                                             "[0,1,0,0],"
                                             "[0,0,1,0],"
                                             "[0,0,0,1],"
                                             "[0,0,0,0]]"};
} // namespace

auto tModelConfigShapes = test("Model/Config/shapes") = []
{
    const auto config = ModelConfig::fromJson(tinyConfigJson());

    check(config.melBins == 80);
    check(config.modelWidth == 384);
    check(config.encoderLayers == 4);
    check(config.decoderLayers == 4);
    check(config.encoderHeads == 6);
    check(config.decoderHeads == 6);
    check(config.encoderFeedForwardWidth == 1536);
    check(config.decoderFeedForwardWidth == 1536);
    check(config.maxSourcePositions == 1500);
    check(config.maxTargetPositions == 448);
    check(config.vocabularySize == 51864);

    check(config.encoderHeadWidth() == 64);
    check(config.decoderHeadWidth() == 64);
    check(config.encoderInputFrames() == 3000);
};

auto tModelConfigOptionalFields = test("Model/Config/optionalFields") = []
{
    const auto bare = ModelConfig::fromJson(tinyConfigJson());

    check(bare.activationFunction == "gelu");
    check(!bare.scaleEmbedding);
    check(bare.endOfSequenceToken == ModelConfig::noToken);
    check(bare.suppressedTokens.empty());
    check(bare.initiallySuppressedTokens.empty());

    const auto full = ModelConfig::fromJson(tinyConfigJson(
        R"("bos_token_id":50257,"eos_token_id":50256,)"
        R"("pad_token_id":50256,"decoder_start_token_id":50257,)"
        R"("scale_embedding":true,"activation_function":"relu",)"
        R"("suppress_tokens":[1,2,7],"begin_suppress_tokens":[220])"));

    check(full.beginningOfSequenceToken == 50257);
    check(full.endOfSequenceToken == 50256);
    check(full.padToken == 50256);
    check(full.decoderStartToken == 50257);
    check(full.scaleEmbedding);
    check(full.activationFunction == "relu");
    check(full.suppressedTokens.size() == 3);
    check(full.suppressedTokens[2] == 7);
    check(full.initiallySuppressedTokens.size() == 1);
    check(full.initiallySuppressedTokens[0] == 220);
};

auto tModelConfigRequiresShapeFields = test("Model/Config/requiresShapeFields") = []
{
    for (const auto* field: {"num_mel_bins",
                             "d_model",
                             "encoder_layers",
                             "decoder_layers",
                             "encoder_attention_heads",
                             "decoder_attention_heads",
                             "encoder_ffn_dim",
                             "decoder_ffn_dim",
                             "max_source_positions",
                             "max_target_positions"})
    {
        const auto without = configWithout(field);
        check(throwsModelError([&] { return ModelConfig::fromJson(without); }));
    }
};

auto tModelConfigRejectsBadShapes = test("Model/Config/rejectsBadShapes") = []
{
    check(throwsModelError([] { return ModelConfig::fromJson("not json"); }));
    check(throwsModelError([] { return ModelConfig::fromJson("[]"); }));

    check(throwsModelError(
        [] { return ModelConfig::fromJson(tinyConfigJson(R"("d_model":0)")); }));

    check(throwsModelError(
        []
        {
            return ModelConfig::fromJson(
                tinyConfigJson(R"("encoder_attention_heads":7)"));
        }));

    check(throwsModelError(
        [] { return ModelConfig::fromJson(tinyConfigJson(R"("d_model":"384")")); }));

    check(throwsModelError(
        [] { return ModelConfig::fromJson(tinyConfigJson(R"("d_model":384.5)")); }));
};

// A later encoder built for whisper and pointed at some other architecture's
// config would otherwise get shapes that parse and mean nothing.
auto tModelConfigRejectsAnotherArchitecture =
    test("Model/Config/rejectsAnotherArchitecture") = []
{
    check(throwsModelError(
        []
        {
            return ModelConfig::fromJson(tinyConfigJson(R"("model_type":"bart")"));
        }));
};

auto tModelConfigMissingFile = test("Model/Config/missingFile") = []
{ check(throwsModelError([] { return ModelConfig::fromFile("no/such.json"); })); };

auto tPreprocessorFields = test("Model/Preprocessor/fields") = []
{
    const auto config = PreprocessorConfig::fromJson(filterbankJson(melMajorRows));

    check(config.sampleRate == 16000);
    check(config.fftSize == 8);
    check(config.hopSize == 2);
    check(config.melBins == 4);
    check(config.chunkSeconds == 30);
    check(config.windowSamples == 480000);
    check(config.maxFrames == 3000);
    check(config.melFilterColumns() == 5);
    check(config.melFilters.size() == 20);

    for (auto bin = 0; bin < config.melBins; ++bin)
    {
        const auto row = config.melFilterRow(bin);
        check(row.size() == 5);

        for (auto column = 0; column < row.size(); ++column)
            check(row[column] == (column == bin ? 1.0f : 0.0f));
    }

    check(config.melFilterRow(-1).empty());
    check(config.melFilterRow(config.melBins).empty());
};

// HF has serialised this matrix both ways round across transformers versions,
// so a frequency-major file has to land in the same mel-major layout.
auto tPreprocessorTransposesFrequencyMajor =
    test("Model/Preprocessor/transposesFrequencyMajor") = []
{
    const auto melMajor = PreprocessorConfig::fromJson(filterbankJson(melMajorRows));
    const auto frequencyMajor =
        PreprocessorConfig::fromJson(filterbankJson(frequencyMajorRows));

    check(melMajor.melFilters == frequencyMajor.melFilters);
};

auto tPreprocessorRejectsBadFilterbank =
    test("Model/Preprocessor/rejectsBadFilterbank") = []
{
    const auto ragged = std::string {"[[1,0,0,0,0],[0,1,0,0],[0,0,1,0,0],"
                                     "[0,0,0,1,0]]"};
    const auto tooFewRows = std::string {"[[1,0,0,0,0],[0,1,0,0,0]]"};

    check(throwsModelError(
        [&] { return PreprocessorConfig::fromJson(filterbankJson(ragged)); }));

    check(throwsModelError(
        [&] { return PreprocessorConfig::fromJson(filterbankJson(tooFewRows)); }));

    check(throwsModelError(
        [&]
        { return PreprocessorConfig::fromJson(filterbankJson("[[1,\"x\"]]")); }));

    check(throwsModelError(
        [&] { return PreprocessorConfig::fromJson(filterbankJson("12")); }));
};

auto tPreprocessorRequiresFields = test("Model/Preprocessor/requiresFields") = []
{
    check(throwsModelError(
        []
        {
            return PreprocessorConfig::fromJson(
                R"({"sampling_rate":16000,"n_fft":8})");
        }));

    check(throwsModelError(
        []
        {
            return PreprocessorConfig::fromJson(
                R"({"sampling_rate":16000,"n_fft":7,"hop_length":2,)"
                R"("feature_size":4,"chunk_length":30,"n_samples":480000,)"
                R"("nb_max_frames":3000,"mel_filters":[[1]]})");
        }));

    check(throwsModelError(
        [] { return PreprocessorConfig::fromFile("no/such.json"); }));
};
