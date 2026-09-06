#include "Common.h"

#include <WhisperEACP/Audio/Audio.h>

#include <eacp/GPU/GPU.h>

// The real openai/whisper-tiny.en files, which are a download and never a
// commit: configure with -DWHISPER_EACP_FETCH_MODEL=ON, or point
// WHISPER_MODEL_DIR at a checkout that already has them. Every test here
// returns early when the file it needs is absent, the same shape as a GPU test
// returning early on an invalid device.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;

namespace
{
constexpr auto configFile = "config.json";
constexpr auto preprocessorFile = "preprocessor_config.json";
constexpr auto weightsFile = "model.safetensors";
} // namespace

auto tTinyEnConfig = test("Model/TinyEn/config") = []
{
    if (!hasModelFile(configFile))
        return;

    const auto config = ModelConfig::fromFile(modelFile(configFile));

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
    check(config.activationFunction == "gelu");
    check(!config.scaleEmbedding);
    check(config.decoderStartToken == 50257);
    check(config.endOfSequenceToken == 50256);
};

// config.json describes the same window Audio/Format.h is compiled against, so
// the two have to agree: max_source_positions is the frame count after the
// encoder's stride-2 convolution, and num_mel_bins is what the filterbank
// produces.
auto tTinyEnConfigAgreesWithAudioFormat =
    test("Model/TinyEn/configAgreesWithAudioFormat") = []
{
    if (!hasModelFile(configFile))
        return;

    const auto config = ModelConfig::fromFile(modelFile(configFile));

    check(config.maxSourcePositions == encoderPositions);
    check(config.encoderInputFrames() == windowFrames);
};

// CLAUDE.md's claim about this file, checked against the file: the mel
// filterbank really is in it, already built, 80 x 201 for tiny.en. Nothing in
// this project reconstructs librosa's filterbank.
auto tTinyEnFilterbank = test("Model/TinyEn/filterbank") = []
{
    if (!hasModelFile(preprocessorFile))
        return;

    const auto config = PreprocessorConfig::fromFile(modelFile(preprocessorFile));

    check(config.melBins == 80);
    check(config.melFilterColumns() == 201);
    check(config.melFilters.size() == 80 * 201);

    // Triangular filters climbing in frequency: every bin has weight somewhere,
    // and a later bin's peak is never below an earlier one's.
    auto previousPeak = -1;

    for (auto bin = 0; bin < config.melBins; ++bin)
    {
        const auto row = config.melFilterRow(bin);
        auto peak = 0;

        for (auto column = 1; column < row.size(); ++column)
            if (row[column] > row[peak])
                peak = column;

        check(row[peak] > 0.0f);
        check(peak >= previousPeak);
        previousPeak = peak;
    }
};

// The other half of CLAUDE.md's claim: preprocessor_config.json restates the
// audio contract, and it agrees with Audio/Format.h value for value. Those
// constants stay compiled in; this is what pins them to the model's.
auto tTinyEnPreprocessorAgreesWithAudioFormat =
    test("Model/TinyEn/preprocessorAgreesWithAudioFormat") = []
{
    if (!hasModelFile(preprocessorFile))
        return;

    const auto config = PreprocessorConfig::fromFile(modelFile(preprocessorFile));

    check(config.sampleRate == sampleRate);
    check(config.fftSize == fftSize);
    check(config.hopSize == hopSize);
    check(config.chunkSeconds == windowSeconds);
    check(config.windowSamples == windowSamples);
    check(config.maxFrames == windowFrames);
    check(config.melFilterColumns() == fftSize / 2 + 1);
};

auto tTinyEnFilterbankToGpuBuffer = test("Model/TinyEn/filterbankToGpuBuffer") = []
{
    if (!hasModelFile(preprocessorFile) || !eacp::GPU::Device::shared().isValid())
        return;

    const auto config = PreprocessorConfig::fromFile(modelFile(preprocessorFile));
    const auto buffer = config.makeMelFilterBuffer();

    check(buffer.isValid());
    check(buffer.size() == 80 * 201 * static_cast<int>(sizeof(float)));
};

// tiny.en ships float32, not float16 — 167 tensors and a 151 MB blob. The
// loader widens F16 and BF16 anyway, because the larger repos ship those.
auto tTinyEnWeightsHeader = test("Model/TinyEn/weightsHeader") = []
{
    if (!hasModelFile(weightsFile))
        return;

    const auto weights = SafeTensors::fromFile(modelFile(weightsFile));

    check(weights.tensors().size() == 167);
    check(weights.metadata().at("format") == "pt");

    for (const auto& tensor: weights.tensors())
        check(tensor.type == TensorType::F32);

    const auto& firstConvolution = weights.info("model.encoder.conv1.weight");
    check(firstConvolution.rank() == 3);
    check(firstConvolution.dimension(0) == 384);
    check(firstConvolution.dimension(1) == 80);
    check(firstConvolution.dimension(2) == 3);

    const auto& embedding = weights.info("model.decoder.embed_tokens.weight");
    check(embedding.dimension(0) == 51864);
    check(embedding.dimension(1) == 384);

    check(weights.contains("model.encoder.layer_norm.weight"));
    check(weights.contains("model.decoder.layers.3.fc2.bias"));
    check(!weights.contains("model.encoder.layers.4.fc1.weight"));
};

// The shapes the header names are the shapes config.json describes, which is
// the check that says the two downloads belong to the same model.
auto tTinyEnWeightsMatchConfig = test("Model/TinyEn/weightsMatchConfig") = []
{
    if (!hasModelFile(weightsFile) || !hasModelFile(configFile))
        return;

    const auto config = ModelConfig::fromFile(modelFile(configFile));
    const auto weights = SafeTensors::fromFile(modelFile(weightsFile));

    check(weights.info("model.encoder.conv1.weight").dimension(0)
          == config.modelWidth);
    check(weights.info("model.encoder.conv1.weight").dimension(1) == config.melBins);
    check(weights.info("model.encoder.embed_positions.weight").dimension(0)
          == config.maxSourcePositions);
    check(weights.info("model.decoder.embed_positions.weight").dimension(0)
          == config.maxTargetPositions);
    check(weights.info("model.decoder.embed_tokens.weight").dimension(0)
          == config.vocabularySize);

    for (auto layer = 0; layer < config.encoderLayers; ++layer)
    {
        const auto prefix = "model.encoder.layers." + std::to_string(layer) + ".";
        check(weights.info(prefix + "fc1.weight").dimension(0)
              == config.encoderFeedForwardWidth);
        check(weights.info(prefix + "self_attn.q_proj.weight").dimension(0)
              == config.modelWidth);
    }
};

auto tTinyEnWeightToGpuBuffer = test("Model/TinyEn/weightToGpuBuffer") = []
{
    if (!hasModelFile(weightsFile) || !eacp::GPU::Device::shared().isValid())
        return;

    const auto weights = SafeTensors::fromFile(modelFile(weightsFile));
    const auto name = std::string {"model.encoder.layer_norm.weight"};

    const auto values = weights.readFloats(name);
    const auto weight = weights.makeBuffer(name);

    // tiny.en is F32 throughout, so this is the unpacked path: the buffer is
    // the blob itself and holds one float per element.
    check(weight.storage == TensorType::F32);
    check(!weight.isPackedHalf());
    check(weight.buffer.isValid());
    check(weight.buffer.size() == values.size() * static_cast<int>(sizeof(float)));

    auto readBack = Vector<float> {};
    readBack.resize(values.size());
    weight.buffer.read(readBack.data(), weight.buffer.size());

    check(readBack == values);
};
