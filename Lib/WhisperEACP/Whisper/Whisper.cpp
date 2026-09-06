#include "Whisper.h"

#include <chrono>
#include <string>

namespace WSP
{
using eacp::GPU::Buffer;
using eacp::GPU::BufferRange;
using eacp::GPU::BufferUsage;
using eacp::GPU::CommandBuffer;
using eacp::GPU::Device;

namespace
{
constexpr auto configFile = "config.json";
constexpr auto preprocessorFile = "preprocessor_config.json";
constexpr auto weightsFileName = "model.safetensors";
constexpr auto tokenizerFile = "tokenizer.json";

constexpr int floatBytes(int elementCount)
{
    return elementCount * (int) sizeof(float);
}

std::filesystem::path requireFile(const std::filesystem::path& directory,
                                  std::string_view name)
{
    auto path = directory / name;
    auto error = std::error_code {};

    if (!std::filesystem::is_regular_file(path, error))
        throw ModelError {"the model directory " + directory.string() + " has no "
                          + std::string {name}};

    return path;
}

double commitSeconds(CommandBuffer& commands)
{
    const auto start = std::chrono::steady_clock::now();
    commands.commit();
    const auto elapsed = std::chrono::steady_clock::now() - start;

    return std::chrono::duration<double>(elapsed).count();
}
} // namespace

void Whisper::load(const std::filesystem::path& modelDirectory)
{
    modelConfig = ModelConfig::fromFile(requireFile(modelDirectory, configFile));
    preprocessor =
        PreprocessorConfig::fromFile(requireFile(modelDirectory, preprocessorFile));
    vocabulary.emplace(
        Tokenizer::fromFile(requireFile(modelDirectory, tokenizerFile)));
    weightsFile.emplace(
        SafeTensors::fromFile(requireFile(modelDirectory, weightsFileName)));

    buildPrompt();
    buildSuppressionMasks();

    maximumTokenCount = modelConfig.maxTargetPositions - promptTokens.size();
}

const Tokenizer& Whisper::tokenizer() const
{
    requireLoaded();
    return *vocabulary;
}

void Whisper::requireLoaded() const
{
    if (!isLoaded())
        throw ModelError {"this Whisper has not loaded a model yet"};
}

void Whisper::requirePrepared() const
{
    if (!isPrepared())
        throw ModelError {"this Whisper has not been prepared on a device yet"};
}

// The front-end's shapes are compiled in — MelShape is written against
// Audio/Format.h, which is the contract Whisper was trained under rather than a
// choice — so a repo whose own preprocessing disagrees is refused here rather
// than dispatched at a stride that is off by a factor. tiny.en and base.en pass;
// large-v3, with 128 mel bins, is exactly what this catches.
void Whisper::requireWhisperShapes() const
{
    if (modelConfig.melBins != melBands)
        throw ModelError {"this model wants " + std::to_string(modelConfig.melBins)
                          + " mel bins and the front-end is built for "
                          + std::to_string(melBands)};

    if (modelConfig.encoderInputFrames() != windowFrames)
        throw ModelError {"this model wants "
                          + std::to_string(modelConfig.encoderInputFrames())
                          + " mel frames and the front-end produces "
                          + std::to_string(windowFrames)};

    if (preprocessor.melBins != melBands
        || preprocessor.melFilterColumns() != spectrumBins)
        throw ModelError {"preprocessor_config.json publishes a "
                          + std::to_string(preprocessor.melBins) + " x "
                          + std::to_string(preprocessor.melFilterColumns())
                          + " filterbank and the front-end binds "
                          + std::to_string(melBands) + " x "
                          + std::to_string(spectrumBins)};
}

// The English-only prompt. `<|startoftranscript|>` is config.json's
// decoder_start_token_id where the file carries one, because that is the field
// HF's generate reads; `<|notimestamps|>` has no field of its own — it is the
// second half of the one forced_decoder_ids entry, [[1, 50362]] — so it comes
// from the vocabulary, which is where its id is actually written down.
void Whisper::buildPrompt()
{
    const auto& specials = vocabulary->specials();

    const auto start = modelConfig.decoderStartToken != ModelConfig::noToken
                           ? modelConfig.decoderStartToken
                           : specials.startOfTranscript;

    if (start == invalidTokenId)
        throw ModelError {"neither config.json nor tokenizer.json names "
                          "<|startoftranscript|>"};

    if (specials.noTimestamps == invalidTokenId)
        throw ModelError {"tokenizer.json names no <|notimestamps|> token"};

    promptTokens.clear();
    promptTokens.add(start);
    promptTokens.add(specials.noTimestamps);
}

void Whisper::buildSuppressionMasks()
{
    const auto vocabularySize = modelConfig.vocabularySize;

    const auto mark = [vocabularySize](Vector<float>& mask, const Vector<int>& ids)
    {
        for (auto id: ids)
        {
            if (id < 0 || id >= vocabularySize)
                throw ModelError {"config.json suppresses token "
                                  + std::to_string(id)
                                  + ", which is outside the vocabulary"};

            mask[id] = 1.0f;
        }
    };

    laterStepSuppression.clear();
    laterStepSuppression.resize(vocabularySize);

    for (auto index = 0; index < vocabularySize; ++index)
        laterStepSuppression[index] = 0.0f;

    mark(laterStepSuppression, modelConfig.suppressedTokens);

    firstStepSuppression = laterStepSuppression;
    mark(firstStepSuppression, modelConfig.initiallySuppressedTokens);
}

void Whisper::setMaximumTokens(int count)
{
    if (count <= 0)
        throw ModelError {"a transcript has to be allowed at least one token"};

    maximumTokenCount = count;
}

void Whisper::prepare(Device& device)
{
    requireLoaded();
    requireWhisperShapes();

    gpu = &device;

    encoder.emplace(
        EncoderShape::fromConfig(modelConfig, modelConfig.encoderInputFrames()));
    decoder.emplace(
        DecoderShape::fromConfig(modelConfig, encoder->shape().positions()));

    frontEnd.prepare(device);
    encoder->prepare(device);
    decoder->prepare(device);
    selection.prepare(device);

    encoderWeights.emplace(*weightsFile, encoder->shape());
    decoderWeights.emplace(*weightsFile, decoder->shape());

    filterBank.emplace(preprocessor.makeMelFilterBuffer());

    paddedSamples.resize(windowSamples);
    stepTokens.resize(promptTokens.size());

    sampleBuffer.emplace(
        device.makeBuffer(floatBytes(windowSamples), BufferUsage::Storage));
    melBuffer.emplace(device.makeBuffer(
        floatBytes(frontEnd.shape().melElementCount()), BufferUsage::Storage));
    encodedBuffer.emplace(device.makeBuffer(
        floatBytes(encoder->shape().elementCount()), BufferUsage::Storage));

    tokenBuffer.emplace(device.makeBuffer(
        (int) sizeof(std::uint32_t) * promptTokens.size(), BufferUsage::Storage));

    // Sized for the prompt rather than for the window: the prompt is the only
    // step with more than one row, and a buffer covering all 448 positions of a
    // 51864-wide vocabulary would be 92 MB for two rows of use.
    logitBuffer.emplace(device.makeBuffer(
        floatBytes(promptTokens.size() * decoder->shape().logitElementCount()),
        BufferUsage::Storage));

    firstStepMaskBuffer.emplace(
        device.makeBuffer(firstStepSuppression.data(),
                          floatBytes(firstStepSuppression.size()),
                          BufferUsage::Storage));
    laterStepMaskBuffer.emplace(
        device.makeBuffer(laterStepSuppression.data(),
                          floatBytes(laterStepSuppression.size()),
                          BufferUsage::Storage));

    sampledIndex.emplace(
        device.makeBuffer((int) sizeof(std::uint32_t), BufferUsage::Storage));
}

void Whisper::prepare()
{
    prepare(Device::shared());
}

// Zero-filled to the window, which is HF's own padding: WhisperFeatureExtractor
// pads to n_samples with zeros rather than reflecting, and the encoder's
// positional embedding is written against all 1500 positions whatever the
// utterance was.
void Whisper::uploadSamples(Span<const float> samples)
{
    for (auto index = 0; index < windowSamples; ++index)
        paddedSamples[index] = index < samples.size() ? samples[index] : 0.0f;

    sampleBuffer->update(paddedSamples.data(), floatBytes(windowSamples));
}

// Ids are indices, so they travel as unsigned integers and Embed reads their bit
// pattern back through asUInt — eacp has no integer input buffer, which plan.md
// records against Codegen/ShaderValue.h:582.
void Whisper::uploadTokens(Span<const TokenId> tokens)
{
    for (auto index = 0; index < tokens.size(); ++index)
        stepTokens[index] = (std::uint32_t) tokens[index];

    tokenBuffer->update(stepTokens.data(),
                        (int) sizeof(std::uint32_t) * tokens.size());
}

double Whisper::encodeAudio()
{
    auto commands = gpu->makeCommandBuffer();

    frontEnd.encode(commands, *sampleBuffer, *filterBank, *melBuffer);
    encoder->encode(commands, *melBuffer, *encoderWeights, *encodedBuffer);

    return commitSeconds(commands);
}

// Greedy sampling over the last row of the step's logits, which is the
// distribution over the token that follows everything decoded so far. The row is
// bound as a range rather than the whole buffer — the same ranged bind the KV
// cache needed — so a prompt step's earlier rows are neither scanned nor copied.
void Whisper::encodeSampling(CommandBuffer& commands,
                             int rowCount,
                             const Buffer& mask)
{
    const auto rowBytes = floatBytes(decoder->shape().logitElementCount());

    selection.logits =
        BufferRange {&*logitBuffer, (rowCount - 1) * rowBytes, rowBytes};
    selection.mask = mask;
    selection.indices = *sampledIndex;
    selection.rowLength = (std::uint32_t) decoder->shape().logitElementCount();

    auto pass = commands.beginCompute();
    pass.dispatch(selection, 1);
}

TokenId Whisper::sampledToken() const
{
    auto index = std::uint32_t {};
    sampledIndex->read(&index, (int) sizeof(index));

    return (TokenId) index;
}

TokenId Whisper::openSequenceAndPrompt()
{
    uploadTokens(promptTokens);

    auto commands = gpu->makeCommandBuffer();

    decoder->beginSequence(commands, *encodedBuffer, *decoderWeights);
    decoder->step(
        commands, *tokenBuffer, promptTokens.size(), *decoderWeights, *logitBuffer);
    encodeSampling(commands, promptTokens.size(), *firstStepMaskBuffer);

    decodeSeconds += commitSeconds(commands);
    ++stepCount;

    return sampledToken();
}

TokenId Whisper::decodeStep(TokenId token)
{
    const auto one = Span<const TokenId> {&token, &token + 1};
    uploadTokens(one);

    auto commands = gpu->makeCommandBuffer();

    decoder->step(commands, *tokenBuffer, 1, *decoderWeights, *logitBuffer);
    encodeSampling(commands, 1, *laterStepMaskBuffer);

    decodeSeconds += commitSeconds(commands);
    ++stepCount;

    return sampledToken();
}

TokenId Whisper::endOfTextToken() const
{
    return modelConfig.endOfSequenceToken != ModelConfig::noToken
               ? modelConfig.endOfSequenceToken
               : vocabulary->specials().endOfText;
}

Vector<TokenId> Whisper::transcribe(Span<const float> samples)
{
    requirePrepared();

    if (samples.size() > windowSamples)
        throw ModelError {"this run holds " + std::to_string(samples.size())
                          + " samples and one window is "
                          + std::to_string(windowSamples)
                          + "; chunking a longer recording is a layer that does "
                            "not exist yet"};

    uploadSamples(samples);

    decodeSeconds = 0.0;
    stepCount = 0;
    encodeSeconds = encodeAudio();

    const auto endOfText = endOfTextToken();
    const auto maxPositions = decoder->shape().maxPositions;

    auto transcript = Vector<TokenId> {};
    auto sampled = openSequenceAndPrompt();

    while (sampled != endOfText && transcript.size() < maximumTokenCount)
    {
        transcript.add(sampled);

        // The window is full when the next step would have no position to embed
        // at, which Decoder::step would throw over — a transcript that ran out
        // of window is a stop, not a failure.
        if (decoder->position() + 1 > maxPositions)
            break;

        sampled = decodeStep(sampled);
    }

    return transcript;
}

std::string Whisper::transcribeText(Span<const float> samples)
{
    const auto tokens = transcribe(samples);
    return textForTokens(tokens);
}

std::string Whisper::textForTokens(Span<const TokenId> tokens) const
{
    requireLoaded();
    return vocabulary->decode(tokens);
}
} // namespace WSP
