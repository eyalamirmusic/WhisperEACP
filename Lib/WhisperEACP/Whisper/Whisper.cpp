#include "Whisper.h"

#include "ResourcesDirectory.h"

#include <WhisperEACP/Model/ModelIO.h>

#include <chrono>
#include <string>
#include <system_error>

namespace WSP
{
using eacp::GPU::Buffer;
using eacp::GPU::BufferRange;
using eacp::GPU::BufferUsage;
using eacp::GPU::CommandBuffer;
using eacp::GPU::ComputePass;
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

bool hasFile(const std::filesystem::path& directory, std::string_view name)
{
    auto error = std::error_code {};
    return std::filesystem::is_regular_file(directory / name, error);
}

std::filesystem::path requireFile(const std::filesystem::path& directory,
                                  std::string_view name)
{
    if (!hasFile(directory, name))
        throw ModelError {"the model directory " + directory.string() + " has no "
                          + std::string {name}};

    return directory / name;
}

bool isDirectory(const std::filesystem::path& path)
{
    auto error = std::error_code {};
    return std::filesystem::is_directory(path, error);
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

    buildGenerationConfig();
}

// The same four parses as above, in the same order, so a set of bytes missing
// what a directory would have been missing fails the same way and says so with
// the same message.
void Whisper::load(const ModelFiles& files)
{
    modelConfig = ModelConfig::fromJson(ModelIO::textOf(files.config));
    preprocessor =
        PreprocessorConfig::fromJson(ModelIO::textOf(files.preprocessorConfig));
    vocabulary.emplace(Tokenizer::fromJsonText(ModelIO::textOf(files.tokenizer)));
    weightsFile.emplace(SafeTensors::fromView(files.weights));

    buildGenerationConfig();
}

std::filesystem::path Whisper::bundledModelDirectory()
{
    return resourcesDirectory() / bundledModelDirectoryName;
}

bool Whisper::hasBundledModel()
{
    const auto directory = bundledModelDirectory();

    return hasFile(directory, configFile) && hasFile(directory, preprocessorFile)
           && hasFile(directory, tokenizerFile)
           && hasFile(directory, weightsFileName);
}

// No directory at all is a build that never asked for the copy, and the message
// says how to ask. One that is there but short a file is a copy that did not
// finish, and load() names the file the way it would for any directory.
void Whisper::loadBundled()
{
    const auto directory = bundledModelDirectory();

    if (!isDirectory(directory))
        throw ModelError {"this binary ships no model: there is no "
                          + directory.string()
                          + "; a model is copied there by calling "
                            "whisper_bundle_model(<target>) in the target's "
                            "CMakeLists, in a build configured with "
                            "-DWHISPER_EACP_FETCH_MODEL=ON"};

    load(directory);
}

void Whisper::buildGenerationConfig()
{
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
    selection.prepare(device, 1, decoder->shape().logitElementCount());

    encoderWeights.emplace(*weightsFile, encoder->shape());
    decoderWeights.emplace(*weightsFile, decoder->shape());

    filterBank.emplace(preprocessor.makeMelFilterBuffer());

    paddedSamples.resize(windowSamples);
    promptIds.resize(promptTokens.size());

    for (auto index = 0; index < promptTokens.size(); ++index)
        promptIds[index] = (std::uint32_t) promptTokens[index];

    sampleBuffer.emplace(
        device.makeBuffer(floatBytes(windowSamples), BufferUsage::Storage));
    melBuffer.emplace(device.makeBuffer(
        floatBytes(frontEnd.shape().melElementCount()), BufferUsage::Storage));
    encodedBuffer.emplace(device.makeBuffer(
        floatBytes(encoder->shape().elementCount()), BufferUsage::Storage));

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

    sequenceTokens.emplace(device.makeBuffer(
        (int) sizeof(std::uint32_t) * (decoder->shape().maxPositions + 1),
        BufferUsage::Storage));
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

void Whisper::uploadPrompt()
{
    sequenceTokens->update(promptIds.data(),
                           (int) sizeof(std::uint32_t) * promptIds.size());
}

double Whisper::encodeAudio()
{
    auto commands = gpu->makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        frontEnd.encode(pass, *sampleBuffer, *filterBank, *melBuffer);
        encoder->encode(pass, *melBuffer, *encoderWeights, *encodedBuffer);
    }

    return commitSeconds(commands);
}

BufferRange Whisper::sequenceSlots(int first, int count) const
{
    constexpr auto slotBytes = (int) sizeof(std::uint32_t);
    return {&*sequenceTokens, first * slotBytes, count * slotBytes};
}

// Greedy sampling over the last row of the step's logits, which is the
// distribution over the token that follows everything decoded so far. The row
// is bound as a range rather than the whole buffer — the same ranged bind the
// KV cache needed — so a prompt step's earlier rows are neither scanned nor
// copied, and the answer lands in the sequence slot the next step embeds.
void Whisper::encodeSampling(ComputePass& pass,
                             int rowCount,
                             const Buffer& mask,
                             int slot)
{
    const auto rowLength = decoder->shape().logitElementCount();
    const auto rowBytes = floatBytes(rowLength);

    selection.encode(
        pass,
        BufferRange {&*logitBuffer, (rowCount - 1) * rowBytes, rowBytes},
        mask,
        sequenceSlots(slot, 1),
        1,
        rowLength);
}

void Whisper::encodeStep(ComputePass& pass, int step)
{
    const auto promptLength = promptTokens.size();

    if (step == 0)
    {
        decoder->beginSequence(pass, *encodedBuffer, *decoderWeights);
        decoder->step(pass,
                      sequenceSlots(0, promptLength),
                      promptLength,
                      *decoderWeights,
                      *logitBuffer);
        encodeSampling(pass, promptLength, *firstStepMaskBuffer, promptLength);
        return;
    }

    decoder->step(pass,
                  sequenceSlots(promptLength + step - 1, 1),
                  1,
                  *decoderWeights,
                  *logitBuffer);
    encodeSampling(pass, 1, *laterStepMaskBuffer, promptLength + step);
}

TokenId Whisper::sampledToken(int step) const
{
    auto id = std::uint32_t {};
    sequenceTokens->read(
        &id, (int) sizeof(id), (int) sizeof(id) * (promptTokens.size() + step));

    return (TokenId) id;
}

TokenId Whisper::endOfTextToken() const
{
    return modelConfig.endOfSequenceToken != ModelConfig::noToken
               ? modelConfig.endOfSequenceToken
               : vocabulary->specials().endOfText;
}

// The search is HF's greedy generate, stepsPerCommit steps at a time: a batch
// is recorded with each step embedding the slot the one before it sampled,
// committed, and its slots read back in order until `<|endoftext|>`, the
// token limit, or the window's end. A step is recorded only while there is a
// position to embed at and a token the transcript could still take, so the
// last batch is short rather than thrown over by Decoder::step.
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
    uploadPrompt();

    decodeSeconds = 0.0;
    stepCount = 0;
    encodeSeconds = encodeAudio();

    const auto endOfText = endOfTextToken();
    const auto maxPositions = decoder->shape().maxPositions;
    const auto promptLength = promptTokens.size();

    auto transcript = Vector<TokenId> {};
    auto nextStep = 0;

    while (true)
    {
        const auto firstStep = nextStep;
        auto commands = gpu->makeCommandBuffer();

        {
            auto pass = commands.beginCompute();

            for (auto recorded = 0; recorded < stepsPerCommit; ++recorded)
            {
                const auto position = promptLength + nextStep;
                const auto pending = transcript.size() + recorded;

                if (position > maxPositions || pending >= maximumTokenCount)
                    break;

                encodeStep(pass, nextStep);
                ++nextStep;
            }
        }

        if (nextStep == firstStep)
            return transcript;

        decodeSeconds += commitSeconds(commands);

        for (auto step = firstStep; step < nextStep; ++step)
        {
            ++stepCount;

            const auto sampled = sampledToken(step);

            if (sampled == endOfText)
                return transcript;

            transcript.add(sampled);
        }
    }
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
