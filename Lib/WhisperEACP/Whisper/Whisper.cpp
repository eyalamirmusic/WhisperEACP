#include "Whisper.h"

#include "ResourcesDirectory.h"

#include <WhisperEACP/Model/ModelIO.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <optional>
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
using eacp::GPU::DispatchOrder;

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

using Clock = std::chrono::steady_clock;

double secondsSince(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
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

void Whisper::setPacksLogitsWeight(bool shouldPack)
{
    packedLogitsWeight = shouldPack;
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
    decoderWeights.emplace(*weightsFile,
                           decoder->shape(),
                           packedLogitsWeight
                               ? DecoderWeights::LogitsWeight::PackedHalfCopy
                               : DecoderWeights::LogitsWeight::Tied);

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

// The mel and the encoder, one command buffer and one pass. The pass is
// concurrent, which for a chain this dependent buys exactly one thing: the
// query, key and value projections of each layer read the same normalised rows
// and write three buffers of their own, so those three overlap. Every other
// boundary is a barrier the recording code spells out, and a barrier in a
// concurrent pass costs a little more than a serial pass's own ordering — so
// this is a win only because eight of some sixty boundaries disappear. On an
// M4 Max over the 30 s window it is 9.9 ms against 9.6 ms.
double Whisper::encodeAudio()
{
    auto commands = gpu->makeCommandBuffer();

    {
        auto pass = commands.beginCompute({}, DispatchOrder::Concurrent);
        frontEnd.encode(pass, *sampleBuffer, *filterBank, *melBuffer);
        pass.barrier();
        encoder->encode(pass, *melBuffer, *encoderWeights, *encodedBuffer);
    }

    const auto start = Clock::now();
    commands.commit();

    return secondsSince(start);
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
        pass.barrier();

        decoder->step(pass,
                      sequenceSlots(0, promptLength),
                      promptLength,
                      *decoderWeights,
                      *logitBuffer);
        pass.barrier();
        encodeSampling(pass, promptLength, *firstStepMaskBuffer, promptLength);
        return;
    }

    decoder->step(pass,
                  sequenceSlots(promptLength + step - 1, 1),
                  1,
                  *decoderWeights,
                  *logitBuffer);
    pass.barrier();
    encodeSampling(pass, 1, *laterStepMaskBuffer, promptLength + step);
}

TokenId Whisper::sampledToken(CommandBuffer& commands, int step) const
{
    auto id = std::uint32_t {};
    commands.read(*sequenceTokens,
                  &id,
                  (int) sizeof(id),
                  (int) sizeof(id) * (promptTokens.size() + step));

    return (TokenId) id;
}

TokenId Whisper::endOfTextToken() const
{
    return modelConfig.endOfSequenceToken != ModelConfig::noToken
               ? modelConfig.endOfSequenceToken
               : vocabulary->specials().endOfText;
}

// The search is HF's greedy generate, one command buffer to a step and
// stepsInFlight of them in the air: step k + 1 is recorded and submitted before
// the host asks the GPU for step k, so the two overlap instead of taking turns.
// Each step's token is read out of that step's own command buffer, which waits
// for it alone — Buffer::read waits for the newest submission, and would
// therefore wait for the step still running — and the run ends on
// `<|endoftext|>`, the token limit or the window's end, with only the steps
// already in the air computed past it.
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
    const auto promptLength = promptTokens.size();

    // How many steps there is room for: the transcript's own limit, and the
    // window's — step j samples into slot promptLength + j, and the sequence
    // buffer holds one slot per position and one more.
    const auto stepLimit = std::min(
        maximumTokenCount, decoder->shape().maxPositions - promptLength + 1);

    // A CommandBuffer is neither copyable nor movable, so the ring holds them
    // in place. Slot j % stepsInFlight belongs to step j, and is replaced only
    // once that step's token has been read.
    auto inFlight = std::array<std::optional<CommandBuffer>, stepsInFlight> {};

    const auto startStep = [&](int step)
    {
        auto& commands = inFlight[step % stepsInFlight].emplace(*gpu);

        // Concurrent for the encode pass's reason and with the same
        // arithmetic: a step is some sixty dispatches and all but a few of its
        // boundaries are barriers, and the few that are not are each layer's
        // three self-attention projections. Those are small — one token's row
        // against a 384-wide weight — which is why overlapping them is worth
        // what the barriers around them cost: 21.8 ms to 21.0 ms for the 25
        // steps of jfk.wav, and 0.87 ms a step to 0.84 ms.
        //
        // The eight cross-attention projections beginSequence opens with
        // overlap here too, and they are the largest independent set there is
        // — 1500 encoder rows each. Giving them a concurrent pass of their own
        // against a serial pass for the steps was measured and changed
        // nothing: at that size a single projection already fills the device,
        // so there is nothing for a second one to overlap with.
        {
            auto pass = commands.beginCompute({}, DispatchOrder::Concurrent);
            encodeStep(pass, step);
        }

        commands.submit();
    };

    const auto decodeStart = Clock::now();

    auto transcript = Vector<TokenId> {};
    auto nextStep = 0;

    while (nextStep < stepsInFlight && nextStep < stepLimit)
    {
        startStep(nextStep);
        ++nextStep;
    }

    for (auto step = 0; step < nextStep; ++step)
    {
        const auto sampled = sampledToken(*inFlight[step % stepsInFlight], step);
        ++stepCount;

        if (sampled == endOfText)
            break;

        transcript.add(sampled);

        // The command buffer just read is the ring slot the next step records
        // into, and exactly one step is started per step read, so the ring
        // refills without ever overwriting one that is still running.
        if (nextStep < stepLimit)
        {
            startStep(nextStep);
            ++nextStep;
        }
    }

    decodeSeconds = secondsSince(decodeStart);

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
