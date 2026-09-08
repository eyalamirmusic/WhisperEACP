#pragma once

#include <WhisperEACP/Decoder/Decoder.h>
#include <WhisperEACP/Encoder/Encoder.h>
#include <WhisperEACP/Mel/Mel.h>
#include <WhisperEACP/Model/Model.h>
#include <WhisperEACP/Tokenizer/Tokenizer.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace WSP
{
// The whole runtime, from 30 s of samples to text: the mel front-end, the
// encoder, the decoder's KV-cached loop, and the greedy search that turns the
// last logit row into the next token.
//
// This is the layer the decoder deliberately left empty. Which tokens are
// suppressed at which step is generation config, so `Argmax` is bound here
// rather than inside `Decoder`, and the two masks below are the whole of that
// config.
//
// A run is three phases, and they are three phases because of where the host
// has to see a number:
//
//   1. one command buffer for the upload, the mel and the encoder, committed;
//   2. one for `beginSequence` and the prompt step, ending in the argmax that
//      picks the first sampled token;
//   3. one command buffer per token after it.
//
// Phase 3 cannot be batched. The token a step samples is the token the next
// step embeds, so the index has to come back to the host between them —
// `commit()` blocks until the buffer is done, and `Buffer::read` is only valid
// once it has. A decoder that could sample on the GPU and feed its own next
// step would collapse the whole loop into one command buffer; nothing in eacp's
// compute layer expresses that today (no indirect dispatch off a sampled index,
// and no integer buffer a kernel could write a token into for `Embed` to read).
//
// **The suppression rule, and where the two references differ.** Matched to
// HuggingFace's greedy `generate`, since that is what every other stage here is
// checked against:
//
//   * `suppress_tokens` from config.json is masked at every step. 90 ids for
//     tiny.en — the non-speech pieces, plus `<|startoftranscript|>` and the
//     `<|nospeech|>` group.
//   * `begin_suppress_tokens` — [220, 50256], a leading space and
//     `<|endoftext|>` — is masked at the *first* sampled step only, which is
//     `SuppressTokensAtBeginLogitsProcessor` with `begin_index` at the end of
//     the prompt.
//   * **Timestamps are not masked.** HF adds a timestamp processor only for
//     `return_timestamps=True`; without it, `<|notimestamps|>` in the prompt is
//     the whole of what keeps timestamps out of the transcript, and it is
//     enough.
//
// whisper.cpp differs on all three, and does so deliberately for a decoder that
// segments (`whisper_process_logits`, src/whisper.cpp:6194):
//
//   * with `no_timestamps`, every id from `<|0.00|>` up is masked at every step
//     — a hard guarantee where HF relies on the prompt;
//   * `<|notimestamps|>`, `<|startoftranscript|>`, `<|startofprev|>`, the task
//     tokens and every language token are masked unconditionally. HF's
//     `suppress_tokens` covers `<|startoftranscript|>` and the `<|nospeech|>`
//     group but *not* `<|notimestamps|>`;
//   * its `suppress_nst` — the same non-speech list HF ships as
//     `suppress_tokens` — defaults to **off**, so a default whisper.cpp run
//     suppresses less than a default HF one, not more.
//
// Its `suppress_blank` (on by default) is exactly `begin_suppress_tokens`, so
// the two agree on the one that decides jfk.wav's first word.
class Whisper
{
public:
    Whisper() = default;

    Whisper(const Whisper&) = delete;
    Whisper& operator=(const Whisper&) = delete;

    // The four files of a HuggingFace Whisper repo, read from one directory:
    // config.json, preprocessor_config.json, model.safetensors and
    // tokenizer.json. A missing one is a ModelError naming it, since "load
    // failed" about a four-file directory is a message that costs the caller
    // the `ls`.
    //
    // No GPU is touched here — the weights are mapped, not uploaded. That is
    // prepare()'s half.
    void load(const std::filesystem::path& modelDirectory);

    // The same four files, from bytes already in memory. ModelFiles is a view:
    // the weights are borrowed rather than copied (SafeTensors::fromView), so
    // the bytes must outlive this object — every tensor this reads, and every
    // buffer prepare() uploads, comes out of them.
    void load(const ModelFiles& files);

    // The model the build copied beside this binary, if it copied one: the
    // four files under bundledModelDirectoryName, in the bundle's Resources on
    // macOS and next to the executable otherwise. A ModelError names the
    // directory it looked for when there is none, since a build that meant to
    // ship a model and never asked for the copy should not read as a corrupt
    // one; a directory that is there but short a file fails the way load(path)
    // does, naming the file.
    //
    // A binary gets one by calling whisper_bundle_model(<target>) in its
    // CMakeLists, in a build configured with -DWHISPER_EACP_FETCH_MODEL=ON.
    // Nothing here copies a model on its own: 151 MB beside every executable is
    // a decision the build makes, not the runtime.
    void loadBundled();

    static bool hasBundledModel();

    // resourcesDirectory() / bundledModelDirectoryName, whether or not anything
    // is there.
    static std::filesystem::path bundledModelDirectory();

    // The directory whisper_bundle_model copies into. Spelled here and in
    // Model/CMakeLists.txt, and nowhere else.
    static constexpr auto bundledModelDirectoryName = "WhisperModel";

    bool isLoaded() const { return weightsFile.has_value(); }

    // Compiles every kernel, sizes every intermediate, and uploads the weights,
    // the filterbank and the two suppression masks.
    //
    // The device argument reaches the pipelines only. The buffers go up through
    // SafeTensors::makeBuffer and PreprocessorConfig::makeMelFilterBuffer,
    // neither of which takes a device — they use Device::shared() — so a
    // prepare() against a second Device would compile there and upload here.
    // Recorded rather than hidden: it is a seam in the loader, not in this.
    void prepare(eacp::GPU::Device& device);
    void prepare();

    bool isPrepared() const { return decoder.has_value(); }

    const ModelConfig& config() const { return modelConfig; }
    const Tokenizer& tokenizer() const;

    // The tokens a run opens the sequence with: `<|startoftranscript|>` and
    // `<|notimestamps|>` for an English-only model.
    //
    // A multilingual repo would put a language token and `<|transcribe|>` or
    // `<|translate|>` between the two — `<|startoftranscript|><|en|>`
    // `<|transcribe|><|notimestamps|>`. Not implemented: choosing the language
    // means either being told it or running the detection pass HF's
    // `detect_language` does, and neither belongs in a first transcribe.
    Span<const TokenId> prompt() const { return promptTokens; }

    // The two masks as Argmax reads them: one float per vocabulary entry,
    // nonzero at a token that may not be sampled. Exposed so the rule above can
    // be asserted by id rather than only through a transcript.
    Span<const float> firstStepMask() const { return firstStepSuppression; }
    Span<const float> laterStepMask() const { return laterStepSuppression; }

    // The most tokens a transcript may hold, `<|endoftext|>` excluded.
    //
    // The default is HF's: `max_length` is `max_target_positions` (448) over the
    // whole decoder sequence, so what is left for sampling is that less the
    // prompt — 446 for tiny.en. whisper.cpp caps a segment at
    // `n_text_ctx/2 - 4` (220, src/whisper.cpp:7221) instead, which is a
    // chunking policy: it decodes 30 s at a time and needs room to carry a
    // previous segment's tokens as a prompt. There is no chunking layer here
    // yet, so the bound that applies is the window's own.
    int maximumTokens() const { return maximumTokenCount; }
    void setMaximumTokens(int count);

    // 16 kHz mono samples, at most one 30 s window of them.
    //
    // Fewer are zero-filled to the window, which is what HF's feature extractor
    // does and what keeps the positional embedding on the frames it was trained
    // for. **More is a ModelError**, not a truncation and not a second window:
    // deciding where to cut a long recording, and what to carry across the cut,
    // is a chunking layer that does not exist yet, and silently transcribing the
    // first thirty seconds of a five-minute file would be the worst of the
    // available answers.
    Vector<TokenId> transcribe(Span<const float> samples);

    // The same run, decoded through the vocabulary with the special tokens
    // dropped — the transcript as text, leading space and all.
    std::string transcribeText(Span<const float> samples);

    std::string textForTokens(Span<const TokenId> tokens) const;

    // Wall clock around the commits of the last run, which is the only clock
    // available: eacp's FrameTimer is driven by Frame and a CommandBuffer has no
    // timestamp hook, so an off-screen compute pass is timed from the host.
    double lastEncodeSeconds() const { return encodeSeconds; }
    double lastDecodeSeconds() const { return decodeSeconds; }

    // How many decoder steps the last run consumed, which is one for the
    // prompt and one per token after it — one more than the transcript holds
    // when the run ended on `<|endoftext|>`, since that token was sampled and
    // then dropped. Steps are committed stepsPerCommit at a time, each
    // embedding the token the one before it sampled on the device, so a run
    // may have computed a few steps past the one it ended on; those are not
    // counted, though the clock above includes them.
    int lastStepCount() const { return stepCount; }

    // How many steps go into one command buffer. A commit is a fixed cost —
    // the submission, the wait and the wake — of a fifth of a step's work,
    // and the GPU idles between one commit and the next; four steps a commit
    // pays it a quarter as often and keeps the GPU fed, at the price of up to
    // three steps computed past the end of a transcript, which a transcript
    // of any length pays once.
    static constexpr int stepsPerCommit = 4;

private:
    void requireLoaded() const;
    void requirePrepared() const;
    void requireWhisperShapes() const;

    void buildPrompt();
    void buildSuppressionMasks();

    // Everything a load computes once the four files are parsed, whichever of
    // the two loads parsed them.
    void buildGenerationConfig();

    void uploadSamples(Span<const float> samples);
    void uploadPrompt();

    double encodeAudio();

    // Step zero opens the sequence and decodes the prompt; step j embeds the
    // token step j - 1 sampled. Every step samples into slot
    // prompt().size() + step of the sequence buffer, and reads its tokens
    // out of the slots before it.
    void encodeStep(eacp::GPU::ComputePass& pass, int step);
    void encodeSampling(eacp::GPU::ComputePass& pass,
                        int rowCount,
                        const eacp::GPU::Buffer& mask,
                        int slot);

    eacp::GPU::BufferRange sequenceSlots(int first, int count) const;
    TokenId sampledToken(int step) const;

    TokenId endOfTextToken() const;

    // The device every command buffer and every buffer this class owns comes
    // from — the one prepare() was handed, which the weights and the filterbank
    // are the documented exception to.
    eacp::GPU::Device* gpu = nullptr;

    ModelConfig modelConfig;
    PreprocessorConfig preprocessor;
    std::optional<Tokenizer> vocabulary;
    std::optional<SafeTensors> weightsFile;

    Vector<TokenId> promptTokens;
    Vector<float> firstStepSuppression;
    Vector<float> laterStepSuppression;
    int maximumTokenCount = 0;

    MelSpectrogram frontEnd;
    std::optional<Encoder> encoder;
    std::optional<Decoder> decoder;
    std::optional<EncoderWeights> encoderWeights;
    std::optional<DecoderWeights> decoderWeights;
    Argmax selection;

    // The padded window, kept rather than built per call: it is 1.9 MB, every
    // run fills the same 480000 floats, and a shorter utterance's tail has to be
    // zeroed again anyway.
    Vector<float> paddedSamples;
    Vector<std::uint32_t> promptIds;

    std::optional<eacp::GPU::Buffer> filterBank;
    std::optional<eacp::GPU::Buffer> sampleBuffer;
    std::optional<eacp::GPU::Buffer> melBuffer;
    std::optional<eacp::GPU::Buffer> encodedBuffer;
    std::optional<eacp::GPU::Buffer> logitBuffer;
    std::optional<eacp::GPU::Buffer> firstStepMaskBuffer;
    std::optional<eacp::GPU::Buffer> laterStepMaskBuffer;

    // The whole sequence as unsigned ids, one slot per position and one more
    // for the token sampled at the last: the prompt uploaded into the first
    // slots, and every slot after them written by the Argmax of one step and
    // read by the Embed of the next, on the device. The host reads a slot
    // back only to learn whether the run is over.
    std::optional<eacp::GPU::Buffer> sequenceTokens;

    double encodeSeconds = 0.0;
    double decodeSeconds = 0.0;
    int stepCount = 0;
};
} // namespace WSP
