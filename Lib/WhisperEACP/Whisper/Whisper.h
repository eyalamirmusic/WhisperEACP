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
// A run is two phases:
//
//   1. one command buffer for the upload, the mel and the encoder;
//   2. one command buffer per decoder step — the first opening the sequence and
//      decoding the prompt, every one after it embedding the token the step
//      before sampled — with stepsInFlight of them in the air at a time.
//
// The token a step samples never reaches the host before the next step needs
// it: `Argmax` writes it into a slot of the sequence buffer and the next step's
// `Embed` reads that same slot on the device, so a step is recorded and
// submitted without the host knowing what the one before it sampled. A slot is
// read back for the stop condition alone, and `CommandBuffer::read` waits for
// that one step rather than for the newest submission, so step k's token is
// read while step k + 1 is already running.
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

    // Whether the weights go to the device as packed fp16 wherever that is
    // bit-exact: every projection weight of the encoder and of the decoder,
    // and the fp16 copy of embed_tokens the logits projection reads instead of
    // the float matrix the gather reads. It is the bytes a run moves, which is
    // what it is bound by — 33 MB of layer projections a decode step read as
    // 16.5, and the vocabulary projection's 80 MB read as 40, which alone was
    // 133 us of a 590 us step.
    //
    // On by default, because it costs no accuracy: a weight is packed only
    // where narrowing is bit-exact, and a Whisper repo's weights are fp16
    // values in an F32 container — OpenAI's checkpoints are fp16 and
    // HuggingFace's conversion only widens them, which Tests/Model asserts over
    // every value of tiny.en. The same numbers are read through readHalf
    // instead of a subscript and accumulated in float32 either way, so the
    // logits come out bit for bit what the float weights produce. A model that
    // would lose something keeps the weights it shipped, and this reads back
    // true having changed nothing there. Set it before prepare(), which is
    // where the weights are uploaded.
    bool packsWeights() const { return packedWeights; }
    void setPacksWeights(bool shouldPack);

    // whisper.cpp's audio_ctx: how many of the encoder's 1500 positions a run
    // computes. Zero, the default, is the whole 30 s window, and every
    // existing number in this tree is that one.
    //
    // A shorter utterance does not need the whole window. The encoder pays for
    // all 1500 positions however little audio there is — the attention inside
    // it is quadratic in them — and every decode step's cross attention then
    // reads all 1500 rows of the projected keys and values. Setting this to
    // the positions the audio actually fills makes both scale with the
    // recording instead: the run computes that many rows out of the first
    // 2 * positions mel frames, adds the first that many rows of the
    // positional embedding, and attends to that many everywhere below. Nothing
    // is reallocated — the buffers are the window's and a short run writes a
    // prefix of each.
    //
    // What it costs is what the model was trained to see. The frames past the
    // audio are silence either way, but at the full window there are 1500
    // positions of it and the positional embedding runs to its own end; a
    // truncated one is a shape the model never saw. The transcripts at each
    // count are measured in Tests/Whisper and in plan.md rather than assumed,
    // and that is why this is off by default.
    int audioContext() const { return audioContextPositions; }
    void setAudioContext(int positions);

    // The positions a recording of sampleCount samples fills, plus
    // marginSeconds of the silence after it, rounded up to audioContextTile
    // and held between audioContextFloor and the window. 320 samples is one
    // position — a 160-sample hop through a convolution of stride two — so a
    // second of audio is 50 of them.
    static int
        audioContextForSamples(int sampleCount,
                               double marginSeconds = defaultAudioContextMargin);

    // What a context is rounded up to. The products under the encoder tile C
    // in 64 x 64, so a count that is a multiple of 64 is one where the last
    // tile of every one of them is whole.
    static constexpr int audioContextTile = 64;

    // Enough silence after the audio that the encoder is not asked to end at
    // the last word — Whisper's own window is 30 s of it — and measured
    // rather than chosen. Over prefixes of jfk.wav from three starting points
    // and at every context from the audio's own length upwards: a context
    // within about a second of what the audio fills either changes the
    // transcript or sends the decoder into a repetition loop (4 s of audio at
    // 256 positions, 9 s at 512), and everything two tiles clear of it agrees
    // with the window. 2.5 s is 125 positions, which after the rounding is
    // never less than those two tiles.
    static constexpr double defaultAudioContextMargin = 2.5;

    // And the floor, which is not a margin but a cliff. Measured over the
    // first 1, 2, 3, 5, 8 and 11 seconds of jfk.wav at every context down to
    // 128: at 384 and above the transcript is the window's, and at 256 and
    // below the decoder falls into a repetition loop and runs to the token
    // limit — "and so and so and so" for 446 tokens, which costs *ten times*
    // what the whole window would have. A context below this is a slower
    // wrong answer rather than a faster one, so a count computed from a short
    // segment is held here. Two tiles of headroom over the 384 that behaved.
    static constexpr int audioContextFloor = 448;

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

    // Wall clock around the last run's two halves, which is the only clock
    // available: eacp's FrameTimer is driven by Frame and a CommandBuffer has no
    // timestamp hook, so an off-screen compute pass is timed from the host.
    //
    // The encode is its own command buffer end to end, its commit. The decode
    // is the first step's submit to the read of the last token, and stops at
    // that read rather than waiting for the steps still in the air behind it —
    // so the two hold the run between them and neither counts the other.
    double lastEncodeSeconds() const { return encodeSeconds; }
    double lastDecodeSeconds() const { return decodeSeconds; }

    // How many decoder steps the last run consumed, which is one for the
    // prompt and one per token after it — one more than the transcript holds
    // when the run ended on `<|endoftext|>`, since that token was sampled and
    // then dropped. A step is counted when its token is read, so the steps
    // still in the air when a run ends are not: they were computed past the
    // end and never looked at, and the clock above stops at the last read
    // rather than waiting for them.
    int lastStepCount() const { return stepCount; }

    // How many decoder steps are in the air at a time. Each is a command buffer
    // of its own, recorded and submitted before the host waits on the step
    // before it, so the GPU takes the next one up the moment it finishes
    // instead of idling across a commit; the price is that the last
    // stepsInFlight - 1 steps of a run are computed past its end, which a
    // transcript of any length pays once.
    static constexpr int stepsInFlight = 2;

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

    // The token step j sampled, out of the command buffer that sampled it:
    // that one waited for, and nothing submitted behind it.
    TokenId sampledToken(eacp::GPU::CommandBuffer& commands, int step) const;

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
    bool packedWeights = true;
    int audioContextPositions = 0;

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
