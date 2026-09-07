#include "Common.h"

#include <eacp/GPU/GPU.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

// The end-to-end fixture: 11 seconds of real speech in, the sentence tiny.en
// produces out. Everything below the transcript is asserted somewhere else in
// this suite against a scalar reference or against whisper.cpp; what this adds
// is that the whole stack composes, which no stage-level test can say.
//
// Skips without a device, without the model, or without the sample, the same
// shape as every other test here that needs a download.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;
using namespace eacp::GPU;

namespace
{
bool canRun()
{
    return Device::shared().isValid() && hasWhisperModel()
           && hasSampleFile(jfkSample);
}

// One prepared runtime for the whole module: loading maps 151 MB and preparing
// compiles every kernel and uploads every tensor, and none of that is what these
// tests are about.
Whisper& preparedModel()
{
    static const auto model = []
    {
        auto whisper = std::make_unique<Whisper>();
        whisper->load(modelDirectory());
        whisper->prepare();
        return whisper;
    }();

    return *model;
}
} // namespace

auto tJfkTranscribes = test("Whisper/jfkTranscribes") = []
{
    if (!canRun())
        return;

    auto& whisper = preparedModel();
    const auto samples = readWavFile(sampleFile(jfkSample));

    check(samples.size() == 176000);
    check(samples.size() < windowSamples);

    const auto tokens = whisper.transcribe(samples);
    const auto text = whisper.textForTokens(tokens);

    check(tokens.size() == jfkTokenCount);
    check(!text.empty());
    check(text.front() == ' ');
    check(trimmed(text) == jfkTranscript);

    std::cout << "  jfk.wav: " << tokens.size() << " tokens, mel + encoder "
              << whisper.lastEncodeSeconds() << " s, decode "
              << whisper.lastDecodeSeconds() << " s over " << whisper.lastStepCount()
              << " steps ("
              << 1000.0 * whisper.lastDecodeSeconds() / whisper.lastStepCount()
              << " ms each)\n";
};

// The transcript holds no special token at all — not `<|endoftext|>`, which the
// loop stops on rather than emitting, and no timestamp, which is the claim the
// unsuppressed timestamp range rests on: `<|notimestamps|>` in the prompt is
// enough on its own, which is HF's position and not whisper.cpp's.
auto tJfkHasNoSpecialTokens = test("Whisper/jfkHasNoSpecialTokens") = []
{
    if (!canRun())
        return;

    auto& whisper = preparedModel();
    const auto samples = readWavFile(sampleFile(jfkSample));
    const auto tokens = whisper.transcribe(samples);

    const auto& specials = whisper.tokenizer().specials();

    for (auto token: tokens)
    {
        check(!whisper.tokenizer().isSpecial(token));
        check(!specials.isTimestamp(token));
        check(whisper.laterStepMask()[token] == 0.0f);
    }

    // And the first one was picked under the stricter mask, so it is neither a
    // bare space nor an immediate end of text.
    check(whisper.firstStepMask()[tokens[0]] == 0.0f);
};

// What the first-step mask is *for*, on an input that can actually show it.
//
// Most inputs cannot. Dropping begin_suppress_tokens and re-running was measured
// over thirteen signals — jfk.wav, silence, white noise, a click, a DC offset,
// jfk reversed, jfk at a hundredth of its level — and twelve of them come back
// the same word for word, because on anything with structure the model's first
// choice is a word whether or not `<|endoftext|>` and a bare space are in the
// running. That makes a transcript fixture a poor witness for this mask, and it
// is worth saying so rather than letting a green suite imply otherwise.
//
// A pure 440 Hz tone is the thirteenth. Asked to transcribe a sine wave the
// model answers `<|endoftext|>` at once — an empty transcript, which is a
// perfectly reasonable answer and exactly the one HF's
// SuppressTokensAtBeginLogitsProcessor forbids at the first sampled position.
// With the mask it says " Oh"; without it, nothing at all.
auto tAToneStillDecodesSomething = test("Whisper/aToneStillDecodesSomething") = []
{
    if (!canRun())
        return;

    auto& whisper = preparedModel();
    auto tone = std::vector<float>((std::size_t) windowSamples);

    for (auto index = std::size_t {}; index < tone.size(); ++index)
        tone[index] = 0.4f
                      * std::sin(2.0f * std::numbers::pi_v<float>
                                 * 440.0f * (float) index / (float) sampleRate);

    const auto tokens = whisper.transcribe(tone);
    const auto endOfText = whisper.tokenizer().specials().endOfText;

    check(tokens.size() >= 1);

    // EA::Vector's subscript is unchecked, so the empty case has to leave rather
    // than assert its way into undefined behaviour: the check above is the
    // finding, and a segfault behind it would only hide the message.
    if (tokens.size() < 1)
        return;

    check(tokens[0] != endOfText);
    check(whisper.firstStepMask()[tokens[0]] == 0.0f);
};

// Twice through the same object gives the same answer: beginSequence resets the
// position and the KV cache is rewritten from row zero, so a second run is not a
// continuation of the first.
auto tTranscribeRepeats = test("Whisper/transcribeRepeats") = []
{
    if (!canRun())
        return;

    auto& whisper = preparedModel();
    const auto samples = readWavFile(sampleFile(jfkSample));

    const auto first = whisper.transcribe(samples);
    const auto second = whisper.transcribe(samples);

    check(first.size() == second.size());

    for (auto index = 0; index < first.size(); ++index)
        check(first[index] == second[index]);
};

// A cap the caller set stops the loop where it says, which is what makes the
// default a policy rather than the only behaviour.
auto tMaximumTokensStopsTheLoop = test("Whisper/maximumTokensStopsTheLoop") = []
{
    if (!canRun())
        return;

    auto& whisper = preparedModel();
    const auto samples = readWavFile(sampleFile(jfkSample));

    whisper.setMaximumTokens(5);
    const auto clipped = whisper.transcribe(samples);

    whisper.setMaximumTokens(whisper.config().maxTargetPositions
                             - whisper.prompt().size());
    const auto whole = whisper.transcribe(samples);

    check(clipped.size() == 5);
    check(whole.size() == jfkTokenCount);

    for (auto index = 0; index < clipped.size(); ++index)
        check(clipped[index] == whole[index]);
};

// The same recording through a runtime loaded from bytes rather than from a
// directory. Every tensor it reads comes out of a borrowed view of a buffer the
// test holds, so this is what says the weights reach the device intact that way
// — which comparing what load() computed cannot, since none of that touches the
// blob.
auto tMemoryLoadedModelTranscribesTheSame =
    test("Whisper/memoryLoadedModelTranscribesTheSame") = []
{
    if (!canRun())
        return;

    const auto samples = readWavFile(sampleFile(jfkSample));
    const auto expected = preparedModel().transcribe(samples);

    auto whisper = Whisper {};
    whisper.load(modelFileBytes().files());
    whisper.prepare();

    const auto tokens = whisper.transcribe(samples);

    check(tokens.size() == expected.size());

    for (auto index = 0; index < tokens.size() && index < expected.size(); ++index)
        check(tokens[index] == expected[index]);
};

// More than a window is refused rather than truncated: which thirty seconds of a
// long recording to keep, and what to carry across the cut, is a chunking layer
// that does not exist here yet.
auto tLongerThanAWindowIsAnError = test("Whisper/longerThanAWindowIsAnError") = []
{
    if (!canRun())
        return;

    auto& whisper = preparedModel();
    const auto tooMuch = std::vector<float>((std::size_t) windowSamples + 1, 0.0f);

    check(
        mentions(modelErrorFrom([&] { whisper.transcribe(tooMuch); }), "chunking"));
};
