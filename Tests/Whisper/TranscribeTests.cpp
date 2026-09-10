#include "Common.h"

#include <eacp/GPU/GPU.h>

#include <cmath>
#include <iostream>
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

// The arithmetic of sizing a context to a recording, which needs no device: a
// position is 320 samples — a 160-sample hop through a convolution of stride
// two — and the count is rounded up to the 64 the products tile at and held
// inside the window.
auto tAudioContextForSamples = test("Whisper/audioContextForSamples") = []
{
    check(Whisper::audioContextTile == 64);

    // jfk.wav is 176000 samples, which is 550 positions: 576 with no margin,
    // 640 with a second of it, and 704 with the default 2.5 s.
    check(Whisper::audioContextForSamples(176000, 0.0) == 576);
    check(Whisper::audioContextForSamples(176000, 1.0) == 640);
    check(Whisper::audioContextForSamples(176000) == 704);

    // The floor holds a short segment up and the window is the ceiling however
    // much is asked for. A second of audio and a second of margin is 100
    // positions, and 100 positions is where the decoder loops.
    check(Whisper::audioContextForSamples(0, 0.0) == Whisper::audioContextFloor);
    check(Whisper::audioContextForSamples(sampleRate, 1.0)
          == Whisper::audioContextFloor);
    check(Whisper::audioContextForSamples(sampleRate) == Whisper::audioContextFloor);
    check(Whisper::audioContextForSamples(windowSamples) == encoderPositions);
    check(Whisper::audioContextForSamples(windowSamples, 0.0) == encoderPositions);
};

auto tAudioContextIsValidated = test("Whisper/audioContextIsValidated") = []
{
    if (!canRun())
        return;

    auto& whisper = preparedModel();

    check(whisper.audioContext() == 0);
    check(throwsModelError([&] { whisper.setAudioContext(-1); }));
    check(throwsModelError([&] { whisper.setAudioContext(encoderPositions + 1); }));
    check(whisper.audioContext() == 0);

    whisper.setAudioContext(640);
    check(whisper.audioContext() == 640);

    whisper.setAudioContext(0);
    check(whisper.audioContext() == 0);
};

// The transcript at every context worth reading, against the full window's.
// The numbers this prints are the measurement the option rests on, and the
// assertion is where they stop being the same sentence: jfk.wav is 11 s, which
// is 550 positions, and a context of 640 — the 550 plus the default second of
// margin, rounded to a tile — reproduces the full window's 24 tokens exactly.
//
// 576 does not. It is a tile above the audio and 0.5 s of margin, and it drops
// the comma after "for you". That is the transcript the model produces when it
// is asked to encode less silence than it was trained on, not a bug in the
// truncation: whisper.cpp at the same audio_ctx produces the same tokens, which
// is what Tests/Oracle asserts.
auto tAudioContextTranscripts = test("Whisper/audioContextTranscripts") = []
{
    if (!canRun())
        return;

    auto& whisper = preparedModel();
    const auto samples = readWavFile(sampleFile(jfkSample));

    for (auto context: {1500, 1024, 768, 640, 576, 512})
    {
        whisper.setAudioContext(context);

        const auto tokens = whisper.transcribe(samples);
        const auto text = trimmed(whisper.textForTokens(tokens));

        std::cout << "  " << context << " positions, " << tokens.size()
                  << " tokens: " << text << "\n";

        if (context >= 640)
        {
            check(tokens.size() == jfkTokenCount);
            check(text == jfkTranscript);
        }
    }

    whisper.setAudioContext(0);

    const auto whole = whisper.transcribe(samples);
    check(whole.size() == jfkTokenCount);
    check(trimmed(whisper.textForTokens(whole)) == jfkTranscript);
};

// The default is off, and off is what every other test in this module runs
// under: a run with no context set encodes the window whatever the audio is.
auto tAudioContextIsOffByDefault = test("Whisper/audioContextIsOffByDefault") = []
{
    auto whisper = Whisper {};
    check(whisper.audioContext() == 0);
};
