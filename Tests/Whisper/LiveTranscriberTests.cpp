#include "Common.h"

#include <WhisperEACP/Whisper/LiveTranscriber.h>

#include <eacp/GPU/GPU.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

// The layer a microphone needs: samples pushed a block at a time, the open
// segment re-transcribed as it fills, and a committed line whenever silence or
// length closes it. The classifier is checked against synthetic blocks with no
// device at all; everything above it is jfk.wav played into the transcriber the
// way a capture callback would deliver it.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;
using namespace eacp::GPU;

namespace
{
constexpr auto blockSamples = sampleRate / 10;

bool canRun()
{
    return Device::shared().isValid() && hasWhisperModel()
           && hasSampleFile(jfkSample);
}

// 44 whole cycles in a 100 ms block, so its RMS is the amplitude over root two
// exactly and the level it should answer is arithmetic rather than a fixture.
std::vector<float> toneBlock(float amplitude)
{
    auto block = std::vector<float>((std::size_t) blockSamples);

    for (auto index = std::size_t {}; index < block.size(); ++index)
        block[index] = amplitude
                       * std::sin(2.0f * std::numbers::pi_v<float>
                                  * 440.0f * (float) index / (float) sampleRate);

    return block;
}

std::vector<float> silentChunk(int sampleCount)
{
    return std::vector<float>((std::size_t) sampleCount, 0.0f);
}

// What a capture callback does: hand over a fixed block and give the runtime a
// turn. Answers whether pending() held anything after the turn.
bool pushChunked(LiveTranscriber& live, Span<const float> samples, int chunk)
{
    auto sawPending = false;

    for (auto offset = 0; offset < samples.size(); offset += chunk)
    {
        live.push(samples.subspan(offset, std::min(chunk, samples.size() - offset)));
        live.update();

        sawPending = sawPending || !live.pending().empty();
    }

    return sawPending;
}

int chunkCount(int sampleCount, int chunk)
{
    return (sampleCount + chunk - 1) / chunk;
}

// The same 100 ms over and over, a block at a time with a turn between them,
// which is the drive the run policy is written against.
void pushRepeatedBlock(LiveTranscriber& live,
                       const std::vector<float>& block,
                       int count)
{
    for (auto index = 0; index < count; ++index)
    {
        live.push(block);
        live.update();
    }
}

// Speech, a pause shorter than the silence hold, then speech again: the runs
// the policy took in each of the three stretches.
struct PauseDrive
{
    int overSpeech = 0;
    int overThePause = 0;
    int total = 0;
};

PauseDrive driveAPause(double minNewSpeechSeconds)
{
    auto options = LiveOptions {};
    options.minNewSpeechSeconds = minNewSpeechSeconds;

    auto live = LiveTranscriber {preparedModel(), options};
    const auto speech = toneBlock(0.1f);
    const auto silence = silentChunk(blockSamples);

    auto drive = PauseDrive {};

    pushRepeatedBlock(live, speech, 15);
    drive.overSpeech = live.stats().runs;

    pushRepeatedBlock(live, silence, 9);
    drive.overThePause = live.stats().runs - drive.overSpeech;

    pushRepeatedBlock(live, speech, 10);
    drive.total = live.stats().runs;

    return drive;
}

// jfk.wav, then silence a block at a time until the hold closes the segment.
struct TrailingSilenceDrive
{
    int overSpeech = 0;
    int overTheSilence = 0;
    std::string committedLine;
};

TrailingSilenceDrive driveJfkThenSilence(double minNewSpeechSeconds)
{
    auto options = LiveOptions {};
    options.minNewSpeechSeconds = minNewSpeechSeconds;

    auto live = LiveTranscriber {preparedModel(), options};
    const auto samples = readWavFile(sampleFile(jfkSample));

    pushChunked(live, samples, blockSamples);

    auto drive = TrailingSilenceDrive {};
    drive.overSpeech = live.stats().runs;

    const auto silence = silentChunk(blockSamples);

    for (auto block = 0; block < 20 && live.committed().size() == 0; ++block)
    {
        live.push(silence);
        live.update();
    }

    drive.overTheSilence = live.stats().runs - drive.overSpeech;

    if (live.committed().size() > 0)
        drive.committedLine = trimmed(live.committed()[0]);

    return drive;
}
} // namespace

// No device, no model, no sample: the classifier is arithmetic over a block.
auto tBlockLevelsClassifySilenceAndSpeech =
    test("Live/blockLevelsClassifySilenceAndSpeech") = []
{
    const auto silence = silentChunk(blockSamples);

    check(std::isfinite(blockLevelDb(silence)));
    check(blockLevelDb(silence) < -100.0f);
    check(!isSpeechBlock(silence, -40.0f));

    const auto speech = toneBlock(0.1f);
    const auto tooQuiet = toneBlock(0.001f);

    check(std::abs(blockLevelDb(speech) + 23.0f) < 0.5f);
    check(isSpeechBlock(speech, -40.0f));

    check(std::abs(blockLevelDb(tooQuiet) + 63.0f) < 0.5f);
    check(!isSpeechBlock(tooQuiet, -40.0f));
};

// Five seconds of nothing reaches the model zero times. Whisper writes a
// sentence over silence, so this is the assertion that keeps a live transcript
// from filling with one.
auto tSilenceNeverReachesTheModel = test("Live/silenceNeverReachesTheModel") = []
{
    if (!canRun())
        return;

    auto live = LiveTranscriber {preparedModel()};
    const auto silence = silentChunk(sampleRate / 100);

    for (auto chunk = 0; chunk < 500; ++chunk)
    {
        live.push(silence);
        live.update();
    }

    check(live.stats().runs == 0);
    check(live.committed().size() == 0);
    check(live.pending().empty());
    check(!live.stats().pendingHasSpeech);
};

// The whole layer on real speech: eleven seconds of jfk.wav a block at a time,
// then two seconds of silence, and the sentence the one-shot runtime produces
// for the same recording comes out of committed() once.
auto tJfkStreamsIntoOneCommittedLine =
    test("Live/jfkStreamsIntoOneCommittedLine") = []
{
    if (!canRun())
        return;

    auto live = LiveTranscriber {preparedModel()};
    const auto samples = readWavFile(sampleFile(jfkSample));
    const auto trailingSilence = silentChunk(2 * sampleRate);

    const auto start = std::chrono::steady_clock::now();

    auto sawPending = pushChunked(live, samples, blockSamples);
    sawPending = pushChunked(live, trailingSilence, blockSamples) || sawPending;

    const auto elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();

    const auto chunks = chunkCount(samples.size(), blockSamples)
                        + chunkCount((int) trailingSilence.size(), blockSamples);

    check(live.committed().size() == 1);
    check(live.pending().empty());
    check(sawPending);
    check(live.stats().runs > 1);
    check(live.stats().runs < chunks);

    if (live.committed().size() != 1)
        return;

    check(trimmed(live.committed()[0]) == jfkTranscript);

    std::cout << "  jfk live: " << live.stats().runs << " runs over " << chunks
              << " chunks, " << elapsed << " s total, last run "
              << live.stats().lastRunSeconds << " s\n    \"" << live.committed()[0]
              << "\"\n";
};

// Thirty-three seconds is more than one window, and the length cut is what
// keeps a segment inside it whatever the audio does. A wider step keeps the
// test to a handful of runs; a longer hold than the recording takes silence out
// of the picture, so the cut is the only thing that can close a segment.
auto tLengthCutCommitsSeveralSegments =
    test("Live/lengthCutCommitsSeveralSegments") = []
{
    if (!canRun())
        return;

    auto options = LiveOptions {};
    options.maxSegmentSeconds = 12.0;
    options.silenceHoldSeconds = 1000.0;
    options.stepSeconds = 4.0;

    auto live = LiveTranscriber {preparedModel(), options};
    const auto samples = readWavFile(sampleFile(jfkSample));

    const auto trailingSilence = silentChunk(2 * sampleRate);

    for (auto pass = 0; pass < 3; ++pass)
        pushChunked(live, samples, blockSamples);

    pushChunked(live, trailingSilence, blockSamples);

    check(live.committed().size() >= 2);
    check(live.stats().pendingSeconds <= options.maxSegmentSeconds);

    for (const auto& line: live.committed())
        check(!trimmed(line).empty());

    std::cout << "  jfk x3 at a 12 s cut: " << live.committed().size()
              << " committed lines, " << live.stats().runs << " runs\n";

    for (const auto& line: live.committed())
        std::cout << "    \"" << line << "\"\n";
};

// A pause the speaker takes mid-sentence is audio the model has nothing to say
// about, and under an unconditional step it costs a run every stepSeconds until
// they carry on. minNewSpeechSeconds is what makes those runs not happen, and
// setting it to zero here is what the policy used to be, so the two drives are
// the before and the after of one change.
auto tAPauseInsideASegmentTakesNoRun =
    test("Live/aPauseInsideASegmentTakesNoRun") = []
{
    if (!canRun())
        return;

    const auto gated = driveAPause(LiveOptions {}.minNewSpeechSeconds);
    const auto unconditional = driveAPause(0.0);

    check(gated.overThePause == 0);
    check(unconditional.overThePause > 0);

    check(gated.overSpeech == unconditional.overSpeech);
    check(gated.total < unconditional.total);

    std::cout << "  a 0.9 s pause: " << gated.total << " runs, "
              << unconditional.total << " on an unconditional step\n";
};

// The other half of the same policy: the run that closes a segment is due on
// audio the last run did not cover, speech or not, since the silence after the
// last word is what Whisper writes that word against. So the hold is never
// re-run and always closed, and the transcript is the one the one-shot runtime
// produces either way.
auto tTheClosingRunStillSeesTheSilence =
    test("Live/theClosingRunStillSeesTheSilence") = []
{
    if (!canRun())
        return;

    const auto gated = driveJfkThenSilence(LiveOptions {}.minNewSpeechSeconds);
    const auto unconditional = driveJfkThenSilence(0.0);

    check(gated.committedLine == jfkTranscript);
    check(gated.committedLine == unconditional.committedLine);

    check(gated.overTheSilence >= 1);
    check(gated.overTheSilence < unconditional.overTheSilence);
    check(gated.overSpeech < unconditional.overSpeech);

    std::cout << "  jfk then its silence hold: " << gated.overSpeech << " + "
              << gated.overTheSilence << " runs, " << unconditional.overSpeech
              << " + " << unconditional.overTheSilence
              << " on an unconditional step\n";
};

// The Stop button: everything pushed and no silence after it still commits, and
// clear() puts the object back where it started.
auto tFlushCommitsAndClearEmpties = test("Live/flushCommitsAndClearEmpties") = []
{
    if (!canRun())
        return;

    auto live = LiveTranscriber {preparedModel()};
    const auto samples = readWavFile(sampleFile(jfkSample));

    live.push(samples);
    live.flush();

    check(live.committed().size() == 1);
    check(live.pending().empty());
    check(live.stats().runs == 1);

    if (live.committed().size() == 1)
        check(trimmed(live.committed()[0]) == jfkTranscript);

    live.clear();

    check(live.committed().size() == 0);
    check(live.pending().empty());
    check(live.stats().runs == 0);
    check(live.stats().pendingSeconds == 0.0);
    check(live.stats().speechSeconds == 0.0);
    check(!live.stats().pendingHasSpeech);
};

// The option this layer is the case for: every run encodes the segment it has
// rather than the 30 s window. What it must not change is the answer — the same
// eleven seconds of jfk.wav come out as the same committed line — and what it
// must change is the context the model was left holding, which is the closing
// segment's own length plus the margin.
//
// The runtime is shared by every test in this module, so the context goes back
// to the window at the end; a run that left it set would quietly transcribe
// every test after this one at a reduced context.
auto tLiveEncodesOnlyTheAudioThereIs = test("Live/encodesOnlyTheAudioThereIs") = []
{
    if (!canRun())
        return;

    auto options = LiveOptions {};
    options.encodeOnlyTheAudioThereIs = true;

    auto& whisper = preparedModel();
    auto live = LiveTranscriber {whisper, options};

    const auto samples = readWavFile(sampleFile(jfkSample));
    const auto trailingSilence = silentChunk(2 * sampleRate);

    pushChunked(live, samples, blockSamples);
    pushChunked(live, trailingSilence, blockSamples);

    const auto context = whisper.audioContext();
    whisper.setAudioContext(0);

    check(live.committed().size() == 1);

    if (live.committed().size() == 1)
        check(trimmed(live.committed()[0]) == jfkTranscript);

    // The segment that closed held the recording and the silence the hold
    // needed, and no more: eleven seconds of speech plus about a second of it,
    // which is a long way short of the window's 1500.
    check(context > Whisper::audioContextForSamples(samples.size(), 0.0));
    check(context < encoderPositions);

    std::cout << "  live at the segment's own length: " << context << " positions, "
              << live.stats().runs << " runs, last run "
              << live.stats().lastRunSeconds << " s\n";
};

// And with the option off — the default — the model is never told anything.
auto tLiveLeavesTheContextAloneByDefault =
    test("Live/leavesTheContextAloneByDefault") = []
{
    if (!canRun())
        return;

    auto& whisper = preparedModel();
    auto live = LiveTranscriber {whisper};

    const auto samples = readWavFile(sampleFile(jfkSample));

    live.push(samples);
    live.flush();

    check(whisper.audioContext() == 0);
};
