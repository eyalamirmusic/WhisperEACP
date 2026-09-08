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
