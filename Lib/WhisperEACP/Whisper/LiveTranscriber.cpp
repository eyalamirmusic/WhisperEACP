#include "LiveTranscriber.h"

#include <WhisperEACP/Audio/Format.h>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace WSP
{
namespace
{
// 100 ms, which is the unit everything here is quantised to: a block is
// classified as a whole, the lead-in and the silence hold are counted in
// blocks, and a segment grows by one.
constexpr auto blockSamples = sampleRate / 10;

// What an all-zero block answers, so a level is always a number a comparison
// and a printf can both take.
constexpr auto silenceFloorDb = -120.0f;

int blocksForSeconds(double seconds)
{
    const auto samples = samplesForSeconds(seconds);
    return samples <= 0 ? 0 : (samples + blockSamples - 1) / blockSamples;
}

std::string withoutSurroundingSpace(const std::string& text)
{
    const auto first = text.find_first_not_of(" \t\r\n");

    if (first == std::string::npos)
        return {};

    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

double secondsForSamples(int samples)
{
    return (double) samples / sampleRate;
}
} // namespace

float blockLevelDb(Span<const float> block)
{
    if (block.empty())
        return silenceFloorDb;

    auto sumOfSquares = 0.0;

    for (auto sample: block)
        sumOfSquares += (double) sample * (double) sample;

    const auto rootMeanSquare = std::sqrt(sumOfSquares / block.size());

    if (rootMeanSquare <= 0.0)
        return silenceFloorDb;

    return std::max(silenceFloorDb, (float) (20.0 * std::log10(rootMeanSquare)));
}

bool isSpeechBlock(Span<const float> block, float thresholdDb)
{
    return blockLevelDb(block) > thresholdDb;
}

LiveTranscriber::LiveTranscriber(Whisper& whisper, LiveOptions optionsToUse)
    : model(whisper)
    , liveOptions(optionsToUse)
{
}

void LiveTranscriber::push(Span<const float> samples)
{
    for (auto sample: samples)
        incoming.add(sample);
}

// At most one model run a call, which is what makes this safe to drive from a
// 20-30 Hz timer: a segment that closes takes the run and leaves the rest of
// the incoming audio for the next call.
bool LiveTranscriber::update()
{
    while (takeNextBlock())
    {
        if (segmentIsFull() || silenceClosesTheSegment())
            return closeSegment();
    }

    return runIsDue() && runModel();
}

void LiveTranscriber::flush()
{
    while (takeNextBlock())
    {
        if (segmentIsFull())
            closeSegment();
    }

    takeRemainingSamples();
    closeSegment();
}

void LiveTranscriber::clear()
{
    incoming.clear();
    committedText.clear();
    pendingText.clear();
    startSegment();

    runCount = 0;
    lastRun = 0.0;
}

LiveStats LiveTranscriber::stats() const
{
    return {runCount,
            lastRun,
            secondsForSamples(segment.size()),
            secondsForSamples(speechSamples),
            segmentHasSpeech};
}

bool LiveTranscriber::takeNextBlock()
{
    if (incoming.size() < blockSamples)
        return false;

    appendBlock(Span<const float> {incoming.data(), blockSamples});
    incoming.erase(incoming.begin(), incoming.begin() + blockSamples);

    return true;
}

// The tail flush() cannot wait for a full block to claim. Shorter than 100 ms,
// so its level is measured over what there is.
void LiveTranscriber::takeRemainingSamples()
{
    if (incoming.size() == 0)
        return;

    appendBlock(incoming);
    incoming.clear();
}

void LiveTranscriber::appendBlock(Span<const float> block)
{
    const auto isSpeech = isSpeechBlock(block, liveOptions.speechThresholdDb);

    for (auto sample: block)
        segment.add(sample);

    if (isSpeech)
    {
        speechSamples += block.size();
        segmentHasSpeech = true;
        trailingSilentBlocks = 0;
        return;
    }

    ++trailingSilentBlocks;

    if (!segmentHasSpeech)
        dropSilenceBeforeTheLeadIn();
}

// Silence before the first speech block would otherwise eat the window a
// minute at a time. Only reachable while the segment holds nothing but
// silence, so no run has covered any of it and nothing but the samples has to
// move.
void LiveTranscriber::dropSilenceBeforeTheLeadIn()
{
    const auto keep = blocksForSeconds(liveOptions.leadInSeconds) * blockSamples;

    if (segment.size() <= keep)
        return;

    segment.erase(segment.begin(), segment.begin() + (segment.size() - keep));
    trailingSilentBlocks = segment.size() / blockSamples;
}

bool LiveTranscriber::segmentIsFull() const
{
    return segment.size() >= maximumSegmentSamples();
}

bool LiveTranscriber::silenceClosesTheSegment() const
{
    const auto hold = std::max(1, blocksForSeconds(liveOptions.silenceHoldSeconds));

    return segmentHasSpeech && trailingSilentBlocks >= hold;
}

// Whisper writes a sentence over silence, so a segment that holds too little
// speech never reaches it — and a segment the last run already saw whole has
// nothing new to say.
bool LiveTranscriber::hasAudioWorthTranscribing() const
{
    return speechSamples >= samplesForSeconds(liveOptions.minSpeechSeconds)
           && segment.size() > coveredSamples;
}

// The first run of a segment happens as soon as there is speech worth sending;
// every one after it wants both a step of new audio and some new speech in it.
// The segment that closes is the exception and does not come through here:
// closeSegment() runs on anything the last run did not cover, silence included,
// since that trailing silence is what lets Whisper finish the last word.
bool LiveTranscriber::runIsDue() const
{
    if (!hasAudioWorthTranscribing())
        return false;

    if (!hasRunThisSegment)
        return true;

    return segment.size() - coveredSamples
               >= samplesForSeconds(liveOptions.stepSeconds)
           && speechSamples - coveredSpeechSamples
                  >= samplesForSeconds(liveOptions.minNewSpeechSeconds);
}

// The context is set per run rather than once, because the segment grows: a
// run over 2 s of it encodes fewer positions than the run over 6 s that closes
// it, and the count each one wants is the one its own audio fills.
bool LiveTranscriber::runModel()
{
    if (liveOptions.encodeOnlyTheAudioThereIs)
        model.setAudioContext(Whisper::audioContextForSamples(
            segment.size(), liveOptions.audioContextMarginSeconds));

    const auto start = std::chrono::steady_clock::now();
    const auto tokens = model.transcribe(segment);
    const auto text = withoutSurroundingSpace(model.textForTokens(tokens));
    const auto elapsed = std::chrono::steady_clock::now() - start;

    lastRun = std::chrono::duration<double>(elapsed).count();
    ++runCount;
    coveredSamples = segment.size();
    coveredSpeechSamples = speechSamples;
    hasRunThisSegment = true;

    if (text == pendingText)
        return false;

    pendingText = text;

    return true;
}

bool LiveTranscriber::closeSegment()
{
    auto changed = hasAudioWorthTranscribing() && runModel();

    if (!pendingText.empty())
    {
        committedText.add(pendingText);
        pendingText.clear();
        changed = true;
    }

    startSegment();

    return changed;
}

void LiveTranscriber::startSegment()
{
    segment.clear();
    speechSamples = 0;
    trailingSilentBlocks = 0;
    coveredSamples = 0;
    coveredSpeechSamples = 0;
    segmentHasSpeech = false;
    hasRunThisSegment = false;
}

// A block below the window, so the block that fills a segment can never carry
// it past what Whisper::transcribe takes.
int LiveTranscriber::maximumSegmentSamples() const
{
    const auto wanted = samplesForSeconds(liveOptions.maxSegmentSeconds);

    return std::clamp(wanted, blockSamples, windowSamples - blockSamples);
}
} // namespace WSP
