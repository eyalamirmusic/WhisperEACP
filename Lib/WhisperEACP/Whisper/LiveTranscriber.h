#pragma once

#include <WhisperEACP/Whisper/Whisper.h>

#include <string>

namespace WSP
{
// The policy, in seconds of audio. Every clock here is audio time (samples
// pushed), never wall time, so a run is deterministic for a given input.
struct LiveOptions
{
    // Re-run the open segment once this much new audio arrived.
    double stepSeconds = 0.5;

    // Close the segment at this length whatever the audio does, which has to
    // stay inside the 30 s window and is clamped to it.
    double maxSegmentSeconds = 25.0;

    // Speech, then this much continuous silence, closes the segment.
    double silenceHoldSeconds = 1.0;

    // A 100 ms block whose RMS (dBFS, 1.0 = 0 dB) is above this is speech.
    float speechThresholdDb = -40.f;

    // A segment with less speech than this is never sent to the model.
    double minSpeechSeconds = 0.3;

    // Silence kept before the first speech block; earlier silence is dropped.
    double leadInSeconds = 0.3;
};

struct LiveStats
{
    // Whisper::transcribe calls so far, and the wall clock of the last one.
    int runs = 0;
    double lastRunSeconds = 0.0;

    // Audio in the open segment, and how much of it was classified as speech.
    double pendingSeconds = 0.0;
    double speechSeconds = 0.0;
    bool pendingHasSpeech = false;
};

// 16 kHz mono samples in, a growing transcript out: the open segment is
// re-transcribed as audio arrives, and closed into `committed()` on silence or
// on length. Main thread only, like the Whisper it drives.
class LiveTranscriber
{
public:
    explicit LiveTranscriber(Whisper& whisper, LiveOptions options = {});

    // Any block size. Buffers only; no GPU work happens here.
    void push(Span<const float> samples);

    // Runs the model if a run is due and closes the segment if the policy says
    // so. Returns true when committed() or pending() changed.
    bool update();

    // Closes the open segment now (the Stop button): a final run if it holds
    // speech, then commit.
    void flush();

    // Drops everything: segments, pending audio, stats.
    void clear();

    const Vector<std::string>& committed() const { return committedText; }
    const std::string& pending() const { return pendingText; }
    const LiveOptions& options() const { return liveOptions; }
    LiveStats stats() const;

private:
    bool takeNextBlock();
    void takeRemainingSamples();
    void appendBlock(Span<const float> block);
    void dropSilenceBeforeTheLeadIn();

    bool segmentIsFull() const;
    bool silenceClosesTheSegment() const;
    bool hasAudioWorthTranscribing() const;
    bool runIsDue() const;

    bool runModel();
    bool closeSegment();
    void startSegment();

    int maximumSegmentSamples() const;

    Whisper& model;
    LiveOptions liveOptions;

    // What push() took and no block has claimed yet, and the open segment the
    // blocks go into.
    Vector<float> incoming;
    Vector<float> segment;

    int speechSamples = 0;
    int trailingSilentBlocks = 0;

    // How much of the segment the last run saw, which is what makes a closing
    // run skippable and what the step is measured against.
    int coveredSamples = 0;
    bool segmentHasSpeech = false;
    bool hasRunThisSegment = false;

    Vector<std::string> committedText;
    std::string pendingText;

    int runCount = 0;
    double lastRun = 0.0;
};

// The classifier, exposed for the tests: RMS of a block in dBFS (silence is a
// large negative number, never -inf), and whether that counts as speech.
float blockLevelDb(Span<const float> block);
bool isSpeechBlock(Span<const float> block, float thresholdDb);
} // namespace WSP
