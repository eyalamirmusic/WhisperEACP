#pragma once

#include <WhisperEACP/Audio/Format.h>
#include <WhisperEACP/Core/Core.h>

#include <MakeASound/MakeASound.h>

#include <atomic>
#include <cstddef>
#include <optional>

namespace WSP
{
// What a meter reads: the loudest absolute sample and the RMS of the last block
// the device delivered, linear (1.0 = full scale).
struct CaptureLevel
{
    float peak = 0.f;
    float rms = 0.f;
};

// The stream as it is actually running, read on the main thread.
struct CaptureStatus
{
    bool running = false;
    int sampleRate = 0;
    int blockSize = 0;
    int streamChannels = 0;
    long long blocks = 0;
    int overflows = 0;
    int dropped = 0;
    bool native = false;
    MakeASound::Error lastError = MakeASound::Error::NoError;
};

// Everything the device callback computes, as free functions: a machine with no
// microphone can still check the arithmetic that feeds the model.

// Averages every channel of a planar block into out, one sample per frame.
void mixToMono(const MakeASound::Buffer& block, Span<float> out);

CaptureLevel measureLevel(Span<const float> samples);

// A microphone opened at Whisper's own rate and mixed to mono, handed to
// whoever calls drain() through a queue the device callback never blocks on.
class Capture
{
public:
    Capture();
    ~Capture();
    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;

    MakeASound::DeviceManager& deviceManager();

    // Devices with at least one input channel, from the manager's enumeration.
    Vector<MakeASound::DeviceInfo> inputDevices() const;

    const MakeASound::StreamConfig& config() const;
    int deviceId() const;
    int firstChannel() const;
    int channelCount() const;

    // Which device and which slice of its channels feed the model. Both apply at
    // once when the stream is running (re-open), and are remembered otherwise.
    // An id that names no input device returns INVALID_DEVICE and changes nothing.
    MakeASound::Error setDevice(int id);
    MakeASound::Error setChannels(int first, int count);

    MakeASound::Error start();
    void stop();
    bool isRunning() const;

    // Main thread. Appends everything the device delivered since the last call,
    // mono at 16 kHz, to `out`; returns how many samples were appended.
    int drain(Vector<float>& out);

    CaptureLevel level() const;
    CaptureStatus status() const;

    // MakeASound's queued notifications, drained on the caller's thread.
    // setNotificationCallback is deliberately unused: it fires on an OS audio
    // thread that may already hold the device, so calling back into the manager
    // from it can deadlock.
    Vector<MakeASound::DeviceNotification> drainNotifications();

private:
    void audioCallback(MakeASound::AudioCallbackInfo& info);
    MakeASound::Error applySelection();
    void prepareForStream();
    void rememberWhetherRateIsNative();
    void discardQueued();

    // Eight seconds at 16 kHz: long enough that a main thread busy with a
    // transcription pass loses nothing while it is away.
    static constexpr auto queueCapacity = std::size_t {1} << 17;

    // The mono block is never allocated on the audio thread, so it is sized once
    // for anything a device could plausibly deliver; a larger block is dropped
    // and counted rather than reaching for the allocator.
    static constexpr auto scratchCapacity = 8192;

    std::optional<MakeASound::DeviceManager> manager;
    MakeASound::StreamConfig streamConfig;
    MakeASound::Error lastError = MakeASound::Error::NoError;

    Vector<float> monoBlock;
    MakeASound::SPSCQueue<float, queueCapacity> queued;

    std::atomic<float> peakLevel {0.f};
    std::atomic<float> rmsLevel {0.f};
    std::atomic<int> deliveredRate {0};
    std::atomic<int> deliveredBlockSize {0};
    std::atomic<int> deliveredChannels {0};
    std::atomic<long long> blocksDelivered {0};
    std::atomic<int> overflowBlocks {0};
    std::atomic<int> droppedSamples {0};
    std::atomic<bool> nativeRate {false};
};
} // namespace WSP
