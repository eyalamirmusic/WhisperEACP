#include "Capture.h"

#include <algorithm>
#include <cmath>

namespace WSP
{
void mixToMono(const MakeASound::Buffer& block, Span<float> out)
{
    const auto channels = block.getNumChannels();

    if (channels <= 0)
    {
        std::fill(out.begin(), out.end(), 0.f);
        return;
    }

    const auto frames = std::min(out.size(), block.getNumSamples());
    const auto scale = 1.f / (float) channels;

    for (auto frame = 0; frame < frames; ++frame)
    {
        auto sum = 0.f;

        for (auto channel = 0; channel < channels; ++channel)
            sum += block.getChannel(channel)[frame];

        out[frame] = sum * scale;
    }
}

CaptureLevel measureLevel(Span<const float> samples)
{
    auto level = CaptureLevel {};

    if (samples.empty())
        return level;

    auto sumOfSquares = 0.0;

    for (auto sample: samples)
    {
        level.peak = std::max(level.peak, std::abs(sample));
        sumOfSquares += (double) sample * (double) sample;
    }

    level.rms = (float) std::sqrt(sumOfSquares / (double) samples.getSize());

    return level;
}

Capture::Capture()
{
    manager.emplace();

    streamConfig = manager->getDefaultInputConfig();

    // Whatever the device runs at, miniaudio converts to what was asked for, so
    // the callback delivers Whisper's rate on any machine and the front-end never
    // resamples. canCaptureNatively() only says whether that conversion happens.
    streamConfig.sampleRate = sampleRate;

    monoBlock.resize(scratchCapacity);
    rememberWhetherRateIsNative();
}

Capture::~Capture()
{
    stop();
}

MakeASound::DeviceManager& Capture::deviceManager()
{
    return *manager;
}

Vector<MakeASound::DeviceInfo> Capture::inputDevices() const
{
    auto devices = Vector<MakeASound::DeviceInfo> {};

    for (const auto& device: manager->getDevices())
        if (device.hasChannels(true))
            devices.add(device);

    return devices;
}

const MakeASound::StreamConfig& Capture::config() const
{
    return streamConfig;
}

int Capture::deviceId() const
{
    return streamConfig.input.has_value() ? streamConfig.input->device.id : -1;
}

int Capture::firstChannel() const
{
    return streamConfig.input.has_value() ? streamConfig.input->firstChannel : 0;
}

int Capture::channelCount() const
{
    return streamConfig.input.has_value() ? streamConfig.input->nChannels : 0;
}

MakeASound::Error Capture::setDevice(int id)
{
    for (const auto& device: manager->getDevices())
    {
        if (device.id != id || !device.hasChannels(true))
            continue;

        streamConfig.input = MakeASound::StreamParameters {device, true};

        return applySelection();
    }

    lastError = MakeASound::Error::INVALID_DEVICE;

    return lastError;
}

MakeASound::Error Capture::setChannels(int first, int count)
{
    if (!streamConfig.input.has_value())
    {
        lastError = MakeASound::Error::INVALID_DEVICE;
        return lastError;
    }

    if (first < 0 || count <= 0
        || first + count > streamConfig.input->device.inputChannels)
    {
        lastError = MakeASound::Error::INVALID_PARAMETER;
        return lastError;
    }

    streamConfig.input->firstChannel = first;
    streamConfig.input->nChannels = count;

    return applySelection();
}

MakeASound::Error Capture::start()
{
    if (!streamConfig.input.has_value())
    {
        lastError = MakeASound::Error::NO_DEVICES_FOUND;
        return lastError;
    }

    manager->stop();
    prepareForStream();

    lastError =
        manager->start(streamConfig, [this](auto& info) { audioCallback(info); });

    return lastError;
}

void Capture::stop()
{
    manager->stop();
}

bool Capture::isRunning() const
{
    return manager->isRunning();
}

int Capture::drain(Vector<float>& out)
{
    auto appended = 0;
    auto sample = 0.f;

    while (queued.pop(sample))
    {
        out.add(sample);
        ++appended;
    }

    return appended;
}

CaptureLevel Capture::level() const
{
    return {peakLevel.load(std::memory_order_relaxed),
            rmsLevel.load(std::memory_order_relaxed)};
}

CaptureStatus Capture::status() const
{
    auto current = CaptureStatus {};

    current.running = manager->isRunning();
    current.sampleRate = deliveredRate.load(std::memory_order_relaxed);
    current.blockSize = deliveredBlockSize.load(std::memory_order_relaxed);
    current.streamChannels = deliveredChannels.load(std::memory_order_relaxed);
    current.blocks = blocksDelivered.load(std::memory_order_relaxed);
    current.overflows = overflowBlocks.load(std::memory_order_relaxed);
    current.dropped = droppedSamples.load(std::memory_order_relaxed);
    current.native = nativeRate.load(std::memory_order_relaxed);
    current.lastError = lastError;

    return current;
}

Vector<MakeASound::DeviceNotification> Capture::drainNotifications()
{
    return manager->drainNotifications();
}

// setConfig() stops the stream before it re-opens, so the scratch is only ever
// resized while nothing is delivering into it.
MakeASound::Error Capture::applySelection()
{
    rememberWhetherRateIsNative();

    if (!manager->isRunning())
    {
        lastError = MakeASound::Error::NoError;
        return lastError;
    }

    manager->stop();
    prepareForStream();

    lastError = manager->setConfig(streamConfig);

    return lastError;
}

void Capture::prepareForStream()
{
    monoBlock.resize(std::max(streamConfig.maxBlockSize, scratchCapacity));

    rememberWhetherRateIsNative();
    discardQueued();

    peakLevel.store(0.f, std::memory_order_relaxed);
    rmsLevel.store(0.f, std::memory_order_relaxed);
    blocksDelivered.store(0, std::memory_order_relaxed);
    overflowBlocks.store(0, std::memory_order_relaxed);
    droppedSamples.store(0, std::memory_order_relaxed);
}

void Capture::rememberWhetherRateIsNative()
{
    auto native = streamConfig.input.has_value()
                  && canCaptureNatively(streamConfig.input->device);

    nativeRate.store(native, std::memory_order_relaxed);
}

void Capture::discardQueued()
{
    auto sample = 0.f;

    while (queued.pop(sample))
        ;
}

void Capture::audioCallback(MakeASound::AudioCallbackInfo& info)
{
    deliveredRate.store(info.sampleRate, std::memory_order_relaxed);
    deliveredBlockSize.store(info.numSamples, std::memory_order_relaxed);
    deliveredChannels.store(info.numInputs, std::memory_order_relaxed);
    blocksDelivered.fetch_add(1, std::memory_order_relaxed);

    if (info.status == MakeASound::AudioCallbackStatus::InputOverflow)
        overflowBlocks.fetch_add(1, std::memory_order_relaxed);

    const auto block = info.getInput();

    if (info.numSamples <= 0 || block.getNumChannels() <= 0)
        return;

    if (info.numSamples > monoBlock.size())
    {
        droppedSamples.fetch_add(info.numSamples, std::memory_order_relaxed);
        return;
    }

    const auto mono = Span<float> {monoBlock}.first(info.numSamples);

    mixToMono(block, mono);

    const auto measured = measureLevel(mono);

    peakLevel.store(measured.peak, std::memory_order_relaxed);
    rmsLevel.store(measured.rms, std::memory_order_relaxed);

    auto dropped = 0;

    for (auto sample: mono)
        if (!queued.push(sample))
            ++dropped;

    if (dropped > 0)
        droppedSamples.fetch_add(dropped, std::memory_order_relaxed);
}
} // namespace WSP
