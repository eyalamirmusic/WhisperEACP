#include "Session.h"

#include <eacp/GPU/GPU.h>

#include <chrono>
#include <exception>
#include <utility>

namespace LiveTranscribe
{
namespace
{
double secondsSince(std::chrono::steady_clock::time_point start)
{
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double>(elapsed).count();
}
} // namespace

bool DecoderChoice::operator==(const DecoderChoice& other) const
{
    return backend == other.backend
           && (backend == WSP::DecoderBackend::kernels || units == other.units);
}

std::string backendName(WSP::EncoderBackend backend)
{
    switch (backend)
    {
        case WSP::EncoderBackend::kernels:
            return "GPU (Metal)";
        case WSP::EncoderBackend::coreML:
            return "Core ML";
        case WSP::EncoderBackend::coreMLWithMel:
            return "Core ML + mel";
    }

    return "?";
}

std::string backendName(const DecoderChoice& choice)
{
    if (choice.backend == WSP::DecoderBackend::kernels)
        return "GPU (Metal)";

    return choice.units == WSP::DecoderComputeUnits::cpuAndNeuralEngine
               ? "Core ML (engine)"
               : "Core ML (CPU)";
}

Session::Session(std::string modelDirectoryToUse,
                 WSP::EncoderBackend encoderAtStart,
                 const DecoderChoice& decoderAtStart)
    : modelDirectory(std::move(modelDirectoryToUse))
{
    message = "loading model...";

    if (encoderAtStart != WSP::EncoderBackend::kernels)
        requestedBackend = encoderAtStart;

    if (decoderAtStart != DecoderChoice {})
        requestedDecoder = decoderAtStart;
}

DecoderChoice Session::decoderChoice() const
{
    return {whisper.decoderBackend(), whisper.decoderComputeUnits()};
}

// A directory on the command line is already on disk; anything else is the
// fetch, and on a first run that is 151 MB before there is a model to map.
// Either way the window is already up when this starts.
void Session::start()
{
    if (!modelDirectory.empty() || WSP::ModelFetch::isAvailable())
    {
        loadModel();

        return;
    }

    modelState = ModelState::Downloading;
    message = "downloading model...";

    download.onFinished = [this](const WSP::ModelFetch::Outcome& outcome)
    {
        if (!outcome.ok)
        {
            failWith("the model could not be fetched: " + outcome.error);

            return;
        }

        modelState = ModelState::Loading;
        loadModel();
    };

    download.start();
}

std::string Session::downloadText() const
{
    return WSP::ModelFetch::progressText(download.progress());
}

void Session::loadModel()
{
    if (modelState != ModelState::Loading)
        return;

    if (!eacp::GPU::Device::shared().isValid())
    {
        failWith("no GPU device available - nothing here can run");
        return;
    }

    const auto start = std::chrono::steady_clock::now();

    try
    {
        whisper.load(modelDirectory.empty() ? WSP::ModelFetch::directory().string()
                                            : modelDirectory);
        whisper.prepare();
    }
    catch (const std::exception& failure)
    {
        failWith(failure.what());
        return;
    }

    // The one place in the tree that turns the audio context on. The library
    // default is the whole 30 s window; this app re-transcribes an open
    // segment every half second and mostly against a segment far shorter than
    // that, so it encodes what the segment holds instead — 5.8 ms a run
    // against 14 ms over jfk.wav at a live pace, with the same transcript.
    auto options = WSP::LiveOptions {};
    options.encodeOnlyTheAudioThereIs = true;

    live.emplace(whisper, options);

    modelLoadSeconds = secondsSince(start);
    modelState = ModelState::Ready;
    message = "model ready";

    onModelSettled();
}

void Session::failWith(std::string reason)
{
    modelState = ModelState::Failed;
    message = std::move(reason);

    onModelSettled();
}

bool Session::startCapture()
{
    const auto error = microphone.start();

    if (error == MakeASound::Error::NoError)
    {
        errorText.clear();
        return true;
    }

    errorText = MakeASound::getErrorMessage(error);

    return false;
}

// The final run before the stream goes away: whatever the open segment holds is
// worth a transcript, and the model is exactly as available here as it is in a
// tick.
void Session::stopCapture()
{
    microphone.stop();

    if (!live.has_value())
        return;

    if (hasSwitchWaiting())
    {
        flushWhenSwitched = true;
        return;
    }

    flushTranscriber();
}

void Session::flushTranscriber()
{
    try
    {
        live->flush();
    }
    catch (const std::exception& failure)
    {
        errorText = failure.what();
    }
}

bool Session::tick()
{
    for (auto notification: microphone.drainNotifications())
    {
        deviceChanged = true;

        if (notification == MakeASound::DeviceNotification::Stopped)
            errorText = "the device stopped on its own";
    }

    captured.clear();
    microphone.drain(captured);

    if (!live.has_value())
        return false;

    if (!captured.empty())
        live->push(captured);

    const auto settled = std::exchange(switchSettled, false);

    if (hasSwitchWaiting())
        return advanceSwitch() || settled;

    return runTranscriber() || settled;
}

bool Session::hasModel() const
{
    return modelState == ModelState::Ready
           || modelState == ModelState::SwitchingBackend;
}

WSP::EncoderBackend Session::wantedEncoderBackend() const
{
    if (modelState == ModelState::SwitchingBackend && switchingHalf == Half::encoder)
        return switchTarget;

    return requestedBackend.value_or(whisper.encoderBackend());
}

DecoderChoice Session::wantedDecoderChoice() const
{
    if (modelState == ModelState::SwitchingBackend && switchingHalf == Half::decoder)
        return decoderTarget;

    return requestedDecoder.value_or(decoderChoice());
}

bool Session::canUseCoreML()
{
    return WSP::Whisper::supportsEncoderBackend(WSP::EncoderBackend::coreML);
}

bool Session::canDecodeOnCoreML()
{
    return WSP::Whisper::supportsDecoderBackend(WSP::DecoderBackend::coreML);
}

void Session::requestEncoderBackend(WSP::EncoderBackend backend)
{
    if (modelState == ModelState::SwitchingBackend)
        return;

    requestedBackend = backend;
}

void Session::requestDecoderChoice(const DecoderChoice& choice)
{
    if (modelState == ModelState::SwitchingBackend)
        return;

    requestedDecoder = choice;
}

bool Session::isSwitchingBackend() const
{
    return modelState == ModelState::SwitchingBackend;
}

bool Session::hasSwitchWaiting() const
{
    return requestedBackend.has_value() || requestedDecoder.has_value()
           || modelState == ModelState::SwitchingBackend;
}

// A run in flight is left to land, and update() is still called for it since
// that starts nothing while one is in the air; the switch begins on the first
// tick with nothing running, so the transcript it was working on is kept.
bool Session::advanceSwitch()
{
    if (modelState == ModelState::SwitchingBackend)
    {
        if (!switchIssued)
            issueSwitch();

        return false;
    }

    if (modelState != ModelState::Ready)
        return false;

    if (live->isRunning())
        return runTranscriber();

    if (!whisper.isTranscribing())
        beginSwitch();

    return false;
}

// The encoder's request first; a decoder request waits for the tick after.
void Session::beginSwitch()
{
    if (requestedBackend.has_value())
    {
        switchingHalf = Half::encoder;
        switchTarget = *std::exchange(requestedBackend, std::nullopt);

        if (switchTarget == whisper.encoderBackend())
            return;
    }
    else
    {
        switchingHalf = Half::decoder;
        decoderTarget = *std::exchange(requestedDecoder, std::nullopt);

        if (decoderTarget == decoderChoice())
            return;
    }

    const auto toCoreML = switchingHalf == Half::encoder
                              ? WSP::isOnCoreML(switchTarget)
                              : decoderTarget.backend == WSP::DecoderBackend::coreML;
    const auto half = switchingHalf == Half::encoder ? "encoder" : "decoder";

    modelState = ModelState::SwitchingBackend;
    switchIssued = false;
    message = toCoreML ? std::string {"compiling "} + half + " for Core ML..."
                       : std::string {"switching "} + half + " to GPU...";
}

void Session::issueSwitch()
{
    switchIssued = true;

    const auto settled = [this] { finishSwitch({}); };
    const auto failed = [this](const std::string& error) { finishSwitch(error); };

    try
    {
        if (switchingHalf == Half::encoder)
            whisper.switchEncoderBackendAsync(switchTarget).then(settled, failed);
        else
            whisper
                .switchDecoderBackendAsync(decoderTarget.backend,
                                           decoderTarget.units)
                .then(settled, failed);
    }
    catch (const std::exception& failure)
    {
        finishSwitch(failure.what());
    }
}

std::string Session::halfDescription(std::string_view where) const
{
    if (switchingHalf == Half::encoder)
        return "encoder " + std::string {where} + " "
               + backendName(whisper.encoderBackend());

    return "decoder " + std::string {where} + " " + backendName(decoderChoice());
}

void Session::finishSwitch(const std::string& error)
{
    modelState = ModelState::Ready;
    switchSettled = true;
    message = halfDescription("on");

    if (error.empty())
        errorText.clear();
    else
        errorText = "the " + halfDescription("stayed on") + ": " + error;

    if (!hasSwitchWaiting() && std::exchange(flushWhenSwitched, false))
        flushTranscriber();
}

// A model failure must not escape the timer: the window carries on with the
// transcript it has and the message beside it, and the stream stops rather than
// filling a queue nothing is draining.
bool Session::runTranscriber()
{
    try
    {
        return live->update();
    }
    catch (const std::exception& failure)
    {
        errorText = failure.what();
        microphone.stop();

        return false;
    }
}

void Session::clearTranscript()
{
    if (live.has_value())
        live->clear();

    errorText.clear();
}

bool Session::takeDeviceChange()
{
    return std::exchange(deviceChanged, false);
}

const WSP::Vector<std::string>& Session::committed() const
{
    static const auto nothing = WSP::Vector<std::string> {};

    return live.has_value() ? live->committed() : nothing;
}

const std::string& Session::pending() const
{
    static const auto nothing = std::string {};

    return live.has_value() ? live->pending() : nothing;
}

WSP::LiveStats Session::liveStats() const
{
    return live.has_value() ? live->stats() : WSP::LiveStats {};
}
} // namespace LiveTranscribe
