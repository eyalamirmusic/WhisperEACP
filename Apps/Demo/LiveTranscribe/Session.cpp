#include "Session.h"

#include <eacp/GPU/GPU.h>

#include <chrono>
#include <exception>
#include <utility>

namespace LiveTranscribe
{
namespace
{
constexpr auto missingBundledModel =
    "this build copied no model beside the binary. Configure with "
    "-DWHISPER_EACP_FETCH_MODEL=ON to get one, or name a HuggingFace Whisper "
    "repo on the command line.";

double secondsSince(std::chrono::steady_clock::time_point start)
{
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double>(elapsed).count();
}
} // namespace

Session::Session(std::string modelDirectoryToUse)
    : modelDirectory(std::move(modelDirectoryToUse))
{
    if (modelDirectory.empty() && !WSP::Whisper::hasBundledModel())
    {
        modelState = ModelState::NotBundled;
        message = missingBundledModel;
        return;
    }

    message = "loading model...";
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
        if (modelDirectory.empty())
            whisper.loadBundled();
        else
            whisper.load(modelDirectory);

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
}

void Session::failWith(std::string reason)
{
    modelState = ModelState::Failed;
    message = std::move(reason);
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

    return runTranscriber();
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
