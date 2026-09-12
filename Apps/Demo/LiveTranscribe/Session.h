#pragma once

#include <WhisperEACP/WhisperEACP.h>

#include <functional>
#include <optional>
#include <string>

namespace LiveTranscribe
{
// Whether there is a model to run: the four files are arriving from
// HuggingFace, one is being mapped and its kernels compiled, one is running, or
// it failed and the message says why.
enum class ModelState
{
    Downloading,
    Loading,
    Ready,
    Failed
};

// Everything the window shows and nothing that draws it: the runtime, the
// microphone, and the live transcriber over the two.
//
// Main thread only, and that is not an accident of how it is used. eacp's GPU
// layer is main-thread only and `Whisper::transcribe` blocks on its own
// commits, so the loop is a timer tick: drain the queue the device callback
// filled, hand it to the transcriber, and let the transcriber decide whether a
// run is due. The audio thread reaches none of this — `Capture` is the whole of
// the boundary.
class Session
{
public:
    explicit Session(std::string modelDirectoryToUse);

    // Called once the state has settled on Ready or Failed, which is what the
    // app's autostart hangs off: the model arrives over the network now, so
    // "after loadModel()" is no longer a place in a function.
    std::function<void()> onModelSettled = [] {};

    // The four files if they are not on disk yet, then the model. Both halves
    // are on the message thread: the download is eacp::OnlineResource's worker
    // thread reporting back here, and the load is the half second of mapping
    // and kernel compilation this posts from the app's constructor so the
    // window is up before it starts.
    void start();

    ModelState state() const { return modelState; }

    // What the status line says while the four files arrive — the file in
    // flight, its place in the set and how far through it the transfer is.
    std::string downloadText() const;
    const std::string& stateMessage() const { return message; }
    bool isReady() const { return modelState == ModelState::Ready; }
    double loadSeconds() const { return modelLoadSeconds; }

    WSP::Capture& capture() { return microphone; }
    const WSP::Capture& capture() const { return microphone; }

    bool startCapture();
    void stopCapture();
    bool isCapturing() const { return microphone.isRunning(); }

    // Drains the microphone into the transcriber and runs whatever the policy
    // says is due. True when the transcript changed.
    bool tick();

    void clearTranscript();

    // True once for each run of device notifications drained since the last
    // ask: the machine's input list is worth rebuilding.
    bool takeDeviceChange();

    const WSP::Vector<std::string>& committed() const;
    const std::string& pending() const;
    WSP::LiveStats liveStats() const;

    double lastEncodeSeconds() const { return whisper.lastEncodeSeconds(); }
    double lastDecodeSeconds() const { return whisper.lastDecodeSeconds(); }

    const std::string& lastError() const { return errorText; }

private:
    void loadModel();
    void failWith(std::string reason);
    bool runTranscriber();

    std::string modelDirectory;
    WSP::ModelFetch::Download download;
    ModelState modelState = ModelState::Loading;
    std::string message;
    std::string errorText;
    double modelLoadSeconds = 0.0;

    WSP::Whisper whisper;
    WSP::Capture microphone;

    // Built once the model is prepared, since it holds a reference to a runtime
    // that can transcribe.
    std::optional<WSP::LiveTranscriber> live;

    WSP::Vector<float> captured;
    bool deviceChanged = false;
};
} // namespace LiveTranscribe
