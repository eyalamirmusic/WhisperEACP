#pragma once

#include <WhisperEACP/WhisperEACP.h>

#include <optional>
#include <string>

namespace LiveTranscribe
{
// Whether there is a model to run: none was copied beside the binary and none
// was named, one is being loaded, one is running, or loading it failed and the
// message says why.
enum class ModelState
{
    NotBundled,
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

    // Maps the weights, compiles every kernel and uploads them: about half a
    // second, so the app posts this rather than calling it from its
    // constructor. The window is up, saying it is loading, before it starts.
    void loadModel();

    ModelState state() const { return modelState; }
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
    void failWith(std::string reason);
    bool runTranscriber();

    std::string modelDirectory;
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
