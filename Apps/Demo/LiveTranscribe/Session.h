#pragma once

#include <WhisperEACP/WhisperEACP.h>

#include <functional>
#include <optional>
#include <string>

namespace LiveTranscribe
{
// Where the decoder runs: its backend, and on Core ML the units it is built
// for, the CPU or the Neural Engine. The units mean nothing on the kernels,
// and two choices of the kernels are the same choice whatever they say.
struct DecoderChoice
{
    WSP::DecoderBackend backend = WSP::DecoderBackend::kernels;
    WSP::DecoderComputeUnits units = WSP::DecoderComputeUnits::cpu;

    bool operator==(const DecoderChoice& other) const;
};

inline constexpr auto decoderOnTheCpu =
    DecoderChoice {WSP::DecoderBackend::coreML, WSP::DecoderComputeUnits::cpu};
inline constexpr auto decoderOnTheEngine = DecoderChoice {
    WSP::DecoderBackend::coreML, WSP::DecoderComputeUnits::cpuAndNeuralEngine};

std::string backendName(WSP::EncoderBackend backend);
std::string backendName(const DecoderChoice& choice);

// Whether there is a model to run: the four files are arriving from
// HuggingFace, one is being mapped and its kernels compiled, one is running,
// its encoder or its decoder is moving to the other backend, or it failed and
// the message says why.
enum class ModelState
{
    Downloading,
    Loading,
    Ready,
    SwitchingBackend,
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
    // encoderOnCoreML asks for the encoder on Core ML (the Neural Engine) from
    // the start, and decoderOnCoreML the decoder (the CPU): the model is
    // prepared on the kernels and switched as soon as it is Ready, exactly as
    // the two request calls would. allOnTheEngine asks for everything on the
    // Neural Engine instead, the mel with the encoder and the decoder too,
    // Whisper::setNeuralEngineThroughout() made as two switches.
    Session(std::string modelDirectoryToUse,
            bool encoderOnCoreML,
            bool decoderOnCoreML,
            bool allOnTheEngine);

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

    // Whether the model is up, whatever its encoder is doing: what Start
    // needs, since capture queues the audio through a switch.
    bool hasModel() const;

    // The encoder the model runs on now, and the one it will run on once a
    // requested switch has been made; the two differ only while one is
    // pending or in progress, and after a switch that failed they agree again
    // on the backend the model stayed on.
    WSP::EncoderBackend encoderBackend() const { return whisper.encoderBackend(); }
    WSP::EncoderBackend wantedEncoderBackend() const;

    static bool canUseCoreML();

    // The same pair for the decoder, which switches apart from the encoder.
    DecoderChoice decoderChoice() const;
    DecoderChoice wantedDecoderChoice() const;

    static bool canDecodeOnCoreML();

    // Remembered, and made on the first tick when no run is in flight; until
    // it is made and settled no run starts and the audio waits in the queue.
    // A request while a switch is in progress is ignored, and one before the
    // model is Ready is made once it is. A switch that fails leaves the model
    // Ready on the backend it was on, with lastError() saying why. With both
    // halves asked for, the encoder's switch is made first and the decoder's
    // on the tick after it settles.
    void requestEncoderBackend(WSP::EncoderBackend backend);
    void requestDecoderChoice(const DecoderChoice& choice);
    bool isSwitchingBackend() const;

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
    void flushTranscriber();

    enum class Half
    {
        encoder,
        decoder
    };

    bool hasSwitchWaiting() const;
    bool advanceSwitch();
    void beginSwitch();
    void issueSwitch();
    void finishSwitch(const std::string& error);
    std::string halfDescription(std::string_view where) const;

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

    // A switch is requested, then begun on a tick with no run in flight, and
    // issued on the tick after, so the window has drawn what it is waiting
    // for before a switch that blocks — the one to the kernels — holds it.
    std::optional<WSP::EncoderBackend> requestedBackend;
    std::optional<DecoderChoice> requestedDecoder;
    Half switchingHalf = Half::encoder;
    WSP::EncoderBackend switchTarget = WSP::EncoderBackend::kernels;
    DecoderChoice decoderTarget;
    bool switchIssued = false;
    bool flushWhenSwitched = false;
    bool switchSettled = false;

    WSP::Vector<float> captured;
    bool deviceChanged = false;
};
} // namespace LiveTranscribe
