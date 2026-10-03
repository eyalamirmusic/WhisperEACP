#include "MainPanel.h"
#include "Session.h"

#include <eacp/Core/Core.h>
#include <eacp/Graphics/Graphics.h>

#include <cstdio>
#include <string>
#include <string_view>

// A microphone, a level meter and a transcript that grows as somebody talks.
//
// The whole loop is one timer on the message thread, and that is the design
// rather than a simplification: eacp's GPU layer is main-thread only and
// `Whisper::transcribe` blocks on its own commits, so a tick drains the queue
// `Capture`'s device callback filled, hands it to `LiveTranscriber`, and lets
// that decide whether a run is due. A tick that runs the model costs tens of
// milliseconds in Release and the window waits for it; the audio thread waits
// for nothing, the queue between the two being all it writes into.

namespace LiveTranscribe
{
namespace
{
constexpr auto usage =
    "usage: LiveTranscribe [--autostart] [--coreml] [--coreml-decoder] [--ane]\n"
    "                      [model directory]\n"
    "\n"
    "  model directory  a HuggingFace Whisper repo: config.json,\n"
    "                   preprocessor_config.json, model.safetensors and\n"
    "                   tokenizer.json. Left out, openai/whisper-tiny.en is\n"
    "                   fetched once and found on disk after that.\n"
    "  --autostart      open the default input as soon as the model is ready,\n"
    "                   and log a status line a second to stdout, so a run can\n"
    "                   be checked without a hand on the mouse.\n"
    "  --coreml         start with the encoder on Core ML's Neural Engine rather\n"
    "                   than the GPU kernels, where the machine can; the encoder\n"
    "                   box switches between the two at any time. The first\n"
    "                   switch on a machine compiles it for the engine, about\n"
    "                   14 s, with the window live throughout.\n"
    "  --coreml-decoder start with the decoder on Core ML (the CPU) rather than\n"
    "                   the GPU kernels; the decoder box switches it too.\n"
    "  --ane            start with everything on the Neural Engine: the encoder\n"
    "                   box on \"Core ML + mel\" and the decoder box on\n"
    "                   \"Core ML (engine)\", which leave the GPU idle.\n";

constexpr auto ticksPerSecond = 30;
constexpr auto ticksBetweenDeviceScans = ticksPerSecond * 5;

struct Arguments
{
    std::string modelDirectory;
    bool autostart = false;
    bool coreML = false;
    bool coreMLDecoder = false;
    bool allOnTheEngine = false;
    bool understood = true;
};

Arguments parseArguments(int argc, char* argv[])
{
    auto arguments = Arguments {};

    for (auto i = 1; i < argc; ++i)
    {
        const auto argument = std::string_view {argv[i]};

        if (argument == "--autostart")
            arguments.autostart = true;
        else if (argument == "--coreml")
            arguments.coreML = true;
        else if (argument == "--coreml-decoder")
            arguments.coreMLDecoder = true;
        else if (argument == "--ane")
            arguments.allOnTheEngine = true;
        else if (arguments.modelDirectory.empty() && !argument.starts_with("--"))
            arguments.modelDirectory = argv[i];
        else
            arguments.understood = false;
    }

    return arguments;
}

// At file scope so the App's members can be initialized from it, the way
// MakeASound's probe app carries its own switches.
Arguments arguments;

eacp::Graphics::WindowOptions makeWindowOptions()
{
    auto options = eacp::Graphics::WindowOptions {};

    options.width = 900;
    options.height = 640;
    options.minWidth = 720;
    options.minHeight = 420;
    options.title = "Live Transcribe";

    return options;
}

struct App
{
    App()
    {
        window.setContentView(panel);
        panel.refresh(true);
        panel.refreshTranscript();

        session.onModelSettled = [this] { modelSettled(); };

        // After the window is up, so it appears at once saying what it is doing
        // rather than half a second later already loaded — or, on a first run,
        // while 151 MB of weights arrives.
        eacp::Threads::callAsync([this] { session.start(); });
    }

    // The model is Ready or it Failed, and on a first run that is after a
    // download rather than after half a second of mapping.
    void modelSettled()
    {
        if (arguments.autostart)
        {
            logStartup();

            if (session.isReady())
                session.startCapture();
        }

        panel.refresh(true);
    }

    // What the window shows before anything is said: how long the model took,
    // and the inputs the box was filled from.
    void logStartup() const
    {
        std::printf(
            "%s, %.2f s\n", session.stateMessage().c_str(), session.loadSeconds());

        for (const auto& device: session.capture().inputDevices())
            std::printf("  %s %-40s %2d in   %s\n",
                        device.id == session.capture().deviceId() ? "->" : "  ",
                        device.name.c_str(),
                        device.inputChannels,
                        WSP::canCaptureNatively(device) ? "native 16 kHz"
                                                        : "resampled");

        std::fflush(stdout);
    }

    void tick()
    {
        ++ticks;

        const auto transcriptChanged = session.tick();
        const auto scanDevices =
            session.takeDeviceChange() || ticks % ticksBetweenDeviceScans == 0;

        panel.refresh(scanDevices);

        if (transcriptChanged)
            panel.refreshTranscript();

        if (arguments.autostart && ticks % ticksPerSecond == 0)
            logStatus();
    }

    void logStatus() const
    {
        const auto stream = session.capture().status();
        const auto level = session.capture().level();
        const auto live = session.liveStats();

        std::printf("%5.1fs  %-9s  %-13s  %-16s  %d Hz  block %d  blocks %lld  "
                    "peak %.4f  rms %.4f  runs %d  pending %.1fs  %s\n",
                    (double) ticks / ticksPerSecond,
                    session.isCapturing() ? "capturing" : "idle",
                    session.isSwitchingBackend()
                        ? "switching"
                        : backendName(session.encoderBackend()).c_str(),
                    backendName(session.decoderChoice()).c_str(),
                    stream.sampleRate,
                    stream.blockSize,
                    stream.blocks,
                    (double) level.peak,
                    (double) level.rms,
                    live.runs,
                    live.pendingSeconds,
                    session.pending().c_str());

        std::fflush(stdout);
    }

    Session session {arguments.modelDirectory,
                     arguments.coreML,
                     arguments.coreMLDecoder,
                     arguments.allOnTheEngine};

    MainPanel panel {session};
    eacp::Graphics::Window window {makeWindowOptions()};
    eacp::Threads::Timer timer {[this] { tick(); }, ticksPerSecond};

    int ticks = 0;
};
} // namespace
} // namespace LiveTranscribe

int main(int argc, char* argv[])
{
    eacp::Apps::setCommandLineArgs(argc, argv);

    LiveTranscribe::arguments = LiveTranscribe::parseArguments(argc, argv);

    if (!LiveTranscribe::arguments.understood)
    {
        std::printf("%s", LiveTranscribe::usage);
        return 2;
    }

    return eacp::Apps::run<LiveTranscribe::App>();
}
