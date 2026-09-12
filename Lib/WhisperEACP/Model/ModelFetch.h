#pragma once

#include <WhisperEACP/Core/Core.h>

#include <eacp/Network/Network.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace WSP
{
// The four files `Whisper::load` reads, fetched from HuggingFace the first time
// a binary here asks for them and found on disk every time after —
// `eacp::OnlineResource` per file, into one directory every binary in this tree
// shares.
//
// It used to be the build that downloaded them: CPM fetched the four at
// configure time and a POST_BUILD step copied 151 MB beside every binary that
// asked. That made a plain configure cost the download before anything
// compiled, put the weights in the build tree once per target, and left a
// binary that could only find a model where its own build had put one. A fetch
// at startup is the same four files, downloaded once per machine instead of
// once per build directory, and a binary that is moved keeps its model.
//
// Nothing here is Whisper-specific but the list: a repo with different files or
// a different revision is this class with another `resources()`.
class ModelFetch
{
public:
    using Freshness = eacp::OnlineResource::Freshness;

    static constexpr auto repository = "openai/whisper-tiny.en";

    // The revision the four URLs name, by commit rather than by `main`: the
    // bytes at a commit cannot change under a copy already on disk, which is
    // what makes `Freshness::trust` the honest default below, and it is the
    // revision every number in plan.md was measured against. It is also the
    // OnlineResource version, so bumping it re-downloads rather than trusting
    // the previous revision's copy.
    //
    // This is what replaced the four SHA256 hashes the configure used to pin.
    // OnlineResource has no content hash to check, so the integrity claim is
    // now the one HTTPS and an immutable URL make rather than one the project
    // checks itself — plan.md says what that costs.
    static constexpr auto revision = "87c7102498dcde7456f24cfd30239ca606ed9063";

    static constexpr int resourceCount = 4;

    // What a fetch of the four came to. `downloaded` is true when any of them
    // came off the network, which is what a second run must not report.
    struct Outcome
    {
        bool ok = false;
        bool cancelled = false;
        bool downloaded = false;

        // Which file failed and why, when it did.
        std::string error;

        // Where the four are, whether or not this fetch put them there.
        std::filesystem::path directory;
    };

    // How far a fetch of the four has got. There is no total across the set:
    // eacp reports a transfer, and only the server says how big a file is
    // before it arrives, so this is the file in flight plus its place in the
    // set. model.safetensors is 98% of the bytes, so its own fraction is the
    // only one worth drawing.
    struct Progress
    {
        // 1-based while a file is in flight, 0 before the first one starts.
        int fileIndex = 0;
        std::string fileName;

        eacp::OnlineResource::Progress file;
    };

    // "model.safetensors (4 of 4)   41.2 of 151.1 MB   27%" — the one sentence
    // every caller that shows a download shows, so the console apps and the
    // window say the same thing.
    static std::string progressText(const Progress& progress);

    using ProgressCallback = std::function<void(const Progress&)>;

    // One directory for every binary in this tree rather than each binary's
    // own. eacp's default is `FilePath::appSupportDirectory() / "Resources"`,
    // which names the folder after the running executable — right for an app
    // that ships, and wrong here, where Transcribe, Benchmark, the demo and
    // eight test binaries would each download a 151 MB copy of their own. So
    // the company and app names are spelled rather than discovered.
    static std::filesystem::path directory();

    // The four as eacp sees them, in the order a fetch takes them: the three
    // small files first, so a run with no network fails on a 2 kB transfer
    // rather than part-way through 151 MB.
    static Vector<eacp::OnlineResource::Info> resources();

    // All four are on disk at this revision. Local only — no request is made,
    // and nothing here distinguishes "never fetched" from "no network".
    static bool isAvailable();

    // Blocking, and main-thread only: it pumps the event loop, which is what a
    // console app's `main` may do and an event-loop callback may not — so call
    // it before `eacp::Apps::run`, not from inside it. onProgress is called on
    // the main thread as it goes.
    //
    // A failure is an Outcome rather than an exception: every caller here
    // either prints it and stops, or skips a test, and neither is an
    // exceptional path. `Whisper::load` is what throws about a bad model.
    static Outcome fetch(
        Freshness freshness = Freshness::trust,
        ProgressCallback onProgress = [](const Progress&) {});

    // The same four files for an app whose event loop is already running: the
    // transfers happen on eacp's worker thread, onFinished lands on the main
    // thread exactly once, and whatever draws the progress polls it on the
    // timer it already has. Destroying this cancels the transfer in flight and
    // hears nothing more from it.
    class Download
    {
    public:
        Download() = default;

        Download(const Download&) = delete;
        Download& operator=(const Download&) = delete;

        std::function<void(const Outcome&)> onFinished = [](const Outcome&) {};

        void start(Freshness freshness = Freshness::trust);
        void cancel();

        bool isRunning() const { return running; }
        Progress progress() const;

    private:
        void fetchNext();
        void fileFinished(const eacp::OnlineResource::Result& result);
        void finish();

        // Posted rather than called from inside a file's own continuation: a
        // trusted copy resolves the Async before start() returns, and the next
        // file's resource would then be constructed where this one is being
        // destroyed. The token is what keeps a posted step off a Download that
        // went away in the meantime.
        void postNextStep();

        Vector<eacp::OnlineResource::Info> queue;

        // Destroying it is what cancels a transfer in flight and abandons its
        // Async, so a Download that goes away mid-download is never called
        // back — which is why there is no destructor here doing it by hand.
        std::unique_ptr<eacp::OnlineResource> current;
        std::shared_ptr<int> lifetime = std::make_shared<int>(0);

        Outcome outcome;
        Freshness freshness = Freshness::trust;
        int index = 0;
        bool running = false;
    };
};
} // namespace WSP
