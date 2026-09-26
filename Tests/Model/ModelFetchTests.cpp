#include "../Support/TestModel.h"
#include "Common.h"

#include <eacp/Core/Utils/FilePath.h>

#include <filesystem>
#include <string>

// The fetch that replaced the configure-time download: where it puts the four
// files, what it asks the server for, and that a second run of it moves no bytes.
//
// The fetch itself ran before this suite opened — Tests/Support/ModelTestMain.cpp
// says why it has to — so what these read is the record it left behind. The
// tests that read the files it fetched are in TinyEnTests.cpp next door.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;

namespace
{
constexpr const char* expectedFileNames[] = {"config.json",
                                             "preprocessor_config.json",
                                             "tokenizer.json",
                                             "model.safetensors"};

bool mentions(const std::string& text, std::string_view part)
{
    return text.find(part) != std::string::npos;
}

std::string urlPrefix()
{
    return std::string {"https://huggingface.co/"} + ModelFetch::repository
           + "/resolve/" + ModelFetch::revision + "/";
}
} // namespace

// The directory is the tree's, not the binary's. eacp's default resource
// directory is named after the running executable, and that is the one thing
// this tree cannot use: eleven binaries would mean eleven copies of 151 MB. So
// the per-binary folder must not be a prefix of where the model goes.
auto tDirectoryIsSharedByEveryBinary = test("ModelFetch/directoryIsShared") = []
{
    const auto directory = ModelFetch::directory();

    check(!directory.empty());
    check(directory.is_absolute());
    check(directory == ModelFetch::directory());
    check(mentions(directory.string(), eacp::FilePath::appDataDirectory().str()));
    check(mentions(directory.string(), "WhisperEACP"));

    // The per-binary folder is not an ancestor of it: that one is named after
    // the running executable, so ModelTests would hold a copy of its own.
    const auto perBinary = eacp::FilePath::appSupportDirectory().str();

    check(!perBinary.empty());
    check(directory.string().rfind(perBinary, 0) != 0);
};

// Four files, each at the pinned revision rather than at main, and each with the
// revision as its version — which is what makes a bump here re-download instead
// of trusting the previous revision's copy.
auto tResourcesNameTheFourFiles = test("ModelFetch/resourcesNameTheFourFiles") = []
{
    const auto resources = ModelFetch::resources();

    check(resources.size() == ModelFetch::resourceCount);

    for (auto index = 0; index < resources.size(); ++index)
    {
        const auto& resource = resources[index];

        check(resource.fileName == expectedFileNames[index]);
        check(resource.url == urlPrefix() + resource.fileName);
        check(resource.version == ModelFetch::revision);
        check(mentions(resource.name, ModelFetch::repository));
    }
};

// The weights last: a machine with no network fails on a 2 kB transfer rather
// than part-way through 151 MB, and the three small files are what a test of the
// config or the vocabulary needs anyway.
auto tWeightsAreFetchedLast = test("ModelFetch/weightsAreFetchedLast") = []
{
    const auto resources = ModelFetch::resources();

    check(resources[resources.size() - 1].fileName == "model.safetensors");
};

// What availability means: the four files in the directory, at this revision.
// Asserted one way only — a file on disk that eacp has no sidecar for is not
// available, which is the state a hand-copied directory is in.
auto tAvailabilityMeansTheFilesAreThere =
    test("ModelFetch/availabilityMeansTheFilesAreThere") = []
{
    if (!ModelFetch::isAvailable())
        return;

    for (const auto* name: expectedFileNames)
    {
        auto error = std::error_code {};
        check(
            std::filesystem::is_regular_file(ModelFetch::directory() / name, error));
    }
};

// The claim the whole change rests on: a run that already had the model moves no
// bytes. The fetch ran at the top of this process with the four files already on
// disk, and came back ok having downloaded nothing.
//
// Skipped on the run that did the downloading, and on a machine with no network —
// the first is true exactly once per machine, and the second is what every model
// test here skips for.
auto tASecondFetchDownloadsNothing =
    test("ModelFetch/aSecondFetchDownloadsNothing") = []
{
    const auto& record = testModelFetch();

    if (!record.ran || !record.availableBefore)
        return;

    check(record.outcome.ok);
    check(!record.outcome.downloaded);
    check(!record.outcome.cancelled);
    check(record.outcome.error.empty());
    check(record.outcome.directory == ModelFetch::directory());
};

// The one sentence every caller that shows a download shows, since three of them
// show it.
auto tProgressTextReadsAsASentence = test("ModelFetch/progressText") = []
{
    auto progress = ModelFetch::Progress {};

    check(mentions(ModelFetch::progressText(progress), "looking"));

    progress.fileIndex = 4;
    progress.fileName = "model.safetensors";
    progress.file.stage = eacp::OnlineResource::Progress::Stage::downloading;
    progress.file.bytesReceived = 75'000'000;
    progress.file.totalBytes = 151'060'136;
    progress.file.fraction = 0.5f;

    const auto text = ModelFetch::progressText(progress);

    check(mentions(text, "model.safetensors"));
    check(mentions(text, "4 of 4"));
    check(mentions(text, "75.0 of 151.1 MB"));
    check(mentions(text, "50%"));

    // A server that declared no length gives no percentage to show, only how
    // much has arrived.
    progress.file.totalBytes = -1;
    progress.file.fraction = -1.0f;

    check(mentions(ModelFetch::progressText(progress), "75.0 MB so far"));
};
