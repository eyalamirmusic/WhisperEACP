#include "../Whisper/Common.h"

#include <WhisperEACP/Whisper/ResourcesDirectory.h>

#include <eacp/GPU/GPU.h>

#include <cstring>
#include <filesystem>

// What whisper_bundle_model(BundledTests) actually put beside this binary: that
// the directory the runtime resolves is a real one, that the four files are in
// it, that they are the four the fetch fetched, and that a runtime loaded out
// of them transcribes the same sentence a directory-loaded one does.
//
// Tests/Whisper/Common.h next door carries the model directory, the sample and
// the pinned transcript, the same way it brings in Tests/Model/Common.h for the
// model directory rather than keeping a second copy.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;
using namespace eacp::GPU;

namespace
{
constexpr auto bundlesModel = WHISPER_EACP_BUNDLES_MODEL != 0;

constexpr const char* modelFileNames[] = {"config.json",
                                          "preprocessor_config.json",
                                          "model.safetensors",
                                          "tokenizer.json"};

bool matchesFetchedFile(const char* name)
{
    const auto bundled =
        ModelIO::readFileBytes(Whisper::bundledModelDirectory() / name);
    const auto fetched = ModelIO::readFileBytes(modelFile(name));

    return bundled.size() == fetched.size()
           && std::memcmp(bundled.data(), fetched.data(), fetched.size()) == 0;
}
} // namespace

// The platform half, on its own and never skipping: whatever the build copied
// or did not, the directory the runtime would look in has to exist, since it
// is the one the executable itself is in — or its bundle's Resources.
auto tResourcesDirectoryIsReal = test("Bundled/resourcesDirectoryIsReal") = []
{
    auto error = std::error_code {};

    check(!resourcesDirectory().empty());
    check(std::filesystem::is_directory(resourcesDirectory(), error));
    check(Whisper::bundledModelDirectory().filename()
          == Whisper::bundledModelDirectoryName);
};

// The build copied a model or it did not, and when it did the runtime has to
// find it: the copy is a POST_BUILD step of this very target, so a step that
// quietly stopped running would leave a binary that built and shipped
// nothing. Only that direction is asserted — a build reconfigured from on to
// off leaves the earlier copy where it was, and a test that read that as a
// failure would be reporting a stale build directory, not a bug.
auto tBundledModelIsThereWhenTheBuildCopiedOne =
    test("Bundled/modelIsThereWhenTheBuildCopiedOne") = []
{
    if (!bundlesModel)
        return;

    check(Whisper::hasBundledModel());
};

// What the copy put beside the binary is what the fetch fetched, byte for byte
// — which is the claim a transcript can only make circumstantially.
auto tBundledFilesMatchTheFetch = test("Bundled/filesMatchTheFetch") = []
{
    if (!bundlesModel || !hasWhisperModel())
        return;

    for (const auto* name: modelFileNames)
        check(matchesFetchedFile(name));
};

// End to end out of the directory beside the binary: the weights are mapped
// from the copy, and nothing but the WAV is read from anywhere else.
auto tBundledModelTranscribes = test("Bundled/modelTranscribes") = []
{
    if (!bundlesModel || !Device::shared().isValid() || !hasSampleFile(jfkSample))
        return;

    auto whisper = Whisper {};
    whisper.loadBundled();
    whisper.prepare();

    // The named local is EA::Span's deleted rvalue-container constructor, which
    // makes transcribe(readWavFile(...)) a compile error — plan.md records it.
    const auto samples = readWavFile(sampleFile(jfkSample));

    const auto tokens = whisper.transcribe(samples);
    const auto text = whisper.textForTokens(tokens);

    check(tokens.size() == jfkTokenCount);
    check(trimmed(text) == jfkTranscript);
};
