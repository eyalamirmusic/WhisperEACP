#include "../Whisper/Common.h"

#include <eacp/GPU/GPU.h>

#include <ResEmbed/ResEmbed.h>

#include <cstring>
#include <string>

// What -DWHISPER_EACP_EMBED_MODEL=ON actually put in this binary: that the four
// files are there at all, that they are the four the fetch fetched, and that a
// runtime loaded out of them transcribes the same sentence a directory-loaded
// one does.
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
constexpr auto embedsModel = WHISPER_EACP_EMBEDS_MODEL != 0;

constexpr const char* modelFileNames[] = {"config.json",
                                          "preprocessor_config.json",
                                          "model.safetensors",
                                          "tokenizer.json"};

bool matchesFileOnDisk(const char* name)
{
    const auto embedded = ResEmbed::get(name, Whisper::embeddedModelCategory);
    const auto onDisk = ModelIO::readFileBytes(modelFile(name));

    return embedded.getSize() == onDisk.size()
           && std::memcmp(embedded.data(), onDisk.data(), onDisk.size()) == 0;
}
} // namespace

// The one test here that never skips, because it is the one that catches the
// failure nothing else would: ResEmbed registers its resources from a static
// initializer in an object whose symbols nothing references, so a linker free
// to drop that object leaves a binary that built and embedded nothing. The
// registry would simply answer "no model" and every test below would skip,
// quietly, on a build that had asked for one.
auto tEmbeddedModelMatchesTheBuild = test("Embedded/modelMatchesTheBuild") = []
{ check(Whisper::hasEmbeddedModel() == embedsModel); };

// What the option embedded is what the fetch fetched, byte for byte — which is
// the claim a transcript can only make circumstantially.
auto tEmbeddedFilesMatchTheFetch = test("Embedded/filesMatchTheFetch") = []
{
    if (!embedsModel || !hasWhisperModel())
        return;

    for (const auto* name: modelFileNames)
        check(matchesFileOnDisk(name));
};

// End to end out of the binary: nothing but the WAV is read from disk, and the
// weights are never copied off the borrowed view ResEmbed hands back.
auto tEmbeddedModelTranscribes = test("Embedded/modelTranscribes") = []
{
    if (!embedsModel || !Device::shared().isValid() || !hasSampleFile(jfkSample))
        return;

    auto whisper = Whisper {};
    whisper.loadEmbedded();
    whisper.prepare();

    // The named local is EA::Span's deleted rvalue-container constructor, which
    // makes transcribe(readWavFile(...)) a compile error — plan.md records it.
    const auto samples = readWavFile(sampleFile(jfkSample));

    const auto tokens = whisper.transcribe(samples);
    const auto text = whisper.textForTokens(tokens);

    check(tokens.size() == jfkTokenCount);
    check(trimmed(text) == jfkTranscript);
};
