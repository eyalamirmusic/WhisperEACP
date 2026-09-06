#include "Common.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

// The runtime's own contracts, none of which needs a GPU: which file a failed
// load names, what the prompt is, and what the two suppression masks hold. The
// end-to-end run is next door in TranscribeTests.cpp.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;

namespace
{
constexpr auto startOfTranscriptToken = 50257;
constexpr auto endOfTextToken = 50256;
constexpr auto noTimestampsToken = 50362;
constexpr auto leadingSpaceToken = 220;

// A model loaded once for the tests that only read what load() computed. The
// weights are mapped rather than uploaded, so this costs a page fault or two
// and no device at all.
const Whisper& loadedModel()
{
    static const auto model = []
    {
        auto whisper = std::make_unique<Whisper>();
        whisper->load(modelDirectory());
        return whisper;
    }();

    return *model;
}

int nonZeroCount(Span<const float> mask)
{
    auto count = 0;

    for (auto value: mask)
        if (value != 0.0f)
            ++count;

    return count;
}
} // namespace

// The four files a HuggingFace repo is, checked in the order load() reads them:
// each directory below holds every file but one, and the error has to name the
// one. "Loading failed" about a four-file directory is a message that costs the
// caller the `ls`.
auto tLoadNamesTheMissingFile = test("Whisper/loadNamesTheMissingFile") = []
{
    if (!hasWhisperModel())
        return;

    const auto missing = [](std::initializer_list<std::string_view> present)
    {
        const auto directory = PartialModelDirectory(present);

        return modelErrorFrom(
            [&]
            {
                auto whisper = Whisper {};
                whisper.load(directory.path());
            });
    };

    check(namesFile(missing({}), "config.json"));
    check(namesFile(missing({"config.json"}), "preprocessor_config.json"));
    check(namesFile(missing({"config.json", "preprocessor_config.json"}),
                    "tokenizer.json"));
    check(namesFile(
        missing({"config.json", "preprocessor_config.json", "tokenizer.json"}),
        "model.safetensors"));

    // The whole set loads, so the four above failed on the file they named and
    // not on something the scratch directory does to all of them.
    const auto whole = PartialModelDirectory {{"config.json",
                                               "preprocessor_config.json",
                                               "tokenizer.json",
                                               "model.safetensors"}};

    auto whisper = Whisper {};
    whisper.load(whole.path());

    check(whisper.isLoaded());
    check(!whisper.isPrepared());
};

auto tLoadNamesADirectoryThatIsNotThere =
    test("Whisper/loadNamesADirectoryThatIsNotThere") = []
{
    const auto nowhere = std::filesystem::temp_directory_path() / "no-such-model";

    const auto message = modelErrorFrom(
        [&]
        {
            auto whisper = Whisper {};
            whisper.load(nowhere);
        });

    check(namesFile(message, "config.json"));
    check(mentions(message, nowhere.string()));
};

// The English-only prompt, which is the whole of what steers this decoding:
// <|startoftranscript|> from config.json's decoder_start_token_id, and
// <|notimestamps|> from the vocabulary, since config.json carries it only as the
// second half of forced_decoder_ids [[1, 50362]].
auto tPromptIsSotThenNoTimestamps = test("Whisper/promptIsSotThenNoTimestamps") = []
{
    if (!hasWhisperModel())
        return;

    const auto& whisper = loadedModel();
    const auto prompt = whisper.prompt();

    check(prompt.size() == 2);
    check(prompt[0] == startOfTranscriptToken);
    check(prompt[1] == noTimestampsToken);
    check(prompt[0] == whisper.config().decoderStartToken);
    check(prompt[1] == whisper.tokenizer().specials().noTimestamps);
};

// HF's max_length is 448 over the whole decoder sequence, prompt included, so
// what is left to sample is that less the prompt.
auto tMaximumTokensIsTheWindowLessThePrompt =
    test("Whisper/maximumTokensIsTheWindowLessThePrompt") = []
{
    if (!hasWhisperModel())
        return;

    const auto& whisper = loadedModel();

    check(whisper.config().maxTargetPositions == 448);
    check(whisper.maximumTokens() == 446);
};

// The rule, asserted by id: suppress_tokens at every step, begin_suppress_tokens
// only at the first. 90 and 2 for tiny.en, and the two lists are disjoint, so
// the first-step mask holds exactly 92.
auto tSuppressionMasks = test("Whisper/suppressionMasks") = []
{
    if (!hasWhisperModel())
        return;

    const auto& whisper = loadedModel();
    const auto& config = whisper.config();
    const auto first = whisper.firstStepMask();
    const auto later = whisper.laterStepMask();

    check(first.size() == config.vocabularySize);
    check(later.size() == config.vocabularySize);

    check(config.suppressedTokens.size() == 90);
    check(config.initiallySuppressedTokens.size() == 2);

    for (auto token: config.suppressedTokens)
    {
        check(later[token] != 0.0f);
        check(first[token] != 0.0f);
    }

    // 220 is a leading space and 50256 is <|endoftext|>: an empty transcript and
    // a transcript starting with a space are the two things the first sampled
    // token may not be, and both are free afterwards.
    check(first[leadingSpaceToken] != 0.0f);
    check(first[endOfTextToken] != 0.0f);
    check(later[leadingSpaceToken] == 0.0f);
    check(later[endOfTextToken] == 0.0f);

    check(nonZeroCount(later) == 90);
    check(nonZeroCount(first) == 92);

    // <|startoftranscript|> is in suppress_tokens and so is masked throughout;
    // <|notimestamps|> is in neither list, which is HF's own answer and not an
    // omission here — whisper.cpp masks it unconditionally instead.
    check(later[startOfTranscriptToken] != 0.0f);
    check(later[noTimestampsToken] == 0.0f);
    check(first[noTimestampsToken] == 0.0f);
};

// Timestamps are not masked, because HF does not mask them: its timestamp
// processor is added only for return_timestamps=True, and <|notimestamps|> in
// the prompt is what keeps them out of a transcript. whisper.cpp with
// no_timestamps masks every id from <|0.00|> up at every step, which is the one
// difference between the two that could show in a transcript at all.
auto tTimestampsAreNotSuppressed = test("Whisper/timestampsAreNotSuppressed") = []
{
    if (!hasWhisperModel())
        return;

    const auto& whisper = loadedModel();
    const auto& specials = whisper.tokenizer().specials();

    check(specials.firstTimestamp != invalidTokenId);
    check(specials.timestampCount > 0);

    for (auto token = specials.firstTimestamp;
         token < whisper.config().vocabularySize;
         ++token)
    {
        check(whisper.firstStepMask()[token] == 0.0f);
        check(whisper.laterStepMask()[token] == 0.0f);
    }
};

// Every call that needs a device or a model says so rather than dereferencing an
// empty optional.
auto tUnpreparedCallsThrow = test("Whisper/unpreparedCallsThrow") = []
{
    auto whisper = Whisper {};
    const auto silence = std::vector<float>(1000, 0.0f);

    check(mentions(modelErrorFrom([&] { whisper.transcribe(silence); }),
                   "not been prepared"));
    check(mentions(modelErrorFrom([&] { whisper.tokenizer(); }),
                   "not loaded a model"));
    check(
        mentions(modelErrorFrom([&] { whisper.prepare(); }), "not loaded a model"));
};
