#include "Common.h"

#include <WhisperEACP/Whisper/Whisper.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

// The top of the stack against the reference's top: our greedy search on
// jfk.wav against whisper.cpp's whisper_full over the same samples, token for
// token.
//
// This is the only assertion in the suite that covers the search itself — the
// prompt, the two suppression masks, the stop condition — rather than the
// arithmetic under it, and it is the one that would catch a suppression rule
// that is subtly the wrong one while every logit still matches.
//
// The two do not decode identically by construction, and the differences are
// named in Whisper.h. The one that matters here is that whisper.cpp's
// `suppress_nst` is **off** by default, so a default run of it does not mask the
// 90 non-speech ids HF ships as `suppress_tokens`; it is turned on below, since
// what this compares is two implementations of the same policy and not two
// policies.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;
using namespace eacp::GPU;

namespace
{
constexpr auto configName = "config.json";
constexpr auto weightsName = "model.safetensors";

bool canRun()
{
    return Device::shared().isValid() && hasGgmlModel() && hasModelFile(configName)
           && hasModelFile(weightsName) && hasModelFile(preprocessorFile)
           && hasModelFile(tokenizerFile);
}

Whisper& ourRuntime()
{
    static const auto runtime = []
    {
        auto whisper = std::make_unique<Whisper>();
        whisper->load(modelDirectory());
        whisper->prepare();
        return whisper;
    }();

    return *runtime;
}

// Greedy, one segment, no timestamps, English, no translation — and no
// temperature fallback, since a run that retried at a higher temperature would
// no longer be the greedy search this is compared against.
whisper_full_params greedyParameters()
{
    auto parameters = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    parameters.n_threads = oracleThreads;
    parameters.language = "en";
    parameters.translate = false;
    parameters.no_timestamps = true;
    parameters.single_segment = true;
    parameters.suppress_blank = true;
    parameters.suppress_nst = true;
    parameters.temperature = 0.0f;
    parameters.temperature_inc = 0.0f;
    parameters.print_progress = false;
    parameters.print_realtime = false;
    parameters.print_special = false;
    parameters.print_timestamps = false;

    return parameters;
}

// Every text token of every segment, in order. The special tokens are dropped
// on both sides: whisper.cpp keeps `<|endoftext|>` as a segment's last token
// where our loop stops on it instead, and that is a difference about where the
// marker is stored rather than about what was decoded.
std::vector<whisper_token> oracleTranscript(whisper_context& context,
                                            const std::vector<float>& samples)
{
    if (whisper_full(
            &context, greedyParameters(), samples.data(), (int) samples.size())
        != 0)
        return {};

    const auto endOfText = whisper_token_eot(&context);
    auto tokens = std::vector<whisper_token> {};

    for (auto segment = 0; segment < whisper_full_n_segments(&context); ++segment)
        for (auto index = 0; index < whisper_full_n_tokens(&context, segment);
             ++index)
        {
            const auto token = whisper_full_get_token_id(&context, segment, index);

            if (token < endOfText)
                tokens.push_back(token);
        }

    return tokens;
}

std::string joinedText(whisper_context& context,
                       const std::vector<whisper_token>& tokens)
{
    auto text = std::string {};

    for (auto token: tokens)
        text += whisper_token_to_str(&context, token);

    return text;
}
} // namespace

auto tTranscriptAgreesWithTheOracle =
    test("Oracle/transcriptAgreesWithTheOracle") = []
{
    if (!canRun())
        return;

    const auto context = loadOracle();

    if (context == nullptr)
        return;

    // whisper.cpp's own front-end and encoder on its side, ours on ours — the
    // samples are the only thing the two share, which is what makes this an
    // end-to-end comparison rather than a decoder one. Zero-filled to the
    // window because that is what our runtime does with a short utterance and
    // what keeps whisper.cpp's tail padding like for like (see
    // MelOracleTests.cpp).
    const auto samples = paddedToWindow(readPcm16Wav(WHISPER_EACP_JFK_WAV));

    if (samples.empty())
        return;

    const auto reference = oracleTranscript(*context, samples);
    check(!reference.empty());

    auto& whisper = ourRuntime();
    const auto mine = whisper.transcribe(samples);

    const auto ourText = whisper.textForTokens(mine);
    const auto theirText = joinedText(*context, reference);

    std::cout << "  ours (" << mine.size() << " tokens):" << ourText << "\n";
    std::cout << "  whisper.cpp (" << reference.size() << " tokens):" << theirText
              << "\n";

    check(mine.size() == (int) reference.size());

    for (auto index = 0; index < mine.size(); ++index)
        check(mine[index] == reference[(std::size_t) index]);

    // The text is the same claim through the vocabulary, which is worth its own
    // assertion because the two decoders are different byte-level
    // implementations and Tests/Oracle already found one string they segment
    // differently.
    check(ourText == theirText);
};
