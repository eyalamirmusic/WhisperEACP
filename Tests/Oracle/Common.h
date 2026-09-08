#pragma once

// whisper.cpp as an in-process oracle — the second tier of the order in
// CLAUDE.md, after the scalar references and before the committed fixtures.
// What makes it worth a dependency is that its public API is addressable stage
// by stage: a mel goes in through whisper_set_mel and logits come back out of
// whisper_get_logits, so a stage of ours can be dropped into an otherwise
// unmodified reference run and a disagreement names the stage rather than the
// transcript.
//
// It reads a GGML .bin converted from the same weights model.safetensors
// holds, so that conversion is a candidate explanation for any numerical
// disagreement and is the first thing to rule out.
//
// Everything here skips on a missing file, the same shape as a GPU test
// returning early when Device::shared().isValid() is false: the GGML model is
// a 75 MB download and the HuggingFace files another 154 MB, and none of them
// is a commit.

#include "../Mel/MelTestSupport.h"
#include "../Model/Common.h"

#include <WhisperEACP/Model/PreprocessorConfig.h>
#include <WhisperEACP/Tokenizer/Tokenizer.h>

#include <ggml-backend.h>
#include <whisper.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace WSP::Testing
{
inline constexpr auto tokenizerFile = "tokenizer.json";
inline constexpr auto preprocessorFile = "preprocessor_config.json";

// Four threads rather than one because the reference is a whole CPU inference
// pass; ggml splits the same work the same way whatever the count, so both
// halves of a comparison see identical arithmetic.
inline constexpr auto oracleThreads = 4;

// WHISPER_EACP_GGML_MODEL is where the CPM fetch links the file, and
// WHISPER_GGML_MODEL in the environment points at a copy somebody already has
// — the same pair as WHISPER_EACP_MODEL_DIR and WHISPER_MODEL_DIR next door.
inline std::filesystem::path ggmlModelPath()
{
    if (const auto* fromEnvironment = std::getenv("WHISPER_GGML_MODEL"))
        return {fromEnvironment};

    return {WHISPER_EACP_GGML_MODEL};
}

inline bool hasGgmlModel()
{
    auto error = std::error_code {};
    return std::filesystem::is_regular_file(ggmlModelPath(), error);
}

struct OracleDeleter
{
    void operator()(whisper_context* context) const { whisper_free(context); }
};

using Oracle = std::unique_ptr<whisper_context, OracleDeleter>;

// whisper.cpp writes a few dozen lines about the model to stderr on every
// load. Silenced so a failing test's output is the failure.
inline void silenceOracleLogging()
{
    whisper_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
}

// Which whisper.cpp this is, printed once. The root CMakeLists builds it with
// every backend off unless Benchmark/ is in the tree, in which case the
// reference's CPU path goes through whatever the defaults compiled in beside
// it — Accelerate, on a Mac — and a number below should be read knowing which.
inline void announceOracleBuild()
{
    static const auto announced = []
    {
        std::cout << "  whisper.cpp " << whisper_version() << ", backends";

        for (auto index = std::size_t {}; index < ggml_backend_dev_count(); ++index)
        {
            const auto device = ggml_backend_dev_get(index);
            std::cout << (index == 0 ? " " : ", ")
                      << ggml_backend_reg_name(ggml_backend_dev_backend_reg(device));
        }

        std::cout << "\n";
        return true;
    }();

    (void) announced;
}

inline Oracle loadOracle()
{
    silenceOracleLogging();
    announceOracleBuild();

    auto parameters = whisper_context_default_params();
    parameters.use_gpu = false;

    return Oracle {whisper_init_from_file_with_params(
        ggmlModelPath().string().c_str(), parameters)};
}

// jfk.wav, which arrives in whisper.cpp's own source tree: 16 kHz mono PCM16,
// and the only real speech in this suite. A reader for exactly that shape is
// twenty lines, and Lib/WhisperEACP/Audio has no file reader to borrow — it
// captures through MakeASound and never opens a file.
inline std::vector<float> readPcm16Wav(const std::filesystem::path& path)
{
    auto file = std::ifstream {path, std::ios::binary};

    if (!file)
        return {};

    auto header = std::array<char, 12> {};
    file.read(header.data(), header.size());

    if (std::string_view {header.data(), 4} != "RIFF"
        || std::string_view {header.data() + 8, 4} != "WAVE")
        return {};

    const auto readChunkHeader = [&file](std::string& id, std::uint32_t& size)
    {
        auto bytes = std::array<char, 8> {};

        if (!file.read(bytes.data(), bytes.size()))
            return false;

        id.assign(bytes.data(), 4);
        std::memcpy(&size, bytes.data() + 4, sizeof(size));
        return true;
    };

    auto id = std::string {};
    auto size = std::uint32_t {};

    while (readChunkHeader(id, size))
    {
        if (id == "data")
        {
            auto samples = std::vector<std::int16_t>(size / sizeof(std::int16_t));
            file.read(reinterpret_cast<char*>(samples.data()), size);

            auto normalised = std::vector<float>(samples.size());

            for (auto i = std::size_t {}; i < samples.size(); ++i)
                normalised[i] = (float) samples[i] / 32768.0f;

            return normalised;
        }

        file.seekg(size + (size & 1u), std::ios::cur);
    }

    return {};
}

// Both paths have to see the same frame count, so the input is exactly the 30 s
// window either way — zero-filled at the end, which is also what keeps the two
// front-ends' tail padding like for like (see MelOracleTests.cpp).
inline std::vector<float> paddedToWindow(std::vector<float> samples)
{
    samples.resize((std::size_t) windowSamples, 0.0f);
    return samples;
}

// Our front-end at Whisper's own shape, with the filterbank the model
// publishes rather than the triangles the rest of Tests/Mel binds: this is the
// stage under test, so nothing about it may be a test's own construction.
inline std::vector<float> ourMelSpectrogram(eacp::GPU::Device& device,
                                            const std::vector<float>& samples)
{
    const auto shape = MelShape {};
    const auto preprocessor =
        PreprocessorConfig::fromFile(modelFile(preprocessorFile));

    const auto samplesBuffer = melTest::upload(device, samples);
    const auto filterBuffer = preprocessor.makeMelFilterBuffer();
    const auto output = melTest::allocate(device, shape.melElementCount());

    auto front = MelSpectrogram {shape};
    front.prepare(device);

    auto commands = device.makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        front.encode(pass, samplesBuffer, filterBuffer, output);
    }

    commands.commit();

    return melTest::download(output, shape.melElementCount());
}

// One decoder step over a prompt, after whatever put a mel into the context's
// state. whisper_decode fills only the row of the last token — the batch it
// builds sets the logits flag on that one alone — so the row this returns is
// the distribution over the token that follows the prompt.
inline std::vector<float>
    oracleLogitsForPrompt(whisper_context& context,
                          const std::vector<whisper_token>& prompt)
{
    if (prompt.empty())
        return {};

    if (whisper_decode(
            &context, prompt.data(), (int) prompt.size(), 0, oracleThreads)
        != 0)
        return {};

    const auto vocabulary = whisper_n_vocab(&context);
    const auto* logits = whisper_get_logits(&context)
                         + (prompt.size() - 1) * (std::size_t) vocabulary;

    return {logits, logits + vocabulary};
}

// The hook the GPU decoder's oracle test will call once Decoder exists: a mel
// and a token prompt in, whisper.cpp's logits for the token after the prompt
// out — one row of n_vocab. Everything either side of the caller's stage is
// then the reference's own, which is the whole point of this tier.
//
// Nothing asserts on the decoder yet; MelOracleTests.cpp is its first caller,
// and it uses this because a logit row is the only observable whisper.cpp
// offers downstream of a mel.
inline std::vector<float>
    oracleLogitsForMel(whisper_context& context,
                       const std::vector<float>& mel,
                       const std::vector<whisper_token>& prompt)
{
    const auto shape = MelShape {};

    if (whisper_set_mel(&context, mel.data(), shape.frameCount, shape.melCount) != 0)
        return {};

    if (whisper_encode(&context, 0, oracleThreads) != 0)
        return {};

    return oracleLogitsForPrompt(context, prompt);
}

// The same, from PCM: whisper.cpp's own front-end feeding its own encoder,
// which is the run a mel of ours is compared against.
inline std::vector<float>
    oracleLogitsForPcm(whisper_context& context,
                       const std::vector<float>& samples,
                       const std::vector<whisper_token>& prompt)
{
    if (whisper_pcm_to_mel(
            &context, samples.data(), (int) samples.size(), oracleThreads)
        != 0)
        return {};

    if (whisper_encode(&context, 0, oracleThreads) != 0)
        return {};

    return oracleLogitsForPrompt(context, prompt);
}

inline float maximumAbsoluteDifference(const std::vector<float>& a,
                                       const std::vector<float>& b)
{
    auto worst = 0.0f;

    for (auto i = std::size_t {}; i < a.size() && i < b.size(); ++i)
        worst = std::max(worst, std::abs(a[i] - b[i]));

    return worst;
}

// How finely a logit comparison can resolve anything at all, measured rather
// than assumed: the same mel again with one ulp added to or subtracted from
// every cell, through the same reference. tiny.en's GGML weights are fp16 and
// there are four encoder layers and four decoder layers of them between a mel
// and a logit, so a last-bit change in the input comes back out as a few
// hundredths of a logit. A front-end difference below that number is one this
// tier cannot see, which is the strongest claim a comparison against this
// reference can make.
//
// The mel is in a two-wide band, so an ulp of a value near one is the right
// size of nudge for every cell of it.
inline float oracleLogitResolution(whisper_context& context,
                                   const std::vector<float>& mel,
                                   const std::vector<float>& logits,
                                   const std::vector<whisper_token>& prompt)
{
    constexpr auto ulp = std::numeric_limits<float>::epsilon();

    auto perturbed = mel;
    auto state = 7u;

    for (auto& value: perturbed)
    {
        state = state * 1664525u + 1013904223u;
        value += (state & 1u) != 0u ? ulp : -ulp;
    }

    return maximumAbsoluteDifference(logits,
                                     oracleLogitsForMel(context, perturbed, prompt));
}

inline int argmaxOf(const std::vector<float>& values)
{
    const auto largest = std::max_element(values.begin(), values.end());
    return largest == values.end() ? -1 : (int) (largest - values.begin());
}
} // namespace WSP::Testing
