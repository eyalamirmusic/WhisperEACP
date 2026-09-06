#pragma once

// The model files and the sample this module needs, and the scratch directory
// the load errors are provoked from.
//
// Tests/Model/Common.h is where modelDirectory(), modelFile() and
// hasModelFile() already live, so this brings that in rather than keeping a
// second copy of the same pair of environment variables.

#include "../Model/Common.h"

#include <WhisperEACP/Whisper/Whisper.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>

namespace WSP::Testing
{
inline bool hasWhisperModel()
{
    return hasModelFile("config.json") && hasModelFile("preprocessor_config.json")
           && hasModelFile("model.safetensors") && hasModelFile("tokenizer.json");
}

// WHISPER_EACP_SAMPLE_DIR is where the CPM fetch assembles the sample, and
// WHISPER_SAMPLE_DIR in the environment points at a copy somebody already has —
// the same pair as WHISPER_EACP_MODEL_DIR and WHISPER_MODEL_DIR next door.
inline std::filesystem::path sampleDirectory()
{
    if (const auto* fromEnvironment = std::getenv("WHISPER_SAMPLE_DIR"))
        return {fromEnvironment};

    return {WHISPER_EACP_SAMPLE_DIR};
}

inline std::filesystem::path sampleFile(std::string_view name)
{
    return sampleDirectory() / name;
}

inline bool hasSampleFile(std::string_view name)
{
    auto error = std::error_code {};
    return std::filesystem::is_regular_file(sampleFile(name), error);
}

// A directory holding symlinks to whichever of the four model files a test
// wants present. Symlinks rather than copies because model.safetensors is
// 151 MB, and a load error is about the name in the directory rather than about
// the bytes behind it.
class PartialModelDirectory
{
public:
    explicit PartialModelDirectory(std::initializer_list<std::string_view> present)
        : directoryPath(std::filesystem::temp_directory_path() / uniqueName())
    {
        auto error = std::error_code {};
        std::filesystem::create_directories(directoryPath, error);

        for (auto name: present)
            std::filesystem::create_symlink(
                modelFile(name), directoryPath / name, error);
    }

    PartialModelDirectory(const PartialModelDirectory&) = delete;
    PartialModelDirectory& operator=(const PartialModelDirectory&) = delete;

    ~PartialModelDirectory()
    {
        auto error = std::error_code {};
        std::filesystem::remove_all(directoryPath, error);
    }

    const std::filesystem::path& path() const { return directoryPath; }

private:
    static std::string uniqueName()
    {
        static auto counter = 0;
        return "whisper-partial-model-" + std::to_string(counter++);
    }

    std::filesystem::path directoryPath;
};

// What a load error has to say, which is more than that it failed: a four-file
// directory missing one file should name the one.
template <typename Body>
std::string modelErrorFrom(Body&& body)
{
    try
    {
        body();
    }
    catch (const ModelError& failure)
    {
        return failure.what();
    }
    catch (...)
    {
        return "a failure that was not a ModelError";
    }

    return "no failure at all";
}

inline bool mentions(std::string_view message, std::string_view what)
{
    return message.find(what) != std::string_view::npos;
}

// The name as a whole word, since "config.json" is a substring of
// "preprocessor_config.json" and an assertion that could not tell those two
// apart would pass on the wrong error.
inline bool namesFile(std::string_view message, std::string_view name)
{
    return mentions(message, " " + std::string {name});
}
} // namespace WSP::Testing
