#pragma once

#include <WhisperEACP/Model/Model.h>

#include <NanoTest/NanoTest.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <string>
#include <string_view>

// A safetensors file is small enough to write by hand, so the bulk of this
// suite needs no download at all: every header below is assembled here, byte
// for byte, and parsed straight back.
namespace WSP::Testing
{
inline Vector<std::uint8_t> assembleWithLength(std::uint64_t declaredLength,
                                               std::string_view header,
                                               const Vector<std::uint8_t>& blob)
{
    auto file = Vector<std::uint8_t> {};
    file.resize(8);
    std::memcpy(file.data(), &declaredLength, sizeof(declaredLength));

    for (auto character: header)
        file.add(static_cast<std::uint8_t>(character));

    for (auto byte: blob)
        file.add(byte);

    return file;
}

inline Vector<std::uint8_t> assemble(std::string_view header,
                                     const Vector<std::uint8_t>& blob = {})
{
    return assembleWithLength(header.size(), header, blob);
}

template <typename Element>
Vector<std::uint8_t> toBytes(std::initializer_list<Element> values)
{
    auto bytes = Vector<std::uint8_t> {};
    bytes.resize(static_cast<int>(values.size() * sizeof(Element)));

    auto* out = bytes.data();

    for (auto value: values)
    {
        std::memcpy(out, &value, sizeof(Element));
        out += sizeof(Element);
    }

    return bytes;
}

// The same assembled bytes, on disk, for the path that maps a file instead of
// taking a buffer. Removed when the test that made it ends, so nothing is left
// in the temporary directory whether the test passed or threw.
class ScratchFile
{
public:
    ScratchFile(std::string_view name, const Vector<std::uint8_t>& bytes)
        : filePath(std::filesystem::temp_directory_path() / name)
    {
        auto out = std::ofstream {filePath, std::ios::binary | std::ios::trunc};
        out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }

    ScratchFile(const ScratchFile&) = delete;
    ScratchFile& operator=(const ScratchFile&) = delete;

    ~ScratchFile()
    {
        auto error = std::error_code {};
        std::filesystem::remove(filePath, error);
    }

    const std::filesystem::path& path() const { return filePath; }

private:
    std::filesystem::path filePath;
};

template <typename Body>
bool throwsModelError(Body&& body)
{
    try
    {
        body();
    }
    catch (const ModelError&)
    {
        return true;
    }
    catch (...)
    {
        return false;
    }

    return false;
}

inline bool nearlyEqual(float actual, float expected, float tolerance = 1.0e-6f)
{
    const auto difference = actual - expected;
    return (difference < 0 ? -difference : difference) <= tolerance;
}

// WHISPER_EACP_MODEL_DIR is where the CPM fetch assembles the real files, and
// WHISPER_MODEL_DIR in the environment points at a checkout somebody already
// has. Both may be absent, which is what the skips below are for.
inline std::filesystem::path modelDirectory()
{
    if (const auto* fromEnvironment = std::getenv("WHISPER_MODEL_DIR"))
        return {fromEnvironment};

    return {WHISPER_EACP_MODEL_DIR};
}

inline std::filesystem::path modelFile(std::string_view name)
{
    return modelDirectory() / name;
}

inline bool hasModelFile(std::string_view name)
{
    auto error = std::error_code {};
    return std::filesystem::is_regular_file(modelFile(name), error);
}
} // namespace WSP::Testing
