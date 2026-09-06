#include "WavFile.h"

#include "Format.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>

namespace WSP
{
namespace
{
constexpr auto riffHeaderBytes = 12;
constexpr auto chunkHeaderBytes = 8;
constexpr auto formatChunkBytes = 16;

// The two WAVE encodings this reads, and the wrapper a file uses instead of
// either once it names a channel mask. An extensible file's real encoding is
// the first two bytes of the SubFormat GUID that follows the extension size.
constexpr auto pcmFormat = 1;
constexpr auto floatFormat = 3;
constexpr auto extensibleFormat = 0xfffe;
constexpr auto extensibleSubFormatOffset = 24;

struct WaveFormat
{
    int encoding = 0;
    int channels = 0;
    int sampleRate = 0;
    int bitsPerSample = 0;
};

[[noreturn]] void fail(const std::filesystem::path& path, std::string_view what)
{
    throw WavError {path.string() + ": " + std::string {what}};
}

Vector<std::uint8_t> readWholeFile(const std::filesystem::path& path)
{
    auto file = std::ifstream {path, std::ios::binary | std::ios::ate};

    if (!file)
        fail(path, "cannot be opened");

    const auto byteCount = file.tellg();

    if (byteCount < 0)
        fail(path, "cannot be sized");

    auto bytes = Vector<std::uint8_t> {};
    bytes.resize((int) byteCount);

    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), (std::streamsize) byteCount);

    if (!file)
        fail(path, "ended before the size it reported");

    return bytes;
}

std::string_view textAt(const Vector<std::uint8_t>& bytes, int offset, int length)
{
    return {reinterpret_cast<const char*>(bytes.data()) + offset,
            (std::size_t) length};
}

// Little-endian throughout, read a byte at a time rather than by memcpy into a
// wider type: RIFF is defined little-endian and this way says so on a big
// endian host as well.
std::uint32_t
    littleEndianAt(const Vector<std::uint8_t>& bytes, int offset, int width)
{
    auto value = std::uint32_t {};

    for (auto index = 0; index < width; ++index)
        value |= (std::uint32_t) bytes[offset + index] << (8 * index);

    return value;
}

int intAt(const Vector<std::uint8_t>& bytes, int offset, int width)
{
    return (int) littleEndianAt(bytes, offset, width);
}

WaveFormat readFormatChunk(const std::filesystem::path& path,
                           const Vector<std::uint8_t>& bytes,
                           int offset,
                           int length)
{
    if (length < formatChunkBytes)
        fail(path, "has a fmt chunk shorter than the 16 bytes it must hold");

    auto format = WaveFormat {};
    format.encoding = intAt(bytes, offset, 2);
    format.channels = intAt(bytes, offset + 2, 2);
    format.sampleRate = intAt(bytes, offset + 4, 4);
    format.bitsPerSample = intAt(bytes, offset + 14, 2);

    if (format.encoding == extensibleFormat)
    {
        if (length < extensibleSubFormatOffset + 2)
            fail(path, "declares WAVE_FORMAT_EXTENSIBLE with no SubFormat");

        format.encoding = intAt(bytes, offset + extensibleSubFormatOffset, 2);
    }

    return format;
}

void requireSupported(const std::filesystem::path& path, const WaveFormat& format)
{
    if (format.channels <= 0)
        fail(path, "declares no channels");

    if (format.sampleRate != sampleRate)
        fail(path,
             "is at " + std::to_string(format.sampleRate)
                 + " Hz, and this reader does not resample: Whisper's front-end "
                   "is written against "
                 + std::to_string(sampleRate) + " Hz");

    const auto isPcm16 = format.encoding == pcmFormat && format.bitsPerSample == 16;
    const auto isFloat32 =
        format.encoding == floatFormat && format.bitsPerSample == 32;

    if (!isPcm16 && !isFloat32)
        fail(path,
             "holds encoding " + std::to_string(format.encoding) + " at "
                 + std::to_string(format.bitsPerSample)
                 + " bits, and this reader decodes only 16-bit PCM and 32-bit "
                   "float");
}

// 32768 rather than 32767, so that the mapping is the exact power of two every
// other implementation uses and -32768 lands on -1 rather than just past it.
float pcm16Sample(const Vector<std::uint8_t>& bytes, int offset)
{
    const auto raw = (std::int16_t) littleEndianAt(bytes, offset, 2);
    return (float) raw / 32768.0f;
}

float floatSample(const Vector<std::uint8_t>& bytes, int offset)
{
    const auto raw = littleEndianAt(bytes, offset, 4);

    auto value = 0.0f;
    std::memcpy(&value, &raw, sizeof(value));

    return value;
}

Vector<float> firstChannelOf(const Vector<std::uint8_t>& bytes,
                             int offset,
                             int length,
                             const WaveFormat& format)
{
    const auto sampleBytes = format.bitsPerSample / 8;
    const auto frameBytes = sampleBytes * format.channels;
    const auto frames = length / frameBytes;

    auto samples = Vector<float> {};
    samples.resize(frames);

    for (auto frame = 0; frame < frames; ++frame)
    {
        const auto at = offset + frame * frameBytes;

        samples[frame] = format.encoding == pcmFormat ? pcm16Sample(bytes, at)
                                                      : floatSample(bytes, at);
    }

    return samples;
}
} // namespace

Vector<float> readWavFile(const std::filesystem::path& path)
{
    const auto bytes = readWholeFile(path);

    if (bytes.size() < riffHeaderBytes || textAt(bytes, 0, 4) != "RIFF"
        || textAt(bytes, 8, 4) != "WAVE")
        fail(path, "is not a RIFF/WAVE file");

    auto format = WaveFormat {};
    auto sawFormat = false;

    // A chunk's body is padded to an even length, and the pad byte is not
    // counted in the size the header gives — so the walk adds it back rather
    // than landing one byte short of the next header for the rest of the file.
    for (auto at = riffHeaderBytes; at + chunkHeaderBytes <= bytes.size();)
    {
        const auto id = textAt(bytes, at, 4);
        const auto declared = littleEndianAt(bytes, at + 4, 4);
        const auto body = at + chunkHeaderBytes;

        if (declared > (std::uint32_t) (bytes.size() - body))
            fail(path, "has a chunk running past the end of the file");

        const auto length = (int) declared;

        if (id == "fmt ")
        {
            format = readFormatChunk(path, bytes, body, length);
            sawFormat = true;
        }
        else if (id == "data")
        {
            if (!sawFormat)
                fail(path, "puts its data chunk before its fmt chunk");

            requireSupported(path, format);
            return firstChannelOf(bytes, body, length, format);
        }

        at = body + length + (length & 1);
    }

    fail(path, "carries no data chunk");
}
} // namespace WSP
