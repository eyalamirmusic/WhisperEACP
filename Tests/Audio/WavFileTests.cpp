#include <WhisperEACP/Audio/Audio.h>

#include <NanoTest/NanoTest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

// A WAV file is small enough to write byte for byte, so nothing here needs a
// download: every file below is assembled in the test and read straight back.
// The one real recording this project transcribes is checked where it is used,
// in Tests/Whisper.

using namespace nano;
using namespace WSP;

namespace
{
// A RIFF/WAVE builder that takes each field rather than deriving it, because
// what is under test includes what the reader does with a field that is wrong.
class WavBuilder
{
public:
    WavBuilder& format(int encoding, int channels, int rate, int bits)
    {
        return chunkOf("fmt ", formatFields(encoding, channels, rate, bits));
    }

    WavBuilder& extensibleFormat(int subFormat, int channels, int rate, int bits)
    {
        auto chunk = formatFields(0xfffe, channels, rate, bits);

        // cbSize, then the extension: valid bits, channel mask, and the
        // SubFormat GUID whose first two bytes carry the real encoding.
        append(chunk, 22, 2);
        append(chunk, bits, 2);
        append(chunk, 3, 4);
        append(chunk, subFormat, 2);

        for (auto index = 0; index < 14; ++index)
            chunk.push_back(0);

        return chunkOf("fmt ", chunk);
    }

    WavBuilder& pcm16Data(const std::vector<std::int16_t>& samples)
    {
        auto chunk = std::vector<std::uint8_t> {};

        for (auto sample: samples)
            append(chunk, (std::uint16_t) sample, 2);

        return chunkOf("data", chunk);
    }

    WavBuilder& floatData(const std::vector<float>& samples)
    {
        auto chunk = std::vector<std::uint8_t> {};

        for (auto sample: samples)
        {
            auto bits = std::uint32_t {};
            std::memcpy(&bits, &sample, sizeof(bits));
            append(chunk, bits, 4);
        }

        return chunkOf("data", chunk);
    }

    // The metadata jfk.wav itself carries between fmt and data, so the walk is
    // exercised rather than assumed. An odd length is worth its own case: the
    // pad byte after such a chunk is not counted in the size the header gives.
    WavBuilder& listChunk(int length)
    {
        return chunkOf("LIST",
                       std::vector<std::uint8_t>((std::size_t) length, 0x20));
    }

    std::vector<std::uint8_t> build(std::string_view riff = "RIFF",
                                    std::string_view wave = "WAVE") const
    {
        auto file = std::vector<std::uint8_t> {};
        appendText(file, riff);
        append(file, (int) (4 + body.size()), 4);
        appendText(file, wave);
        file.insert(file.end(), body.begin(), body.end());

        return file;
    }

private:
    static void appendText(std::vector<std::uint8_t>& bytes, std::string_view text)
    {
        for (auto character: text)
            bytes.push_back((std::uint8_t) character);
    }

    static void
        append(std::vector<std::uint8_t>& bytes, std::uint32_t value, int width)
    {
        for (auto index = 0; index < width; ++index)
            bytes.push_back((std::uint8_t) ((value >> (8 * index)) & 0xffu));
    }

    static std::vector<std::uint8_t>
        formatFields(int encoding, int channels, int rate, int bits)
    {
        auto chunk = std::vector<std::uint8_t> {};
        append(chunk, (std::uint32_t) encoding, 2);
        append(chunk, (std::uint32_t) channels, 2);
        append(chunk, (std::uint32_t) rate, 4);
        append(chunk, (std::uint32_t) (rate * channels * bits / 8), 4);
        append(chunk, (std::uint32_t) (channels * bits / 8), 2);
        append(chunk, (std::uint32_t) bits, 2);

        return chunk;
    }

    WavBuilder& chunkOf(std::string_view id, const std::vector<std::uint8_t>& chunk)
    {
        appendText(body, id);
        append(body, (std::uint32_t) chunk.size(), 4);
        body.insert(body.end(), chunk.begin(), chunk.end());

        if ((chunk.size() & 1u) != 0u)
            body.push_back(0);

        return *this;
    }

    std::vector<std::uint8_t> body;
};

// The bytes on disk, removed when the test that made them ends.
class ScratchWav
{
public:
    ScratchWav(std::string_view name, const std::vector<std::uint8_t>& bytes)
        : filePath(std::filesystem::temp_directory_path() / name)
    {
        auto out = std::ofstream {filePath, std::ios::binary | std::ios::trunc};
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  (std::streamsize) bytes.size());
    }

    ScratchWav(const ScratchWav&) = delete;
    ScratchWav& operator=(const ScratchWav&) = delete;

    ~ScratchWav()
    {
        auto error = std::error_code {};
        std::filesystem::remove(filePath, error);
    }

    const std::filesystem::path& path() const { return filePath; }

private:
    std::filesystem::path filePath;
};

std::string errorFrom(const std::filesystem::path& path)
{
    try
    {
        readWavFile(path);
    }
    catch (const WavError& failure)
    {
        return failure.what();
    }
    catch (...)
    {
        return "a failure that was not a WavError";
    }

    return "no failure at all";
}

std::string errorFromBytes(const std::vector<std::uint8_t>& bytes,
                           std::string_view name)
{
    try
    {
        readWavBytes(bytes, name);
    }
    catch (const WavError& failure)
    {
        return failure.what();
    }
    catch (...)
    {
        return "a failure that was not a WavError";
    }

    return "no failure at all";
}

bool mentions(std::string_view message, std::string_view what)
{
    return message.find(what) != std::string_view::npos;
}
} // namespace

auto tReadsPcm16 = test("Audio/wavReadsPcm16") = []
{
    const auto builder = WavBuilder {}
                             .format(1, 1, sampleRate, 16)
                             .pcm16Data({0, 32767, -32768, 16384});

    const auto file = ScratchWav {"whisper-pcm16.wav", builder.build()};
    const auto samples = readWavFile(file.path());

    check(samples.size() == 4);
    check(samples[0] == 0.0f);
    check(samples[1] == 32767.0f / 32768.0f);
    check(samples[2] == -1.0f);
    check(samples[3] == 0.5f);
};

auto tReadsFloat32 = test("Audio/wavReadsFloat32") = []
{
    const auto builder =
        WavBuilder {}.format(3, 1, sampleRate, 32).floatData({0.0f, 0.25f, -0.75f});

    const auto file = ScratchWav {"whisper-float32.wav", builder.build()};
    const auto samples = readWavFile(file.path());

    check(samples.size() == 3);
    check(samples[0] == 0.0f);
    check(samples[1] == 0.25f);
    check(samples[2] == -0.75f);
};

// WAVE_FORMAT_EXTENSIBLE is what a recorder writes once it names a channel
// mask, and the encoding then lives in the SubFormat GUID rather than in the
// format field.
auto tReadsExtensible = test("Audio/wavReadsExtensible") = []
{
    const auto builder = WavBuilder {}
                             .extensibleFormat(1, 1, sampleRate, 16)
                             .pcm16Data({8192, -8192});

    const auto file = ScratchWav {"whisper-extensible.wav", builder.build()};
    const auto samples = readWavFile(file.path());

    check(samples.size() == 2);
    check(samples[0] == 0.25f);
    check(samples[1] == -0.25f);
};

// The first channel, not a mixdown: which channels to keep is a decision about
// the signal, and this reader has no way for a caller to state one.
auto tTakesTheFirstChannel = test("Audio/wavTakesTheFirstChannel") = []
{
    const auto builder = WavBuilder {}
                             .format(1, 2, sampleRate, 16)
                             .pcm16Data({16384, -16384, 8192, -8192});

    const auto file = ScratchWav {"whisper-stereo.wav", builder.build()};
    const auto samples = readWavFile(file.path());

    check(samples.size() == 2);
    check(samples[0] == 0.5f);
    check(samples[1] == 0.25f);
};

// jfk.wav puts a LIST/INFO between fmt and data, so a reader that took the
// third chunk to be the samples would read metadata as audio.
auto tWalksPastUnknownChunks = test("Audio/wavWalksPastUnknownChunks") = []
{
    const auto builder =
        WavBuilder {}.format(1, 1, sampleRate, 16).listChunk(26).pcm16Data({16384});

    const auto file = ScratchWav {"whisper-list.wav", builder.build()};
    const auto samples = readWavFile(file.path());

    check(samples.size() == 1);
    check(samples[0] == 0.5f);
};

// A chunk body is padded to an even length and the pad byte is not counted in
// the size, so a walk that did not add it back would land one byte short of
// every header after the first odd chunk — and read the data chunk's id as
// garbage.
auto tWalksPastAnOddChunk = test("Audio/wavWalksPastAnOddChunk") = []
{
    const auto builder = WavBuilder {}
                             .format(1, 1, sampleRate, 16)
                             .listChunk(5)
                             .pcm16Data({16384, -16384});

    const auto file = ScratchWav {"whisper-odd.wav", builder.build()};
    const auto samples = readWavFile(file.path());

    check(samples.size() == 2);
    check(samples[0] == 0.5f);
    check(samples[1] == -0.5f);
};

auto tRefusesOtherSampleRates = test("Audio/wavRefusesOtherSampleRates") = []
{
    const auto builder =
        WavBuilder {}.format(1, 1, 44100, 16).pcm16Data({0, 1, 2, 3});

    const auto file = ScratchWav {"whisper-44k.wav", builder.build()};
    const auto message = errorFrom(file.path());

    check(mentions(message, "44100"));
    check(mentions(message, "16000"));
    check(mentions(message, "resample"));
};

auto tRefusesUnknownEncodings = test("Audio/wavRefusesUnknownEncodings") = []
{
    const auto builder =
        WavBuilder {}.format(1, 1, sampleRate, 24).pcm16Data({0, 1, 2});

    const auto file = ScratchWav {"whisper-24bit.wav", builder.build()};
    const auto message = errorFrom(file.path());

    check(mentions(message, "24"));
    check(mentions(message, "16-bit PCM"));
};

auto tRefusesWhatIsNotAWave = test("Audio/wavRefusesWhatIsNotAWave") = []
{
    const auto notRiff = ScratchWav {"whisper-not-riff.wav",
                                     WavBuilder {}
                                         .format(1, 1, sampleRate, 16)
                                         .pcm16Data({0})
                                         .build("RIFX", "WAVE")};

    const auto notWave = ScratchWav {"whisper-not-wave.wav",
                                     WavBuilder {}
                                         .format(1, 1, sampleRate, 16)
                                         .pcm16Data({0})
                                         .build("RIFF", "AVI ")};

    check(mentions(errorFrom(notRiff.path()), "not a RIFF/WAVE file"));
    check(mentions(errorFrom(notWave.path()), "not a RIFF/WAVE file"));

    const auto missing = std::filesystem::temp_directory_path() / "no-such.wav";
    check(mentions(errorFrom(missing), "cannot be opened"));
};

auto tRefusesAFileWithNoData = test("Audio/wavRefusesAFileWithNoData") = []
{
    const auto file = ScratchWav {
        "whisper-no-data.wav", WavBuilder {}.format(1, 1, sampleRate, 16).build()};

    check(mentions(errorFrom(file.path()), "no data chunk"));
};

// readWavFile is readWavBytes over the file's bytes, so the two have to agree
// sample for sample: reading a file is a way of getting to the decode rather
// than a decode of its own.
auto tReadsBytesLikeAFile = test("Audio/wavReadsBytesLikeAFile") = []
{
    const auto bytes = WavBuilder {}
                           .format(1, 1, sampleRate, 16)
                           .listChunk(5)
                           .pcm16Data({0, 32767, -32768, 16384})
                           .build();

    const auto file = ScratchWav {"whisper-bytes.wav", bytes};

    const auto fromFile = readWavFile(file.path());
    const auto fromBytes = readWavBytes(bytes, "whisper-bytes.wav");

    check(fromFile.size() == 4);
    check(fromBytes.size() == fromFile.size());

    for (auto index = 0; index < fromFile.size(); ++index)
        check(fromBytes[index] == fromFile[index]);
};

// What `name` is for: a buffer has no path to put in front of the colon, so the
// caller says what to call it and every message is otherwise the file's own.
auto tBytesErrorsNameTheSource = test("Audio/wavBytesErrorsNameTheSource") = []
{
    const auto wrongRate =
        WavBuilder {}.format(1, 1, 44100, 16).pcm16Data({0, 1, 2, 3}).build();

    const auto message = errorFromBytes(wrongRate, "embedded/jfk.wav");

    check(mentions(message, "embedded/jfk.wav"));
    check(mentions(message, "44100"));
    check(mentions(message, "resample"));

    const auto nothing = std::vector<std::uint8_t> {};
    const auto empty = errorFromBytes(nothing, "an-empty-resource");

    check(mentions(empty, "an-empty-resource"));
    check(mentions(empty, "not a RIFF/WAVE file"));
};
