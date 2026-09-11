#include "SafeTensors.h"

#include "ModelIO.h"

#include <eacp/GPU/Device/Device.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>

namespace WSP
{
namespace
{
static_assert(std::endian::native == std::endian::little,
              "safetensors stores its header length and its blob "
              "little-endian, and this reader memcpys both");

constexpr auto headerLengthBytes = std::uint64_t {8};

// Far above any real tensor and far below the point where the byte count could
// wrap, so the shape product can be checked without a wider integer type.
constexpr auto largestElementCount = std::int64_t {1} << 40;

std::uint64_t readHeaderLength(Span<const std::uint8_t> fileBytes)
{
    auto length = std::uint64_t {};
    std::memcpy(&length, fileBytes.data(), headerLengthBytes);
    return length;
}

float halfToFloat(std::uint16_t bits)
{
    const auto sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const auto exponent = static_cast<std::uint32_t>((bits >> 10) & 0x1Fu);
    const auto mantissa = static_cast<std::uint32_t>(bits & 0x03FFu);

    if (exponent == 0x1Fu)
        return std::bit_cast<float>(sign | 0x7F800000u | (mantissa << 13));

    if (exponent == 0)
    {
        constexpr auto smallestSubnormal = 1.0f / (1 << 24);
        const auto magnitude = static_cast<float>(mantissa) * smallestSubnormal;

        return sign != 0 ? -magnitude : magnitude;
    }

    constexpr auto biasDifference = std::uint32_t {127 - 15};
    const auto rebiased = (exponent + biasDifference) << 23;

    return std::bit_cast<float>(sign | rebiased | (mantissa << 13));
}

float bfloatToFloat(std::uint16_t bits)
{
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

template <typename Element, typename Widen>
void widenEach(Span<const std::uint8_t> source, Span<float> destination, Widen widen)
{
    for (auto index = 0; index < destination.size(); ++index)
    {
        auto element = Element {};
        std::memcpy(&element,
                    source.data()
                        + static_cast<std::size_t>(index) * sizeof(Element),
                    sizeof(Element));

        destination[index] = widen(element);
    }
}

void widenToFloat(const TensorInfo& tensor,
                  Span<const std::uint8_t> source,
                  Span<float> destination)
{
    switch (tensor.type)
    {
        case TensorType::F32:
            std::memcpy(destination.data(),
                        source.data(),
                        static_cast<std::size_t>(source.size()));
            return;

        case TensorType::F16:
            widenEach<std::uint16_t>(source, destination, halfToFloat);
            return;

        case TensorType::BF16:
            widenEach<std::uint16_t>(source, destination, bfloatToFloat);
            return;

        case TensorType::F64:
            widenEach<double>(source,
                              destination,
                              [](double value)
                              { return static_cast<float>(value); });
            return;

        default:
            throw ModelError {"tensor '" + tensor.name + "' has type "
                              + std::string {tensorTypeName(tensor.type)}
                              + ", which is not a float type"};
    }
}

Vector<std::int64_t> parseShape(const Miro::Json::Object& entry,
                                const std::string& name)
{
    const auto what = "tensor '" + name + "'";
    const auto& value = ModelIO::field(entry, "shape", what);

    if (!value.isArray())
        throw ModelError {what + " shape is not an array"};

    auto shape = Vector<std::int64_t> {};

    for (const auto& dimension: value.asArray())
    {
        const auto extent = ModelIO::asInteger(dimension, what + " shape");

        if (extent < 0)
            throw ModelError {what + " has a negative dimension"};

        shape.add(extent);
    }

    return shape;
}

std::int64_t checkedElementCount(const Vector<std::int64_t>& shape,
                                 const std::string& name)
{
    auto count = std::int64_t {1};

    for (auto extent: shape)
    {
        if (extent != 0 && count > largestElementCount / extent)
            throw ModelError {"tensor '" + name + "' has an implausible shape"};

        count *= extent;
    }

    return count;
}

TensorType parseType(const Miro::Json::Object& entry, const std::string& name)
{
    const auto what = "tensor '" + name + "'";
    const auto& value = ModelIO::field(entry, "dtype", what);

    if (!value.isString())
        throw ModelError {what + " dtype is not a string"};

    if (const auto type = findTensorType(value.asString()))
        return *type;

    throw ModelError {what + " has unknown dtype '" + value.asString() + "'"};
}

void parseOffsets(const Miro::Json::Object& entry, TensorInfo& tensor)
{
    const auto what = "tensor '" + tensor.name + "'";
    const auto& value = ModelIO::field(entry, "data_offsets", what);

    if (!value.isArray() || value.asArray().size() != 2)
        throw ModelError {what + " data_offsets is not a pair"};

    const auto start = ModelIO::asInteger(value.asArray()[0], what + " start");
    const auto end = ModelIO::asInteger(value.asArray()[1], what + " end");

    if (start < 0 || end < start)
        throw ModelError {what + " has a reversed or negative byte range"};

    tensor.blobOffset = static_cast<std::uint64_t>(start);
    tensor.byteCount = static_cast<std::uint64_t>(end - start);
}

TensorInfo parseTensor(const std::string& name,
                       const Miro::Json::Value& value,
                       std::uint64_t blobSize)
{
    if (!value.isObject())
        throw ModelError {"tensor '" + name + "' is not an object"};

    const auto& entry = value.asObject();

    auto tensor = TensorInfo {};
    tensor.name = name;
    tensor.type = parseType(entry, name);
    tensor.shape = parseShape(entry, name);
    parseOffsets(entry, tensor);

    if (tensor.blobOffset + tensor.byteCount > blobSize)
        throw ModelError {"tensor '" + name + "' runs past the end of the blob"};

    const auto expected =
        checkedElementCount(tensor.shape, name) * bytesPerElement(tensor.type);

    if (static_cast<std::uint64_t>(expected) != tensor.byteCount)
        throw ModelError {"tensor '" + name
                          + "' has a shape its byte range "
                            "cannot hold"};

    return tensor;
}

eacp::GPU::Buffer uploadBytes(Span<const std::uint8_t> raw)
{
    return eacp::GPU::Device::shared().makeBuffer(
        raw.data(), raw.size(), eacp::GPU::BufferUsage::Storage);
}

// A buffer of halves is read one 32-bit word at a time — readHalf(i) fetches
// word i / 2 and picks a side of it — so an odd count of halves needs a
// padding half rather than a read one element past the allocation. An even
// count, which is every weight matrix of an even width, still goes up
// untouched.
eacp::GPU::Buffer uploadPackedHalves(Span<const std::uint8_t> raw)
{
    constexpr auto wordBytes = 4;
    const auto remainder = raw.size() % wordBytes;

    if (remainder == 0)
        return uploadBytes(raw);

    auto padded = Vector<std::uint8_t> {};
    padded.resize(raw.size() + wordBytes - remainder);
    std::memcpy(padded.data(), raw.data(), static_cast<std::size_t>(raw.size()));

    return uploadBytes(padded);
}

TensorBuffer uploadWidenedFloats(const Vector<float>& values,
                                 const std::string& name)
{
    const auto byteCount = static_cast<std::int64_t>(values.size())
                           * static_cast<std::int64_t>(sizeof(float));

    if (byteCount > std::numeric_limits<int>::max())
        throw ModelError {"tensor '" + name + "' is too large for a GPU buffer"};

    return {eacp::GPU::Device::shared().makeBuffer(values.data(),
                                                   static_cast<int>(byteCount),
                                                   eacp::GPU::BufferUsage::Storage),
            TensorType::F32};
}

// float32 to IEEE binary16, round to nearest even, overflowing to infinity —
// the same rule Metal's own narrowing follows, so a value packed here is the
// value a shader would have packed. Written out rather than taken from _Float16
// because MSVC has no such type and this is the one place the project narrows.
std::uint16_t packHalf(float value)
{
    auto bits = std::uint32_t {};
    std::memcpy(&bits, &value, sizeof(bits));

    const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
    const auto rawExponent = (bits >> 23) & 0xFFu;
    const auto mantissa = bits & 0x7FFFFFu;

    if (rawExponent == 0xFFu)
        return static_cast<std::uint16_t>(sign | 0x7C00u
                                          | (mantissa != 0u ? 0x200u : 0u));

    const auto exponent = static_cast<std::int32_t>(rawExponent) - 127 + 15;

    if (exponent >= 0x1F)
        return static_cast<std::uint16_t>(sign | 0x7C00u);

    // Below the smallest subnormal's halfway point nothing survives the shift.
    if (exponent < -10)
        return sign;

    const auto roundUp = [](std::uint32_t result,
                            std::uint32_t dropped,
                            std::uint32_t half) -> std::uint32_t
    {
        const auto atLeastHalf = (dropped & half) != 0u;
        const auto aboveHalf = (dropped & (half - 1u)) != 0u;

        return result + (atLeastHalf && (aboveHalf || (result & 1u)) ? 1u : 0u);
    };

    if (exponent <= 0)
    {
        const auto shift = static_cast<std::uint32_t>(14 - exponent);
        const auto full = mantissa | 0x800000u;
        const auto dropped = full & ((1u << shift) - 1u);
        const auto rounded = roundUp(full >> shift, dropped, 1u << (shift - 1u));

        return static_cast<std::uint16_t>(sign | rounded);
    }

    const auto packed =
        (static_cast<std::uint32_t>(exponent) << 10) | (mantissa >> 13);
    const auto rounded = roundUp(packed, mantissa & 0x1FFFu, 0x1000u);

    return static_cast<std::uint16_t>(sign | rounded);
}

std::map<std::string, std::string> parseMetadata(const Miro::Json::Value& value)
{
    if (!value.isObject())
        throw ModelError {"__metadata__ is not an object"};

    auto metadata = std::map<std::string, std::string> {};

    for (const auto& [key, entry]: value.asObject())
    {
        if (!entry.isString())
            throw ModelError {"__metadata__ entry '" + key + "' is not a string"};

        metadata.emplace(key, entry.asString());
    }

    return metadata;
}
} // namespace

std::int64_t TensorInfo::elementCount() const
{
    auto count = std::int64_t {1};

    for (auto extent: shape)
        count *= extent;

    return count;
}

int TensorInfo::rank() const
{
    return shape.size();
}

std::int64_t TensorInfo::dimension(int index) const
{
    return index >= 0 && index < shape.size() ? shape[index] : 0;
}

SafeTensors SafeTensors::fromFile(const std::filesystem::path& path)
{
    try
    {
        auto file = SafeTensors {};
        file.source = ByteSource::Mapped;
        file.mappedFile.emplace(eacp::FilePath {path});

        if (!file.mappedFile->isValid())
            throw ModelError {"cannot be opened and mapped"};

        file.readHeader();

        return file;
    }
    catch (const ModelError& error)
    {
        throw ModelError {"'" + path.string() + "': " + error.what()};
    }
}

SafeTensors SafeTensors::fromBytes(Vector<std::uint8_t> fileBytes)
{
    auto file = SafeTensors {};
    file.source = ByteSource::Owned;
    file.ownedBytes = std::move(fileBytes);
    file.readHeader();

    return file;
}

SafeTensors SafeTensors::fromView(Span<const std::uint8_t> bytes)
{
    auto file = SafeTensors {};
    file.source = ByteSource::Borrowed;
    file.borrowedBytes = bytes;
    file.readHeader();

    return file;
}

Span<const std::uint8_t> SafeTensors::fileBytes() const
{
    if (source == ByteSource::Mapped)
        return mappedFile->bytes();

    if (source == ByteSource::Borrowed)
        return borrowedBytes;

    return ownedBytes;
}

void SafeTensors::readHeader()
{
    const auto bytes = fileBytes();
    const auto fileSize = static_cast<std::uint64_t>(bytes.getSize());

    // A mapping is not bounded by the int a Vector indexes with, so this is
    // where the limit lives now. It is eacp::GPU::Buffer's, whose sizes are
    // int, and every byte count narrowed to an int below is narrowed under it.
    if (fileSize > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw ModelError {"safetensors file is larger than 2 GB"};

    if (fileSize < headerLengthBytes)
        throw ModelError {"safetensors file is shorter than its length prefix"};

    const auto headerSize = readHeaderLength(bytes);

    if (headerSize > fileSize - headerLengthBytes)
        throw ModelError {"safetensors header length runs past the end of "
                          "the file"};

    blobOffset = headerLengthBytes + headerSize;

    const auto header = ModelIO::parseObject(
        ModelIO::textOf(Span<const std::uint8_t> {bytes.data() + headerLengthBytes,
                                                  static_cast<int>(headerSize)}),
        "safetensors header");

    const auto blobSize = fileSize - blobOffset;

    for (const auto& [name, value]: header.asObject())
    {
        if (name == "__metadata__")
        {
            metadataEntries = parseMetadata(value);
            continue;
        }

        entries.add(parseTensor(name, value, blobSize));
    }
}

const Vector<TensorInfo>& SafeTensors::tensors() const
{
    return entries;
}

Vector<std::string> SafeTensors::names() const
{
    auto found = Vector<std::string> {};
    found.reserve(entries.size());

    for (const auto& tensor: entries)
        found.add(tensor.name);

    return found;
}

const TensorInfo* SafeTensors::find(std::string_view name) const
{
    const auto found =
        std::lower_bound(entries.begin(),
                         entries.end(),
                         name,
                         [](const TensorInfo& tensor, std::string_view key)
                         { return std::string_view {tensor.name} < key; });

    if (found == entries.end() || std::string_view {found->name} != name)
        return nullptr;

    return &*found;
}

bool SafeTensors::contains(std::string_view name) const
{
    return find(name) != nullptr;
}

const TensorInfo& SafeTensors::info(std::string_view name) const
{
    if (const auto* found = find(name))
        return *found;

    throw ModelError {"no tensor named '" + std::string {name} + "'"};
}

const std::map<std::string, std::string>& SafeTensors::metadata() const
{
    return metadataEntries;
}

Span<const std::uint8_t> SafeTensors::rawBytes(const TensorInfo& tensor) const
{
    return {fileBytes().data() + blobOffset + tensor.blobOffset,
            static_cast<int>(tensor.byteCount)};
}

Span<const std::uint8_t> SafeTensors::rawBytes(std::string_view name) const
{
    return rawBytes(info(name));
}

Vector<float> SafeTensors::readFloats(std::string_view name) const
{
    const auto& tensor = info(name);

    auto values = Vector<float> {};
    values.resize(static_cast<int>(tensor.elementCount()));
    widenToFloat(tensor, rawBytes(tensor), values);

    return values;
}

void SafeTensors::readFloats(std::string_view name, Span<float> destination) const
{
    const auto& tensor = info(name);

    if (destination.size() != static_cast<int>(tensor.elementCount()))
        throw ModelError {"destination for '" + tensor.name
                          + "' does not match its element count"};

    widenToFloat(tensor, rawBytes(tensor), destination);
}

TensorBuffer SafeTensors::makeBuffer(std::string_view name) const
{
    const auto& tensor = info(name);

    // Already a layout a kernel binds, so these go straight from the blob
    // rather than through a widened copy of themselves: F32 as the floats a
    // subscript reads, F16 as the packed halves readHalf reads.
    if (tensor.type == TensorType::F32)
        return {uploadBytes(rawBytes(tensor)), tensor.type};

    if (tensor.type == TensorType::F16)
        return {uploadPackedHalves(rawBytes(tensor)), tensor.type};

    // BF16 and F64 have no shader read of their own, so the conversion has to
    // happen somewhere and doing it once here beats doing it in every kernel.
    return uploadWidenedFloats(readFloats(name), tensor.name);
}

TensorBuffer SafeTensors::makeFloatBuffer(std::string_view name) const
{
    const auto& tensor = info(name);

    if (tensor.type == TensorType::F32)
        return {uploadBytes(rawBytes(tensor)), tensor.type};

    // F16 joins BF16 and F64 on the widening path here, rather than going up
    // packed as makeBuffer would have it: the caller binds this to a program
    // that subscripts floats, and a packed buffer there would be read at half
    // the stride it was written at, silently, on both backends.
    return uploadWidenedFloats(readFloats(name), tensor.name);
}

std::optional<TensorBuffer>
    SafeTensors::makeExactHalfBuffer(std::string_view name) const
{
    const auto& tensor = info(name);

    if (tensor.type == TensorType::F16)
        return makeBuffer(name);

    const auto values = readFloats(name);

    auto halves = Vector<std::uint16_t> {};
    halves.resize(values.size());

    for (auto index = 0; index < values.size(); ++index)
    {
        halves[index] = packHalf(values[index]);

        // A NaN weight compares unequal to itself and takes this exit, which
        // is the right answer for a tensor nothing here should be packing.
        if (!(halfToFloat(halves[index]) == values[index]))
            return {};
    }

    const auto byteCount = static_cast<std::int64_t>(halves.size())
                           * static_cast<std::int64_t>(sizeof(std::uint16_t));

    if (byteCount > std::numeric_limits<int>::max())
        throw ModelError {"tensor '" + tensor.name
                          + "' is too large for a GPU buffer"};

    const auto bytes = Span<const std::uint8_t> {
        reinterpret_cast<const std::uint8_t*>(halves.data()),
        static_cast<int>(byteCount)};

    return TensorBuffer {uploadPackedHalves(bytes), TensorType::F16};
}
} // namespace WSP
