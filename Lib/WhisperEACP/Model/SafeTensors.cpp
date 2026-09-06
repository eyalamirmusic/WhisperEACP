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
        return fromBytes(ModelIO::readFileBytes(path));
    }
    catch (const ModelError& error)
    {
        throw ModelError {"'" + path.string() + "': " + error.what()};
    }
}

SafeTensors SafeTensors::fromBytes(Vector<std::uint8_t> fileBytes)
{
    auto file = SafeTensors {};
    file.bytes = std::move(fileBytes);

    const auto fileSize = static_cast<std::uint64_t>(file.bytes.size());

    if (fileSize < headerLengthBytes)
        throw ModelError {"safetensors file is shorter than its length prefix"};

    const auto headerSize = readHeaderLength(file.bytes);

    if (headerSize > fileSize - headerLengthBytes)
        throw ModelError {"safetensors header length runs past the end of "
                          "the file"};

    file.blobOffset = headerLengthBytes + headerSize;

    const auto header = ModelIO::parseObject(
        ModelIO::textOf(Span<const std::uint8_t> {
            file.bytes.data() + headerLengthBytes, static_cast<int>(headerSize)}),
        "safetensors header");

    const auto blobSize = fileSize - file.blobOffset;

    for (const auto& [name, value]: header.asObject())
    {
        if (name == "__metadata__")
        {
            file.metadataEntries = parseMetadata(value);
            continue;
        }

        file.entries.add(parseTensor(name, value, blobSize));
    }

    return file;
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
    return {bytes.data() + blobOffset + tensor.blobOffset,
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

eacp::GPU::Buffer SafeTensors::makeBuffer(std::string_view name) const
{
    const auto& tensor = info(name);
    auto& device = eacp::GPU::Device::shared();

    // Already the layout a kernel binds, so an F32 tensor goes straight from
    // the blob rather than through a widened copy of itself — which for
    // tiny.en, whose every tensor is F32, is every tensor.
    if (tensor.type == TensorType::F32)
    {
        const auto raw = rawBytes(tensor);
        return device.makeBuffer(
            raw.data(), raw.size(), eacp::GPU::BufferUsage::Storage);
    }

    const auto values = readFloats(name);
    const auto byteCount = static_cast<std::int64_t>(values.size())
                           * static_cast<std::int64_t>(sizeof(float));

    if (byteCount > std::numeric_limits<int>::max())
        throw ModelError {"tensor '" + tensor.name
                          + "' is too large for a GPU buffer"};

    return device.makeBuffer(
        values.data(), static_cast<int>(byteCount), eacp::GPU::BufferUsage::Storage);
}
} // namespace WSP
