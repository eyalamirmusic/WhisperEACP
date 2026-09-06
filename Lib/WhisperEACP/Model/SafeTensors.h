#pragma once

#include <WhisperEACP/Core/Core.h>
#include <WhisperEACP/Model/ModelError.h>
#include <WhisperEACP/Model/TensorType.h>

#include <eacp/GPU/Buffer/Buffer.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>

namespace WSP
{
// One tensor as the header describes it: what it is called, what it is made of,
// its shape, and where its bytes sit in the blob the header is followed by.
struct TensorInfo
{
    std::string name;
    TensorType type = TensorType::F32;
    Vector<std::int64_t> shape;
    std::uint64_t blobOffset = 0;
    std::uint64_t byteCount = 0;

    std::int64_t elementCount() const;
    int rank() const;

    // Zero past the end, so a caller checking a shape against a config reads
    // the same "not that" for a missing axis as for a zero-length one.
    std::int64_t dimension(int index) const;
};

// A safetensors file held in memory: eight bytes of little-endian header
// length, that many bytes of JSON naming every tensor, then one raw blob the
// offsets in that JSON are relative to.
//
// Nothing here trusts the header. A file shorter than the length prefix, a
// header length that runs past the end, JSON that does not parse or is not an
// object, an entry missing dtype / shape / data_offsets, a dtype this build
// does not know, a negative or reversed byte range, a range that leaves the
// blob, and a shape whose element count disagrees with that range are each a
// ModelError rather than a read past the end of the buffer. `__metadata__` is
// not a tensor and is kept separately.
class SafeTensors
{
public:
    static SafeTensors fromFile(const std::filesystem::path& path);
    static SafeTensors fromBytes(Vector<std::uint8_t> fileBytes);

    // Sorted by name, since the header is read out of an ordered map.
    const Vector<TensorInfo>& tensors() const;
    Vector<std::string> names() const;

    // Null when the file has no such tensor; info() is the same lookup for a
    // caller that would only turn the null back into an error.
    const TensorInfo* find(std::string_view name) const;
    bool contains(std::string_view name) const;
    const TensorInfo& info(std::string_view name) const;

    // The `__metadata__` entry, which the format defines as string to string.
    // Empty when the file carries none.
    const std::map<std::string, std::string>& metadata() const;

    Span<const std::uint8_t> rawBytes(const TensorInfo& tensor) const;
    Span<const std::uint8_t> rawBytes(std::string_view name) const;

    // Widened to float32 on the way out whatever the file holds, because the
    // shader EDSL has no half: every buffer a kernel binds is float32, so the
    // conversion has to happen somewhere and doing it once at load beats doing
    // it in every kernel that reads a weight.
    Vector<float> readFloats(std::string_view name) const;
    void readFloats(std::string_view name, Span<float> destination) const;

    eacp::GPU::Buffer makeBuffer(std::string_view name) const;

private:
    SafeTensors() = default;

    Vector<std::uint8_t> bytes;
    std::uint64_t blobOffset = 0;
    Vector<TensorInfo> entries;
    std::map<std::string, std::string> metadataEntries;
};
} // namespace WSP
