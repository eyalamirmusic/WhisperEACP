#pragma once

#include <WhisperEACP/Core/Core.h>
#include <WhisperEACP/Model/ModelError.h>
#include <WhisperEACP/Model/TensorType.h>

#include <eacp/Core/Utils/MemoryMappedFile.h>
#include <eacp/GPU/Buffer/Buffer.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
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

// A tensor uploaded to the device, and what the buffer's elements are: F32 for
// the float buffer a kernel subscripts, F16 for one still packed two halves to
// a word, which a kernel reads through InputBuffer::readHalf.
//
// The two travel together because nothing about a GPU::Buffer says which of
// them it holds, and binding a packed buffer where a float one is expected is
// wrong by a factor of two in every index while staying silent on both
// backends.
struct TensorBuffer
{
    eacp::GPU::Buffer buffer;
    TensorType storage = TensorType::F32;

    bool isPackedHalf() const { return storage == TensorType::F16; }
};

// A safetensors file: eight bytes of little-endian header length, that many
// bytes of JSON naming every tensor, then one raw blob the offsets in that JSON
// are relative to.
//
// fromFile maps the file rather than reading it, so a 151 MB blob is never
// resident before the first GPU copy and the pages a load touches arrive from
// the page cache as it touches them; fromBytes takes a buffer already in
// memory. Either way the bytes outlive every Span handed out, which is why this
// is move-only. fromView is the third, and the one where that is the caller's
// promise rather than this object's.
//
// Nothing here trusts the header. A file shorter than the length prefix, a
// header length that runs past the end, JSON that does not parse or is not an
// object, an entry missing dtype / shape / data_offsets, a dtype this build
// does not know, a negative or reversed byte range, a range that leaves the
// blob, and a shape whose element count disagrees with that range are each a
// ModelError rather than a read past the end of the mapping. `__metadata__` is
// not a tensor and is kept separately.
class SafeTensors
{
public:
    static SafeTensors fromFile(const std::filesystem::path& path);
    static SafeTensors fromBytes(Vector<std::uint8_t> fileBytes);

    // Bytes this does not own and does not copy. **The caller guarantees they
    // outlive this SafeTensors and every Span it hands out** — rawBytes(),
    // readFloats() and makeBuffer() all read straight out of them.
    //
    // For weights the caller keeps alive itself — a buffer it holds for the
    // run, or bytes a host process owns. A buffer the caller is done with
    // belongs in fromBytes instead, which takes ownership of it and costs
    // nothing extra to do so.
    static SafeTensors fromView(Span<const std::uint8_t> bytes);

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

    // Widened to float32 on the way out whatever the file holds, for a caller
    // that wants the values on the CPU. What the GPU gets is makeBuffer's
    // business and not necessarily this: fp16 to fp32 is exact, so widening
    // here and widening in the shader agree bit for bit.
    Vector<float> readFloats(std::string_view name) const;
    void readFloats(std::string_view name, Span<float> destination) const;

    // F32 and F16 go to the device as they lie in the blob — the first is the
    // float buffer a kernel subscripts, the second the packed one readHalf
    // reads — so neither costs a widened copy. BF16 and F64 have no shader
    // read of their own and are widened here.
    TensorBuffer makeBuffer(std::string_view name) const;

    // The same tensor as packed halves — but only when that is the same
    // tensor, which is to say when every value narrows and widens back to the
    // bits it started with. Empty otherwise, so a caller asking for half the
    // bytes never silently gets a different matrix.
    //
    // Not the rare case it sounds. OpenAI's Whisper checkpoints are fp16, and
    // HuggingFace's conversion widens them into an F32 container: all 167
    // tensors of tiny.en's model.safetensors round-trip exactly, so the fp16
    // copy of any of them is bit-identical arithmetic at half the bandwidth.
    // A repo genuinely trained and saved in fp32 answers empty and keeps the
    // float weight it shipped.
    std::optional<TensorBuffer> makeExactHalfBuffer(std::string_view name) const;

private:
    // Which of the three constructions the bytes came from. Kept as a state
    // rather than derived from a cached Span, because two of the three would
    // have to be a Span into a member of this object and this object is moved:
    // Whisper::load builds one and emplaces it into an optional.
    enum class ByteSource
    {
        Owned,
        Mapped,
        Borrowed
    };

    SafeTensors() = default;

    Span<const std::uint8_t> fileBytes() const;
    void readHeader();

    ByteSource source = ByteSource::Owned;
    Vector<std::uint8_t> ownedBytes;
    Span<const std::uint8_t> borrowedBytes;
    std::optional<eacp::MemoryMappedFile> mappedFile;
    std::uint64_t blobOffset = 0;
    Vector<TensorInfo> entries;
    std::map<std::string, std::string> metadataEntries;
};
} // namespace WSP
