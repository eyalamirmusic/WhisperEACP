#pragma once

#include <optional>
#include <string_view>

namespace WSP
{
// The dtype tags safetensors names in its header. F32 is what
// openai/whisper-tiny.en actually ships; F16 is what the larger repos use. The
// integer tags are here so an unknown one is an error rather than a silent
// misread of the blob.
enum class TensorType
{
    Bool,
    U8,
    I8,
    U16,
    I16,
    F16,
    BF16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64
};

int bytesPerElement(TensorType type);
bool isFloatingPoint(TensorType type);
std::string_view tensorTypeName(TensorType type);

// Empty for a tag this build does not know, which is the whole reason it
// returns an optional rather than a default.
std::optional<TensorType> findTensorType(std::string_view name);
} // namespace WSP
