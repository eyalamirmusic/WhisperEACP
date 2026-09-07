#pragma once

#include <WhisperEACP/Core/Core.h>

#include <cstdint>

namespace WSP
{
// The four files of a HuggingFace Whisper repo as bytes already in memory: the
// same set Whisper::load reads out of a directory, for a caller that has them
// without a filesystem to read them from — a model embedded in the binary, or
// one downloaded into a buffer.
//
// Views, not buffers. Nothing here owns a byte, and the weights in particular
// are borrowed rather than copied all the way down (SafeTensors::fromView), so
// whatever these point at has to outlive the Whisper they are handed to.
struct ModelFiles
{
    Span<const std::uint8_t> config;
    Span<const std::uint8_t> preprocessorConfig;
    Span<const std::uint8_t> tokenizer;
    Span<const std::uint8_t> weights;
};
} // namespace WSP
