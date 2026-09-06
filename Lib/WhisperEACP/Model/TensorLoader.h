#pragma once

#include <WhisperEACP/Model/SafeTensors.h>

#include <initializer_list>
#include <string>
#include <string_view>

namespace WSP
{
// The shape a caller was compiled for, outermost axis first, as the config
// gives it rather than as the file claims it.
using TensorShape = std::initializer_list<int>;

// The checks a weight struct runs on the way from a safetensors file to the
// TensorBuffer it hands a kernel: the file carries the tensor, the tensor has
// the shape this build was compiled for, and its storage is one the program
// bound to it can read. Each failure is a ModelError naming the tensor — the
// alternative is a kernel walking a buffer at a stride it does not have, which
// is silent on both backends.
//
// `component` is the word those messages use for the caller — "encoder",
// "decoder" — so a refusal says which of the two would not take the file
// rather than leaving that to a stack trace. It is the only thing these checks
// know about who is running them, which is what lets one copy serve both.
//
// **fp16 storage.** A projection weight may stay packed: Linear has a
// half-reading variant and the caller picks it by TensorBuffer::storage. Every
// other tensor is bound to a program with no half read, so a packed one there
// would be wrong by a factor of two in every index while staying silent, and
// loadFloatTensor refuses it. That is the whole rule: half where a program can
// read half, an error everywhere else, and never a silent bind.
struct TensorLoader
{
    const SafeTensors& file;
    std::string_view component;

    const TensorInfo& require(const std::string& name) const;

    void checkShape(const TensorInfo& tensor, TensorShape expected) const;

    // The positional table is the one tensor whose first axis is a capacity
    // rather than an extent, so only its width is a shape. How many positions
    // are enough is the caller's question, and its message with it.
    void checkPositionalWidth(const TensorInfo& tensor, int width) const;

    void rejectPackedHalf(const TensorBuffer& loaded,
                          const std::string& name,
                          std::string_view reader) const;

    // Bound to a program that subscripts a float buffer, so a packed one would
    // be read at half the stride it was written at — silently, on both
    // backends. `reader` is the kernel the refusal names.
    TensorBuffer loadFloatTensor(const std::string& name,
                                 TensorShape expected,
                                 std::string_view reader) const;

    // A projection weight, which is the one operand that may stay packed:
    // Linear has a half-reading variant and the caller picks it by storage.
    TensorBuffer loadProjectionWeight(const std::string& name,
                                      TensorShape expected) const;
};
} // namespace WSP
