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

// What a weight struct asks its projection weights to be stored as on the
// device. Float is whatever the file holds, uploaded as it lies in the blob.
// ExactHalf asks for packed fp16 and gets it wherever narrowing is bit-exact,
// which halves what every projection reads without changing a single value:
// SafeTensors::makeExactHalfBuffer keeps a narrowed tensor only when every
// value widens back to the bits it started with, and a repo that would lose
// something keeps the float weight it shipped.
//
// Not the rare case it sounds. OpenAI's Whisper checkpoints are fp16 and
// HuggingFace's conversion only widens them into an F32 container, so for a
// Whisper repo the fallback never fires — Tests/Model asserts the round-trip
// over all 37,760,256 of tiny.en's values — and the arithmetic is identical to
// the last bit: the same numbers, read through readHalf instead of a
// subscript, accumulated in float32 either way.
enum class WeightPacking
{
    Float,
    ExactHalf
};

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
// other tensor is bound to a program with no half read, so loadFloatTensor
// uploads the widened float copy of an fp16 one — which is what lets a repo
// shipped in fp16, `whisper-base` and everything above it, load at all. That
// is the whole rule: half where a program can read half, floats everywhere
// else whatever the file holds, and never a packed buffer under a float
// subscript, which would be wrong by a factor of two in every index while
// staying silent on both backends.
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

    // Bound to a program that subscripts a float buffer, so the answer is
    // floats whatever the file holds: an fp16 tensor is widened on the way to
    // the device rather than bound at half the stride it was written at.
    TensorBuffer loadFloatTensor(const std::string& name,
                                 TensorShape expected) const;

    // A projection weight, which is the one operand that may stay packed:
    // Linear has a half-reading variant and the caller picks it by storage.
    // Under WeightPacking::ExactHalf it is packed wherever that is exact and
    // uploaded as the file holds it wherever it is not, so the caller asks for
    // half the bytes rather than promising them.
    TensorBuffer loadProjectionWeight(const std::string& name,
                                      TensorShape expected,
                                      WeightPacking packing) const;
};
} // namespace WSP
