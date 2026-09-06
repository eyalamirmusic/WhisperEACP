#include "Common.h"

#include <cmath>
#include <vector>

// The reference itself, checked before anything is checked against it. No
// device: this tier is scalar C++ over a safetensors file assembled in memory,
// and it is what a GPU decoder will later be compared to — a property it fails
// here would be a property every comparison inherits.
//
// What these catch, mutation-checked rather than trusted green: dropping the
// causal mask fails referenceIsCausal, reading the logits from anything but
// embed_tokens fails referenceLogitsAreTied, and losing cross-attention
// entirely fails referenceReadsTheEncoder.
//
// What they do not catch, tried and recorded rather than left to be discovered:
// projecting self-attention's keys and values from the raw hidden state instead
// of the normalised one, and running cross-attention before self-attention.
// Both are rearrangements that stay finite, stay causal and stay tied, so
// nothing a decoder can check about itself sees them — the reference's own
// definition was verified against transformers' modeling_whisper.py instead
// (Common.h says where), and the tier that would catch them is plan.md's
// second: whisper.cpp's whisper_decode and whisper_get_logits as an oracle,
// which is not wired up yet.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;

namespace
{
std::vector<int> promptTokens()
{
    return {5, 17, 3, 11};
}

ReferenceDecoding decode(const ReferenceDecoderModel& model,
                         const DecoderShape& shape,
                         const std::vector<int>& tokens)
{
    return referenceDecode(model, shape, syntheticEncoderOutput(shape), tokens);
}
} // namespace

auto tReferenceDecoderIsFinite = test("Decoder/referenceIsFinite") = []
{
    const auto shape = smallDecoderShape();
    const auto file = syntheticDecoderFile(shape, ProjectionStorage::Float);
    const auto model = readReferenceDecoderModel(file, shape);
    const auto tokens = promptTokens();

    const auto decoded = decode(model, shape, tokens);

    check((int) decoded.hidden.size() == (int) tokens.size() * shape.width);
    check((int) decoded.logits.size() == (int) tokens.size() * shape.vocabularySize);

    for (auto value: decoded.hidden)
        check(std::isfinite(value));

    for (auto value: decoded.logits)
        check(std::isfinite(value));
};

// The causal mask, stated as the property it exists for: a token cannot see one
// that comes after it, so replacing token 2 leaves every row before it exactly
// where it was. Bit-identical rather than close — the rows before the change
// are the same arithmetic on the same inputs, and a mask that leaked would
// disagree in the first digits rather than the last.
auto tReferenceDecoderIsCausal = test("Decoder/referenceIsCausal") = []
{
    const auto shape = smallDecoderShape();
    const auto file = syntheticDecoderFile(shape, ProjectionStorage::Float);
    const auto model = readReferenceDecoderModel(file, shape);

    auto tokens = promptTokens();
    const auto before = decode(model, shape, tokens);

    tokens[2] = (tokens[2] + 7) % shape.vocabularySize;
    const auto after = decode(model, shape, tokens);

    for (auto row = 0; row < 2; ++row)
    {
        for (auto column = 0; column < shape.width; ++column)
        {
            const auto at = (std::size_t) (row * shape.width + column);
            check(before.hidden[at] == after.hidden[at]);
        }

        for (auto token = 0; token < shape.vocabularySize; ++token)
        {
            const auto at = (std::size_t) (row * shape.vocabularySize + token);
            check(before.logits[at] == after.logits[at]);
        }
    }

    // And the row that changed did change, so the comparison above is a mask
    // rather than a decoder that ignores its input.
    auto moved = false;

    for (auto column = 0; column < shape.width; ++column)
    {
        const auto at = (std::size_t) (2 * shape.width + column);
        moved = moved || before.hidden[at] != after.hidden[at];
    }

    check(moved);
};

// The logits projection has no weight of its own — there is no proj_out.weight
// in a Whisper safetensors file — so this says the row really is the hidden
// state dotted against embed_tokens, which is the tie the loader is built
// around.
auto tReferenceDecoderLogitsAreTied = test("Decoder/referenceLogitsAreTied") = []
{
    const auto shape = smallDecoderShape();
    const auto file = syntheticDecoderFile(shape, ProjectionStorage::Float);
    const auto model = readReferenceDecoderModel(file, shape);

    const auto decoded = decode(model, shape, promptTokens());
    constexpr auto row = 1;

    for (auto token = 0; token < shape.vocabularySize; ++token)
    {
        auto expected = 0.0;

        for (auto column = 0; column < shape.width; ++column)
            expected +=
                decoded.hidden[(std::size_t) (row * shape.width + column)]
                * (double) model.tokenEmbedding[token * shape.width + column];

        const auto at = (std::size_t) (row * shape.vocabularySize + token);
        check(std::abs(decoded.logits[at] - expected)
              <= 1e-12 * (1.0 + std::abs(expected)));
    }
};

// Cross-attention is the half of a decoder block that a self-attention-only
// reference would still pass every test above: change the encoder output and
// every row has to move, including the first.
auto tReferenceDecoderReadsTheEncoder = test("Decoder/referenceReadsTheEncoder") = []
{
    const auto shape = smallDecoderShape();
    const auto file = syntheticDecoderFile(shape, ProjectionStorage::Float);
    const auto model = readReferenceDecoderModel(file, shape);
    const auto tokens = promptTokens();

    const auto before = decode(model, shape, tokens);

    auto altered = syntheticEncoderOutput(shape);
    altered[0] += 1.f;

    const auto after = referenceDecode(model, shape, altered, tokens);

    check(before.hidden[0] != after.hidden[0]);
};
