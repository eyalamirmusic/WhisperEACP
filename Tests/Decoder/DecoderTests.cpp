#include "Common.h"

#include <iostream>
#include <string>
#include <vector>

// The GPU decoder against the scalar reference, mutation-checked rather than
// trusted green. Each of these was applied to Decoder.cpp and the suite rerun:
//
//   running cross-attention before self-attention        6 of these fail
//   projecting k and v from the unnormalised hidden      6 fail
//   binding the cache one row past where it belongs      6 fail
//   embedding at position zero on every step             3 fail
//   dropping the causal flag                             5 fail
//
// The last one is the one worth reading twice: cachedStepsMatchTheReference
// survives it, and has to. A step of one query against n cached keys masks
// nothing — the query stands at the last of them — so the mask is only load
// bearing when a step carries more than one token, which is why the prompt is
// decoded in one call in half the tests here and one token at a time in the
// other half.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;
using namespace eacp::GPU;

namespace
{
// The prompt the small-shape tests decode: four ids, none of them zero, none
// repeated, and all inside a 24-token vocabulary — so a gather reading the
// wrong row lands on a row that exists and produces numbers rather than
// failing.
std::vector<int> promptTokens()
{
    return {5, 17, 3, 11};
}

// Both halves of a step against the reference rows it should have reproduced,
// reported as numbers rather than only as a pass: a tolerance nobody has
// measured against is a tolerance that was guessed.
double checkStepAgainstReference(const StepResult& result,
                                 const ReferenceDecoding& expected,
                                 const DecoderShape& shape,
                                 int firstRow,
                                 int rowCount,
                                 double tolerance)
{
    const auto hidden = worstError(
        result.hidden, rowsOf(expected.hidden, firstRow, rowCount, shape.width));

    const auto logits = worstError(
        result.logits,
        rowsOf(expected.logits, firstRow, rowCount, shape.logitElementCount()));

    check(hidden <= tolerance);
    check(logits <= tolerance);

    for (auto row = 0; row < rowCount; ++row)
        check((int) result.argmax[row]
              == argmaxRow(
                  expected.logits, firstRow + row, shape.logitElementCount()));

    return std::max(hidden, logits);
}

void checkPromptInOneStep(ProjectionStorage projectionStorage,
                          std::string_view label,
                          double tolerance)
{
    const auto shape = smallDecoderShape();
    const auto file = syntheticDecoderFile(shape, projectionStorage);
    const auto encoderOutput = syntheticEncoderOutput(shape);
    const auto tokens = promptTokens();

    auto run = DecoderRun {shape, file};
    run.begin(encoderOutput);

    const auto result = run.step(tokens);
    const auto expected = referenceDecode(
        readReferenceDecoderModel(file, shape), shape, encoderOutput, tokens);

    check(run.position() == (int) tokens.size());

    const auto worst = checkStepAgainstReference(
        result, expected, shape, 0, (int) tokens.size(), tolerance);

    std::cout << "  " << label << ": worst error " << worst << "\n";
}
} // namespace

// The tier plan.md calls the bulk of the suite: a whole prompt through the GPU
// decoder in one step, against a scalar reference written from HuggingFace's
// definition, over weights that went through a real safetensors header — so the
// loader's name and shape mapping is under test alongside the arithmetic.
//
// 3.6e-7 measured against 5e-6 asserted, and the four small-shape comparisons
// below span 1.8e-7 to 4.1e-7 — the encoder's tolerance for the encoder's
// reason: the GPU accumulates every sum in float32 where the reference
// accumulates in double, and exp, rsqrt and the erf helper are each a float32
// intrinsic whose last digits are the backend's rather than libm's. A
// divergence that is not those is a stride, and a stride is wrong by whole
// digits rather than by the seventh one.
auto tDecoderPromptMatchesReference = test("Decoder/promptMatchesReference") = []
{
    if (!Device::shared().isValid())
        return;

    checkPromptInOneStep(ProjectionStorage::Float, "prompt in one step", 5e-6);
};

// The KV cache, stated as the property it exists for: the same tokens fed one
// at a time have to produce the same rows the prompt did. The reference is what
// both are compared to rather than to each other, so a cache that agreed with a
// prompt path that was itself wrong would still fail.
auto tDecoderCachedStepsMatchTheReference =
    test("Decoder/cachedStepsMatchTheReference") = []
{
    if (!Device::shared().isValid())
        return;

    const auto shape = smallDecoderShape();
    const auto file = syntheticDecoderFile(shape, ProjectionStorage::Float);
    const auto encoderOutput = syntheticEncoderOutput(shape);
    const auto tokens = promptTokens();

    const auto expected = referenceDecode(
        readReferenceDecoderModel(file, shape), shape, encoderOutput, tokens);

    auto run = DecoderRun {shape, file};
    run.begin(encoderOutput);

    auto worst = 0.0;

    for (auto index = 0; index < (int) tokens.size(); ++index)
    {
        const auto result = run.step({tokens[(std::size_t) index]});
        check(run.position() == index + 1);

        worst = std::max(
            worst,
            checkStepAgainstReference(result, expected, shape, index, 1, 5e-6));
    }

    std::cout << "  one token at a time: worst error " << worst << "\n";
};

// A prompt, then single steps, then a second sequence on a different encoder
// output — which is what a run of two utterances is, and the only thing that
// checks the reset: the cross-attention keys and values are re-projected, the
// position goes back to zero, and the self-attention cache is overwritten from
// row zero rather than appended to.
auto tDecoderMixedStepsAndASecondSequence =
    test("Decoder/mixedStepsAndASecondSequence") = []
{
    if (!Device::shared().isValid())
        return;

    const auto shape = smallDecoderShape();
    const auto file = syntheticDecoderFile(shape, ProjectionStorage::Float);
    const auto model = readReferenceDecoderModel(file, shape);
    const auto tokens = promptTokens();

    const auto firstEncoderOutput = syntheticEncoderOutput(shape);
    const auto firstExpected =
        referenceDecode(model, shape, firstEncoderOutput, tokens);

    auto run = DecoderRun {shape, file};
    run.begin(firstEncoderOutput);

    auto worst = 0.0;

    const auto opening = run.step({tokens[0], tokens[1]});
    worst = std::max(
        worst, checkStepAgainstReference(opening, firstExpected, shape, 0, 2, 5e-6));

    for (auto index = 2; index < (int) tokens.size(); ++index)
    {
        const auto result = run.step({tokens[(std::size_t) index]});
        worst = std::max(
            worst,
            checkStepAgainstReference(result, firstExpected, shape, index, 1, 5e-6));
    }

    check(run.position() == (int) tokens.size());

    auto secondEncoderOutput = syntheticEncoderOutput(shape);

    for (auto index = 0; index < secondEncoderOutput.size(); ++index)
        secondEncoderOutput[index] = -0.5f * secondEncoderOutput[index] + 0.25f;

    const auto secondTokens = std::vector<int> {7, 2, 9};
    const auto secondExpected =
        referenceDecode(model, shape, secondEncoderOutput, secondTokens);

    run.begin(secondEncoderOutput);
    check(run.position() == 0);

    const auto second = run.step(secondTokens);
    check(run.position() == (int) secondTokens.size());

    worst = std::max(
        worst,
        checkStepAgainstReference(
            second, secondExpected, shape, 0, (int) secondTokens.size(), 5e-6));

    std::cout << "  two sequences: worst error " << worst << "\n";
};

// The same comparison over a file whose projection weights are fp16, which is
// what the larger repos ship: the loader leaves them packed, the decoder
// dispatches them through the half-reading Linear, and the reference runs on
// the same halves widened. Widening is exact both sides, so the tolerance is
// the float32 accumulation one again rather than fp16's three digits.
//
// embed_tokens stays F32 in both runs, because the loader rejects a packed one:
// it is the gather's table as well as the logits projection's weight, and the
// gather subscripts floats.
auto tDecoderMatchesReferenceWithPackedWeights =
    test("Decoder/matchesReferenceWithPackedWeights") = []
{
    if (!Device::shared().isValid())
        return;

    checkPromptInOneStep(ProjectionStorage::PackedHalf, "packed halves", 5e-6);
};

// Greedy sampling, which is the layer above the decoder: the Argmax kernel over
// the GPU's own logits with an all-zero mask, against argmaxRow over the
// reference's. The same assertion rides along inside every comparison above and
// in the tiny.en tests, where the row is 51864 wide; this is where it is stated.
auto tDecoderGreedyArgmaxAgreesWithTheReference =
    test("Decoder/greedyArgmaxAgreesWithTheReference") = []
{
    if (!Device::shared().isValid())
        return;

    const auto shape = smallDecoderShape();
    const auto file = syntheticDecoderFile(shape, ProjectionStorage::Float);
    const auto encoderOutput = syntheticEncoderOutput(shape);
    const auto tokens = promptTokens();

    auto run = DecoderRun {shape, file};
    run.begin(encoderOutput);

    const auto result = run.step(tokens);
    const auto expected = referenceDecode(
        readReferenceDecoderModel(file, shape), shape, encoderOutput, tokens);

    check(result.argmax.size() == (int) tokens.size());

    for (auto row = 0; row < (int) tokens.size(); ++row)
        check((int) result.argmax[row]
              == argmaxRow(expected.logits, row, shape.logitElementCount()));
};

// A step past the end of the window, which is the one bound the decoder has to
// enforce itself: the cache row it would bind is outside the buffer, and a
// BufferRange past a buffer's end binds nothing at all rather than failing. So
// the check is on the host, before anything is recorded — and "before" is what
// the untouched logits say.
auto tDecoderStepPastTheWindowIsAnError =
    test("Decoder/stepPastTheWindowIsAnError") = []
{
    if (!Device::shared().isValid())
        return;

    const auto shape = smallDecoderShape();
    const auto file = syntheticDecoderFile(shape, ProjectionStorage::Float);

    auto run = DecoderRun {shape, file};
    run.begin(syntheticEncoderOutput(shape));

    auto whole = std::vector<int> {};

    for (auto index = 0; index < shape.maxPositions; ++index)
        whole.push_back(index % shape.vocabularySize);

    run.step(whole);
    check(run.position() == shape.maxPositions);

    check(run.stepThrowsLeavingLogitsUntouched({1}));
    check(run.position() == shape.maxPositions);
};
