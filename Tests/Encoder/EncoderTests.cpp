#include "Common.h"

#include <iostream>

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;
using namespace eacp::GPU;

namespace
{
// Small enough that a scalar reference in doubles runs the whole forward pass
// instantly, and lopsided enough that a confused stride shows up: no two
// extents equal, none a multiple of the 8 x 8 dispatch group, and a head width
// (4) that is neither the model width nor the number of heads.
EncoderShape smallShape()
{
    return {.melBins = 6,
            .width = 8,
            .heads = 2,
            .layers = 2,
            .feedForwardWidth = 16,
            .inputFrames = 12};
}

std::string modelErrorText(const std::function<void()>& body)
{
    try
    {
        body();
    }
    catch (const ModelError& error)
    {
        return error.what();
    }
    catch (...)
    {
        return "some other exception";
    }

    return "no exception";
}

bool mentions(const std::string& text, std::string_view fragment)
{
    return text.find(fragment) != std::string::npos;
}

void checkAgainstReference(ProjectionStorage projectionStorage,
                           std::string_view label,
                           double tolerance)
{
    const auto shape = smallShape();
    const auto file = syntheticEncoderFile(shape, projectionStorage);
    const auto mel = syntheticMel(shape);

    const auto result = runEncoder(shape, file, mel);
    const auto expected =
        referenceEncode(readReferenceModel(file, shape), shape, mel);

    check(result.size() == shape.elementCount());
    check((int) expected.size() == shape.elementCount());

    const auto worst = worstError(result, expected);
    std::cout << "  " << label << ": worst error " << worst << "\n";

    check(worst <= tolerance);
}
} // namespace

// No device needed: these are the numbers the whole module is indexed by, and
// getting one of them wrong misplaces every dispatch after it.
auto tEncoderShapeCounts = test("Encoder/shapeCounts") = []
{
    const auto shape = smallShape();

    check(shape.convolutionFrames() == 12);
    check(shape.positions() == 6);
    check(shape.headWidth() == 4);
    check(shape.elementCount() == 48);
    check(shape.scoreElementCount() == 72);
    check(shape.scoreRowCount() == 12);
    check(shape.feedForwardElementCount() == 96);
    check(shape.attentionScale() == 0.5f);

    // Whisper's own: 3000 mel frames become the 1500 positions
    // max_source_positions counts, because conv2 has stride two.
    auto whisper = smallShape();
    whisper.inputFrames = 3000;
    check(whisper.convolutionFrames() == 3000);
    check(whisper.positions() == 1500);
};

// The tier plan.md calls the bulk of the suite: the whole forward pass against
// a scalar reference written from HuggingFace's definition, over weights that
// went through a real safetensors header — so the loader's name and shape
// mapping is under test alongside the arithmetic.
//
// The two runs below measure 3.3e-7 and 2.5e-7, so 5e-6 is a factor of fifteen
// of room: the GPU accumulates every sum in float32 where the reference
// accumulates in double, and exp, rsqrt and the erf helper are each a float32
// intrinsic whose last digits are the backend's rather than libm's. A
// divergence that is not those is a stride, and a stride is wrong by whole
// digits rather than by the seventh one.
auto tEncoderMatchesScalarReference = test("Encoder/matchesScalarReference") = []
{
    if (!Device::shared().isValid())
        return;

    checkAgainstReference(ProjectionStorage::Float, "float weights", 5e-6);
};

// The same comparison over a file whose projection weights are fp16, which is
// what the larger repos ship: the loader leaves them packed, the encoder
// dispatches them through the half-reading Linear, and the reference runs on
// the same halves widened. Widening is exact both sides, so the tolerance is
// the float32 accumulation one again rather than fp16's three digits.
auto tEncoderMatchesScalarReferenceWithPackedWeights =
    test("Encoder/matchesScalarReferenceWithPackedWeights") = []
{
    if (!Device::shared().isValid())
        return;

    checkAgainstReference(ProjectionStorage::PackedHalf, "packed halves", 5e-6);
};

auto tEncoderMissingTensorIsAnError = test("Encoder/missingTensorIsAnError") = []
{
    const auto shape = smallShape();

    auto builder = TensorFileBuilder {};
    builder.addFloats(encoderTensor("conv1.weight"),
                      {shape.width, shape.melBins, 3},
                      spreadValues(shape.width * shape.melBins * 3, 7u, 0.4f));

    const auto file = builder.parse();
    const auto text = modelErrorText([&] { EncoderWeights {file, shape}; });

    check(mentions(text, "model.encoder.conv1.bias"));
    check(mentions(text, "the file has none"));
};

auto tEncoderWrongShapedTensorIsAnError =
    test("Encoder/wrongShapedTensorIsAnError") = []
{
    const auto shape = smallShape();
    const auto wrongWidth = shape.width + 1;

    auto builder = TensorFileBuilder {};
    builder.addFloats(encoderTensor("conv1.weight"),
                      {wrongWidth, shape.melBins, 3},
                      spreadValues(wrongWidth * shape.melBins * 3, 8u, 0.4f));

    const auto file = builder.parse();
    const auto text = modelErrorText([&] { EncoderWeights {file, shape}; });

    check(mentions(text, "model.encoder.conv1.weight"));
    check(mentions(text, "[9, 6, 3]"));
    check(mentions(text, "[8, 6, 3]"));
};

// A model that carries fewer positions than the run needs, which is the one
// axis where "not equal" is not the test.
auto tEncoderShortEmbeddingIsAnError = test("Encoder/shortEmbeddingIsAnError") = []
{
    if (!Device::shared().isValid())
        return;

    auto shape = smallShape();
    const auto file = syntheticEncoderFile(shape, ProjectionStorage::Float);

    shape.inputFrames = 40;
    const auto text = modelErrorText([&] { EncoderWeights {file, shape}; });

    check(mentions(text, "model.encoder.embed_positions.weight"));
    check(mentions(text, "carries 6 positions"));
};

// The half of the fp16 rule that is a refusal: Conv1d subscripts a float
// buffer, so a packed weight bound to it would be read at half the stride it
// was written at, silently, on both backends.
auto tEncoderPackedConvolutionIsAnError =
    test("Encoder/packedConvolutionIsAnError") = []
{
    if (!Device::shared().isValid())
        return;

    const auto shape = smallShape();

    auto builder = TensorFileBuilder {};
    builder.addHalves(encoderTensor("conv1.weight"),
                      {shape.width, shape.melBins, 3},
                      spreadValues(shape.width * shape.melBins * 3, 9u, 0.4f));

    const auto file = builder.parse();
    const auto text = modelErrorText([&] { EncoderWeights {file, shape}; });

    check(mentions(text, "model.encoder.conv1.weight"));
    check(mentions(text, "fp16"));
    check(mentions(text, "Conv1d"));
};

// Weights loaded against one shape and dispatched against another: nothing
// about a GPU buffer says which shape filled it, so the two are compared
// outright before the first dispatch.
auto tEncoderShapeMismatchIsAnError = test("Encoder/shapeMismatchIsAnError") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    const auto shape = smallShape();
    const auto file = syntheticEncoderFile(shape, ProjectionStorage::Float);
    const auto weights = EncoderWeights {file, shape};

    auto shorter = shape;
    shorter.inputFrames = 8;

    auto encoder = Encoder {shorter};
    encoder.prepare(device);

    const auto mel = storageOf(syntheticMel(shorter));
    const auto output = outputFor(shorter.elementCount());
    auto commands = device.makeCommandBuffer();
    auto pass = commands.beginCompute();

    check(throwsModelError([&] { encoder.encode(pass, mel, weights, output); }));
};

// A context shorter than the shape — whisper.cpp's audio_ctx — against the
// same scalar reference. What the reduced run has to equal is not the full
// run's first rows, which are a different answer because the model attends to
// the whole window; it is a shape built at that length in the first place,
// over that many mel frames. That is the claim, and the reference is computed
// from the shorter shape's own definition rather than from the reduced run.
auto tEncoderReducedContextMatchesTheShorterShape =
    test("Encoder/reducedContextMatchesTheShorterShape") = []
{
    if (!Device::shared().isValid())
        return;

    const auto shape = smallShape();
    const auto context = shape.positions() / 2;

    auto shorter = shape;
    shorter.inputFrames = shape.framesForPositions(context);

    check(shorter.inputFrames == 6);
    check(shorter.positions() == context);

    const auto file = syntheticEncoderFile(shape, ProjectionStorage::Float);
    const auto mel = syntheticMel(shape);

    auto result = runEncoder(shape, file, mel, context);
    const auto expected =
        referenceEncode(readReferenceModel(file, shorter),
                        shorter,
                        melPrefix(mel, shape, shorter.inputFrames));

    check(result.size() == context * shape.width);
    check((int) expected.size() == context * shape.width);

    const auto worst = worstError(result, expected);
    std::cout << "  " << context << " of " << shape.positions()
              << " positions: worst error " << worst << "\n";

    check(worst <= 5e-6);

    // And the full run is still the full run: the same encoder, asked for
    // every position again, answers what it always did.
    const auto whole = runEncoder(shape, file, mel);
    const auto reference =
        referenceEncode(readReferenceModel(file, shape), shape, mel);

    check(worstError(whole, reference) <= 5e-6);
};

// The two ends of the range and the rounding in between, none of which needs a
// device: a count is the frames it reads and the frames are the count again.
auto tEncoderContextFrames = test("Encoder/contextFrames") = []
{
    const auto shape = smallShape();

    check(shape.framesForPositions(3) == 6);
    check(shape.positions(6) == 3);
    check(shape.framesForPositions(shape.positions()) == shape.inputFrames);

    // A count past the window is the window rather than a longer run.
    check(shape.framesForPositions(shape.positions() + 4) == shape.inputFrames);

    auto whisper = smallShape();
    whisper.inputFrames = 3000;

    check(whisper.framesForPositions(768) == 1536);
    check(whisper.positions(1536) == 768);
    check(whisper.positions(2 * 64) == 64);
};

auto tEncoderTooLongAContextIsAnError = test("Encoder/tooLongAContextIsAnError") = []
{
    if (!Device::shared().isValid())
        return;

    const auto shape = smallShape();
    const auto file = syntheticEncoderFile(shape, ProjectionStorage::Float);
    const auto mel = syntheticMel(shape);

    const auto message =
        modelErrorText([&] { runEncoder(shape, file, mel, shape.positions() + 1); });

    check(mentions(message, "was asked to run over 7"));
};
