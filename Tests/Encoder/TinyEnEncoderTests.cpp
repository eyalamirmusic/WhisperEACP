#include "Common.h"

#include <chrono>
#include <iostream>

// The real openai/whisper-tiny.en weights, which are a download and never a
// commit: configure with -DWHISPER_EACP_FETCH_MODEL=ON, or point
// WHISPER_MODEL_DIR at a checkout that already has them. Both tests here return
// early when the files are absent, the same shape as a GPU test returning early
// on an invalid device.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;
using namespace eacp::GPU;

namespace
{
constexpr auto configFile = "config.json";
constexpr auto weightsFile = "model.safetensors";

// 64 mel frames, which is 32 positions after conv2 — short enough that 4 layers
// of tiny.en's 384-wide blocks run in doubles on the CPU inside a test, and
// long enough that the convolutions' padded taps at both ends are exercised.
constexpr auto shortInputFrames = 64;

bool hasModel()
{
    return hasModelFile(configFile) && hasModelFile(weightsFile);
}

double secondsSince(std::chrono::steady_clock::time_point start)
{
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double>(elapsed).count();
}
} // namespace

// The same scalar reference the synthetic model is checked against, driven by
// the real weights. A shape mistake that a small random model happens to
// survive — a head width that only divides evenly at 8, a positional embedding
// read at the wrong row — has nowhere to hide here.
auto tTinyEnShortInputMatchesReference =
    test("Encoder/TinyEn/shortInputMatchesScalarReference") = []
{
    if (!hasModel() || !Device::shared().isValid())
        return;

    const auto config = ModelConfig::fromFile(modelFile(configFile));
    const auto shape = EncoderShape::fromConfig(config, shortInputFrames);

    check(shape.width == 384);
    check(shape.heads == 6);
    check(shape.headWidth() == 64);
    check(shape.layers == 4);
    check(shape.positions() == 32);

    const auto file = SafeTensors::fromFile(modelFile(weightsFile));
    const auto mel = syntheticMel(shape);

    const auto result = runEncoder(shape, file, mel);
    const auto expected =
        referenceEncode(readReferenceModel(file, shape), shape, mel);

    const auto worst = worstError(result, expected);
    std::cout << "  tiny.en over " << shortInputFrames << " frames: worst error "
              << worst << "\n";

    // 1.8e-5 measured, and 1e-4 asserted: four blocks of 384- and 1536-wide
    // float32 sums accumulate further from a double reference than the
    // two-layer synthetic model does, and the same assertion has to hold
    // against HLSL's intrinsics as against MSL's.
    check(worst <= 1e-4);
};

// The shape the model is actually run at: a full 30 second window, 1500 x 384
// out. No reference — a scalar encoder over 1500 positions is hours — so what
// this asserts is that every value came back a number, which is what a
// dispatch that ran off the end of a buffer, an unnormalised softmax row or a
// zero variance would not produce.
auto tTinyEnFullWindow = test("Encoder/TinyEn/fullWindow") = []
{
    auto& device = Device::shared();

    if (!hasModel() || !device.isValid())
        return;

    const auto config = ModelConfig::fromFile(modelFile(configFile));
    const auto shape = EncoderShape::fromConfig(config, config.encoderInputFrames());

    check(shape.inputFrames == 3000);
    check(shape.positions() == config.maxSourcePositions);

    const auto file = SafeTensors::fromFile(modelFile(weightsFile));
    const auto weights = EncoderWeights {file, shape};

    const auto mel = storageOf(syntheticMel(shape));
    const auto output = outputFor(shape.elementCount());

    auto encoder = Encoder {shape};
    encoder.prepare(device);

    auto commands = device.makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        encoder.encode(pass, mel, weights, output);
    }

    const auto start = std::chrono::steady_clock::now();
    commands.commit();
    const auto elapsed = secondsSince(start);

    const auto result = readBack(output, shape.elementCount());
    check(result.size() == 1500 * 384);

    auto largest = 0.f;

    for (auto index = 0; index < result.size(); ++index)
    {
        check(std::isfinite(result[index]));
        largest = std::max(largest, std::abs(result[index]));
    }

    std::cout << "  tiny.en over 3000 frames: " << elapsed << " s on the GPU, "
              << "largest output " << largest << "\n";
};

// The real weights over a context shorter than the shape, against the same
// scalar reference: 64 mel frames loaded, 32 of them read, and 16 of the
// shape's 32 positions computed. A reduced run is a shape built at that length
// — that is what whisper.cpp's audio_ctx computes and what this asserts — so
// the reference is the 32-frame shape's own forward pass, over the 32-frame
// prefix of the same mel.
auto tTinyEnReducedContextMatchesReference =
    test("Encoder/TinyEn/reducedContextMatchesScalarReference") = []
{
    if (!hasModel() || !Device::shared().isValid())
        return;

    const auto config = ModelConfig::fromFile(modelFile(configFile));
    const auto shape = EncoderShape::fromConfig(config, shortInputFrames);
    const auto context = shape.positions() / 2;

    auto shorter = shape;
    shorter.inputFrames = shape.framesForPositions(context);

    check(context == 16);
    check(shorter.inputFrames == 32);
    check(shorter.positions() == context);

    const auto file = SafeTensors::fromFile(modelFile(weightsFile));
    const auto mel = syntheticMel(shape);

    const auto result = runEncoder(shape, file, mel, context);
    const auto expected =
        referenceEncode(readReferenceModel(file, shorter),
                        shorter,
                        melPrefix(mel, shape, shorter.inputFrames));

    const auto worst = worstError(result, expected);
    std::cout << "  tiny.en over " << context << " of " << shape.positions()
              << " positions: worst error " << worst << "\n";

    check(worst <= 1e-4);
};

// What WeightPacking::ExactHalf actually did to the real weights: tiny.en's
// tensors are fp16 values in an F32 container, so every one of the six
// projections of every layer narrows exactly and none of them falls back to
// the float upload. 28 MB of encoder weights read as 14.
//
// The fallback is the thing worth asserting against. It is silent by design —
// a weight that would lose something keeps the float it shipped — so a bug
// that stopped the packing from ever firing would leave every other assertion
// in this suite passing and the bandwidth exactly where it was.
auto tTinyEnEncoderPacksEveryProjection =
    test("Encoder/TinyEn/exactHalfPacksEveryProjection") = []
{
    if (!hasModel() || !Device::shared().isValid())
        return;

    const auto config = ModelConfig::fromFile(modelFile(configFile));
    const auto shape = EncoderShape::fromConfig(config, shortInputFrames);

    const auto file = SafeTensors::fromFile(modelFile(weightsFile));
    const auto weights = EncoderWeights {file, shape, WeightPacking::ExactHalf};

    for (const auto& layer: weights.layers)
    {
        check(layer.queryWeight.isPackedHalf());
        check(layer.keyWeight.isPackedHalf());
        check(layer.valueWeight.isPackedHalf());
        check(layer.attentionOutputWeight.isPackedHalf());
        check(layer.feedForwardWeight.isPackedHalf());
        check(layer.feedForwardOutputWeight.isPackedHalf());

        // The biases and the norms are subscripted inside Linear and
        // LayerNorm, neither of which has a packed read, so the policy leaves
        // them alone.
        check(!layer.queryBias.isPackedHalf());
        check(!layer.attentionNormWeight.isPackedHalf());
    }

    // The convolutions and the positional rows are float as well: the policy
    // is over the projections, and those three go through the loader's float
    // path whatever it is set to.
    check(!weights.firstConvolutionWeight.isPackedHalf());
    check(!weights.secondConvolutionWeight.isPackedHalf());
    check(!weights.positionalEmbedding.isPackedHalf());
};
