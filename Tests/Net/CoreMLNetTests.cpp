#include "Common.h"

#include <iostream>
#include <map>
#include <set>
#include <string>

// The encoder recorded into a Core ML program and read back, with no device and
// no Core ML: the enumerated input, the op counts the body has to lower to, and
// the blob holding every weight once under its name in the file. This half
// runs on every platform.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;

namespace
{
// tiny.en's layout at a fifth of its width, over the whole window, so the
// program is the real one's in every count but the positional table is 1500
// rows of 64 rather than of 384.
EncoderShape narrowWindowShape()
{
    auto shape = EncoderShape {};
    shape.melBins = 80;
    shape.width = 64;
    shape.heads = 2;
    shape.layers = 4;
    shape.feedForwardWidth = 256;
    shape.inputFrames = windowFrames;
    return shape;
}

int countOf(const eacp::ML::MIL::Specification& specification, std::string_view type)
{
    auto count = 0;

    for (const auto& operation: specification.program.main.block.operations)
        count += operation.type == type ? 1 : 0;

    return count;
}

bool isBlobConstant(const eacp::ML::MIL::Operation& operation)
{
    if (operation.type != "const")
        return false;

    for (const auto& attribute: operation.attributes)
        if (std::holds_alternative<eacp::ML::MIL::BlobFileValue>(
                attribute.value.value))
            return true;

    return false;
}

std::string identifierFor(std::string name)
{
    for (auto& character: name)
    {
        const auto isWord = (character >= 'a' && character <= 'z')
                            || (character >= 'A' && character <= 'Z')
                            || (character >= '0' && character <= '9')
                            || character == '_';

        if (!isWord)
            character = '_';
    }

    return name;
}

// Every encoder tensor of the file, which is every one the program needs.
Vector<std::string> encoderTensorNames(const SafeTensors& file)
{
    auto names = Vector<std::string> {};

    for (const auto& name: file.names())
        if (name.starts_with("model.encoder."))
            names.add(name);

    return names;
}

struct ProgramCheck
{
    bool fused = true;
    int layers = 0;
};

void checkProgram(const CoreMLNet& net,
                  const SafeTensors& file,
                  const ProgramCheck& expected)
{
    check(net.isValid());

    if (!net.isValid())
    {
        std::cout << "  first graph error: " << net.errors()[0] << "\n";
        return;
    }

    const auto specification = net.graph().specification();

    check(specification.program.main.opset
          == (expected.fused ? "CoreML8" : "CoreML7"));

    check(net.inputNames().size() == 1 && net.inputNames()[0] == "mel");
    check(net.outputNames().size() == 1 && net.outputNames()[0] == "rows");

    const auto& mel = specification.description.inputs[0];
    check(mel.name == "mel");
    check(mel.shape == Vector<std::int64_t> {1, 80, 3000});
    check(mel.enumeratedShapes.size() == Whisper::enumeratedAudioContextCount);

    // The default first, as eacp lists it, and every member exactly once.
    check(mel.enumeratedShapes[0] == mel.shape);

    for (auto context: Whisper::enumeratedAudioContexts())
    {
        const auto member = Vector<std::int64_t> {1, 80, 2 * context};
        auto count = 0;

        for (const auto& shape: mel.enumeratedShapes)
            count += shape == member ? 1 : 0;

        check(count == 1);
    }

    check(specification.description.outputs.size() == 1);
    check(specification.description.outputs[0].name == "rows");

    const auto layers = expected.layers;

    check(countOf(specification, "conv") == 2);
    check(countOf(specification, "gelu") == 2 + layers);
    check(countOf(specification, "layer_norm") == 2 * layers + 1);
    check(countOf(specification, "linear") == 6 * layers);
    check(countOf(specification, "shape") == 1);
    check(countOf(specification, "slice_by_index") == 1);

    if (expected.fused)
    {
        check(countOf(specification, "scaled_dot_product_attention") == layers);
        check(countOf(specification, "softmax") == 0);
    }
    else
    {
        check(countOf(specification, "scaled_dot_product_attention") == 0);
        check(countOf(specification, "softmax") == layers);
        check(countOf(specification, "matmul") == 2 * layers);
    }

    // Every weight in the blob once, under its key in the file; the only
    // unnamed ones are the key projections' zero biases.
    auto named = std::map<std::string, int> {};
    auto offsets = std::set<std::uint64_t> {};
    auto blobConstants = 0;

    for (const auto& operation: specification.program.main.block.operations)
    {
        if (!isBlobConstant(operation))
            continue;

        ++blobConstants;
        named[operation.outputs[0].name] += 1;

        for (const auto& attribute: operation.attributes)
            if (const auto* blob = std::get_if<eacp::ML::MIL::BlobFileValue>(
                    &attribute.value.value))
                offsets.insert(blob->offset);
    }

    const auto tensors = encoderTensorNames(file);

    check(blobConstants == tensors.size() + layers);
    check((int) offsets.size() == blobConstants);

    for (const auto& name: tensors)
        check(named[identifierFor(name)] == 1);

    check(net.graph().toText().find("model_encoder_layers_0_self_attn_k_proj_weight")
          != std::string::npos);
}

CoreMLNet
    recordedNet(const EncoderShape& shape, const EncoderWeights& weights, bool fused)
{
    auto net = CoreMLNet {audioContextSpan(), CoreMLNetOptions {fused, 2}};
    recordCoreMLEncoder(net, shape, weights);
    return net;
}
} // namespace

auto tRecordsTheEncoderProgram = test("Net/CoreML/recordsTheEncoderProgram") = []
{
    const auto shape = narrowWindowShape();
    const auto file = syntheticEncoderFile(shape, ProjectionStorage::Float);
    const auto weights =
        EncoderWeights {file, shape, WeightPacking::Float, WeightPlacement::Host};

    for (auto fused: {true, false})
    {
        const auto net = recordedNet(shape, weights, fused);
        checkProgram(net, file, {fused, shape.layers});
    }
};

// An fp16 file goes into the blob as its own bytes rather than widened and
// narrowed again, and makes the same program.
auto tHalfFileRecordsTheSameProgram =
    test("Net/CoreML/halfFileRecordsTheSameProgram") = []
{
    const auto shape = narrowWindowShape();
    const auto file =
        syntheticEncoderFile(shape, ProjectionStorage::EveryTensorHalf);
    const auto weights =
        EncoderWeights {file, shape, WeightPacking::Float, WeightPlacement::Host};

    const auto net = recordedNet(shape, weights, true);
    checkProgram(net, file, {true, shape.layers});

    const auto floatFile = syntheticEncoderFile(shape, ProjectionStorage::Float);
    const auto floatWeights = EncoderWeights {
        floatFile, shape, WeightPacking::Float, WeightPlacement::Host};

    check(net.build().weights.size()
          == recordedNet(shape, floatWeights, true).build().weights.size());
};

auto tRefusesASecondInputOrOutput =
    test("Net/CoreML/refusesASecondInputOrOutput") = []
{
    const auto shape = narrowWindowShape();
    const auto file = syntheticEncoderFile(shape, ProjectionStorage::Float);
    const auto weights =
        EncoderWeights {file, shape, WeightPacking::Float, WeightPlacement::Host};

    auto net = CoreMLNet {audioContextSpan(), {}};

    const auto mel =
        net.input(namedBinding({80, 3000}, "mel"), {80, 3000}, DType::float32);

    check(throwsLogicError(
        [&]
        {
            net.input(namedBinding({80, 3000}, "other"), {80, 3000}, DType::float32);
        }));

    const auto hidden = net.conv1d(net.transpose(mel),
                                   weights.firstConvolutionWeight,
                                   weights.firstConvolutionBias,
                                   1,
                                   1,
                                   Activation::gelu);

    net.output(hidden, namedBinding({}, "rows"));
    check(throwsLogicError([&] { net.output(hidden, namedBinding({}, "more")); }));

    auto other = CoreMLNet {audioContextSpan(), {}};
    check(throwsLogicError([&] { other.transpose(mel); }));
};

// The mel is the window's, and every context reads a prefix of it, so an
// input shorter than the largest context would read past its end.
auto tRefusesAnInputShorterThanItsLargestContext =
    test("Net/CoreML/refusesAnInputShorterThanItsLargestContext") = []
{
    auto net = CoreMLNet {audioContextSpan(), {}};

    check(throwsLogicError(
        [&]
        {
            net.input(namedBinding({80, 2998}, "mel"), {80, 2998}, DType::float32);
        }));

    const auto fixed = Array<int, 1> {448};
    auto fixedNet = CoreMLNet {Span<const int> {fixed.data(), 1}, {}};
    fixedNet.input(namedBinding({80, 3000}, "mel"), {80, 3000}, DType::float32);
    check(fixedNet.inputNames().size() == 1);
};

// A device tensor's file bytes view a SafeTensors that may be gone by the time
// the net reads them, so only host weights go into the blob.
auto tRefusesDeviceWeights = test("Net/CoreML/refusesDeviceWeights") = []
{
    if (!eacp::GPU::Device::shared().isValid())
        return;

    const auto shape = narrowWindowShape();
    const auto file = syntheticEncoderFile(shape, ProjectionStorage::Float);
    const auto weights =
        EncoderWeights {file, shape, WeightPacking::Float, WeightPlacement::Device};

    auto net = CoreMLNet {audioContextSpan(), {}};
    check(throwsLogicError([&] { recordCoreMLEncoder(net, shape, weights); }));
};

// The graph backend reads no buffer and runs no decoder, and says so rather
// than recording something it would not run.
auto tRefusesWhatItCannotRun = test("Net/CoreML/refusesWhatItCannotRun") = []
{
    const auto shape = narrowWindowShape();
    const auto file = syntheticEncoderFile(shape, ProjectionStorage::Float);
    const auto weights =
        EncoderWeights {file, shape, WeightPacking::Float, WeightPlacement::Host};

    auto net = CoreMLNet {audioContextSpan(), {}};

    check(throwsLogicError([&] { net.cached(Cache {}); }));
    check(throwsLogicError([&] { net.makeCache(4, 4); }));

    const auto mel =
        net.input(namedBinding({80, 3000}, "mel"), {80, 3000}, DType::float32);
    const auto frames = net.transpose(mel);

    check(throwsLogicError([&] { net.transpose(frames); }));
    check(throwsLogicError(
        [&]
        { net.layerNorm(mel, weights.finalNormWeight, weights.finalNormBias); }));
    check(throwsLogicError([&] { net.rows(weights.positionalEmbedding, 1, 4); }));

    const auto table = net.rows(weights.positionalEmbedding, 0, 4);
    check(throwsLogicError([&] { net.add(table, table); }));
    check(throwsLogicError([&] { net.output(table, namedBinding({}, "rows")); }));

    const auto hidden = net.conv1d(frames,
                                   weights.firstConvolutionWeight,
                                   weights.firstConvolutionBias,
                                   1,
                                   1,
                                   Activation::gelu);
    const auto keys =
        net.linear(hidden, weights.layers[0].keyWeight, nullptr, Activation::none);

    check(throwsLogicError([&] { net.attention(keys, keys, keys, 2, true); }));

    auto buffer =
        eacp::GPU::Device::shared().makeBuffer(16, eacp::GPU::BufferUsage::Storage);

    if (buffer.isValid())
        check(throwsLogicError(
            [&]
            {
                net.output(hidden,
                           Binding {eacp::GPU::BufferRange::of(buffer), {}, "rows"});
            }));
};

// A table shorter than the largest context the program serves is the model's
// fault, not the caller's.
auto tRefusesAShortPositionalTable =
    test("Net/CoreML/refusesAShortPositionalTable") = []
{
    auto shape = narrowWindowShape();
    shape.inputFrames = 64;

    const auto file = syntheticEncoderFile(shape, ProjectionStorage::Float);
    const auto weights =
        EncoderWeights {file, shape, WeightPacking::Float, WeightPlacement::Host};

    const auto record = [&]
    {
        auto net = CoreMLNet {audioContextSpan(), {}};
        recordEncoder(net,
                      shape,
                      weights,
                      namedBinding({shape.melBins, windowFrames}, "mel"),
                      namedBinding({}, "rows"),
                      windowFrames);
    };

    check(throwsModelError(record));
};

auto tRecordsTinyEn = test("Net/CoreML/recordsTinyEn") = []
{
    if (!hasWhisperModel())
        return;

    const auto config = ModelConfig::fromFile(modelFile("config.json"));
    const auto shape = EncoderShape::fromConfig(config, windowFrames);
    const auto file = SafeTensors::fromFile(modelFile("model.safetensors"));
    const auto weights =
        EncoderWeights {file, shape, WeightPacking::Float, WeightPlacement::Host};

    const auto start = std::chrono::steady_clock::now();
    const auto net = recordedNet(shape, weights, true);
    const auto recordSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();

    checkProgram(net, file, {true, shape.layers});

    const auto package = net.build();

    std::cout << "  tiny.en recorded in " << recordSeconds * 1000.0 << " ms, blob "
              << package.weights.size() << " bytes\n";

    // 8.2 million encoder parameters at two bytes each, in 64-byte records.
    check(package.weights.size() > 16'000'000);
    check(package.weights.size() < 17'000'000);
};
