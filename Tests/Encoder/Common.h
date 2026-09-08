#pragma once

#include <WhisperEACP/Encoder/Encoder.h>

// The two suites next door rather than a third copy of what they hold: the GPU
// plumbing and the spread of inputs from Tests/Kernels, and the safetensors
// assembly and the model-directory skips from Tests/Model. This is the module
// that is assembled out of the other two, so its tests are as well.
#include "../Kernels/Common.h"
#include "../Model/Common.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace WSP::Testing
{
// The tensor names, written out here rather than taken from the loader: this is
// what says the loader looks for the names HuggingFace actually ships, and a
// list shared with the code under test could not say it.
inline std::string encoderTensor(std::string_view suffix)
{
    return "model.encoder." + std::string {suffix};
}

inline std::string encoderLayerTensor(int index, std::string_view suffix)
{
    return encoderTensor("layers.") + std::to_string(index) + "."
           + std::string {suffix};
}

// ---------------------------------------------------------------------------
// A safetensors file assembled in memory, so the loader's name and shape
// mapping is under test alongside the arithmetic.
// ---------------------------------------------------------------------------

// Round to nearest, ties to even, for the values below — every one of which
// fp16 holds as a normal, since spreadValues never lands nearer zero than a
// thousandth of its range. Infinities and subnormals do not arise and are not
// handled.
inline std::uint16_t packHalf(float value)
{
    auto bits = std::uint32_t {};
    std::memcpy(&bits, &value, sizeof(bits));

    const auto sign = (std::uint16_t) ((bits >> 16) & 0x8000u);
    const auto magnitude = bits & 0x7FFFFFFFu;

    if (magnitude == 0)
        return sign;

    const auto exponent = (int) ((magnitude >> 23) & 0xFFu) - 127 + 15;
    const auto mantissa = magnitude & 0x7FFFFFu;
    const auto packed =
        (std::uint16_t) (sign | (unsigned) (exponent << 10) | (mantissa >> 13));

    const auto remainder = mantissa & 0x1FFFu;
    const auto roundsUp =
        remainder > 0x1000u || (remainder == 0x1000u && (packed & 1u) != 0u);

    return (std::uint16_t) (packed + (roundsUp ? 1u : 0u));
}

class TensorFileBuilder
{
public:
    void addFloats(const std::string& name,
                   const std::vector<int>& shape,
                   const Vector<float>& values)
    {
        const auto begin = (std::uint64_t) blob.size();

        for (auto index = 0; index < values.size(); ++index)
        {
            auto bytes = std::uint32_t {};
            std::memcpy(&bytes, &values[index], sizeof(bytes));

            for (auto byte = 0; byte < 4; ++byte)
                blob.add((std::uint8_t) ((bytes >> (byte * 8)) & 0xFFu));
        }

        entries.push_back({name, "F32", shape, begin, (std::uint64_t) blob.size()});
    }

    void addHalves(const std::string& name,
                   const std::vector<int>& shape,
                   const Vector<float>& values)
    {
        const auto begin = (std::uint64_t) blob.size();

        for (auto index = 0; index < values.size(); ++index)
        {
            const auto bits = packHalf(values[index]);
            blob.add((std::uint8_t) (bits & 0xFFu));
            blob.add((std::uint8_t) (bits >> 8));
        }

        entries.push_back({name, "F16", shape, begin, (std::uint64_t) blob.size()});
    }

    SafeTensors parse() const
    {
        return SafeTensors::fromBytes(assemble(header(), blob));
    }

private:
    struct Entry
    {
        std::string name;
        std::string dtype;
        std::vector<int> shape;
        std::uint64_t begin = 0;
        std::uint64_t end = 0;
    };

    std::string header() const
    {
        auto text = std::string {"{"};

        for (const auto& entry: entries)
        {
            if (text.size() > 1)
                text += ",";

            text += "\"" + entry.name + "\":{\"dtype\":\"" + entry.dtype
                    + "\",\"shape\":[";

            for (auto axis = 0u; axis < entry.shape.size(); ++axis)
                text += (axis == 0 ? "" : ",") + std::to_string(entry.shape[axis]);

            text += "],\"data_offsets\":[" + std::to_string(entry.begin) + ","
                    + std::to_string(entry.end) + "]}";
        }

        return text + "}";
    }

    std::vector<Entry> entries;
    Vector<std::uint8_t> blob;
};

// A trained layer norm's weight sits near one, and a test whose norms scale
// their rows to nearly nothing would compare numbers no arithmetic could get
// wrong.
inline Vector<float> shiftedValues(int count, unsigned seed, float range)
{
    auto values = spreadValues(count, seed, range);

    for (auto index = 0; index < values.size(); ++index)
        values[index] += 1.f;

    return values;
}

// Which tensors go in packed, for the two runs of the same comparison: none of
// them, or every projection weight — the only ones a program in Kernels/ can
// read as halves.
enum class ProjectionStorage
{
    Float,
    PackedHalf
};

inline SafeTensors syntheticEncoderFile(const EncoderShape& shape,
                                        ProjectionStorage projectionStorage)
{
    auto builder = TensorFileBuilder {};
    auto seed = 1000u;

    auto next = [&](int count, float range)
    { return spreadValues(count, seed++, range); };

    auto addProjection =
        [&](const std::string& name, int outputWidth, int innerCount)
    {
        const auto values = next(outputWidth * innerCount, 0.5f);
        const auto extents = std::vector<int> {outputWidth, innerCount};

        if (projectionStorage == ProjectionStorage::PackedHalf)
            builder.addHalves(name, extents, values);
        else
            builder.addFloats(name, extents, values);
    };

    const auto width = shape.width;
    const auto kernelSize = EncoderShape::convolutionKernelSize;

    builder.addFloats(encoderTensor("conv1.weight"),
                      {width, shape.melBins, kernelSize},
                      next(width * shape.melBins * kernelSize, 0.4f));
    builder.addFloats(encoderTensor("conv1.bias"), {width}, next(width, 0.2f));

    builder.addFloats(encoderTensor("conv2.weight"),
                      {width, width, kernelSize},
                      next(width * width * kernelSize, 0.4f));
    builder.addFloats(encoderTensor("conv2.bias"), {width}, next(width, 0.2f));

    builder.addFloats(encoderTensor("embed_positions.weight"),
                      {shape.positions(), width},
                      next(shape.positions() * width, 0.3f));

    for (auto layer = 0; layer < shape.layers; ++layer)
    {
        builder.addFloats(encoderLayerTensor(layer, "self_attn_layer_norm.weight"),
                          {width},
                          shiftedValues(width, seed++, 0.2f));
        builder.addFloats(encoderLayerTensor(layer, "self_attn_layer_norm.bias"),
                          {width},
                          next(width, 0.2f));

        addProjection(
            encoderLayerTensor(layer, "self_attn.q_proj.weight"), width, width);
        builder.addFloats(encoderLayerTensor(layer, "self_attn.q_proj.bias"),
                          {width},
                          next(width, 0.2f));

        addProjection(
            encoderLayerTensor(layer, "self_attn.k_proj.weight"), width, width);

        addProjection(
            encoderLayerTensor(layer, "self_attn.v_proj.weight"), width, width);
        builder.addFloats(encoderLayerTensor(layer, "self_attn.v_proj.bias"),
                          {width},
                          next(width, 0.2f));

        addProjection(
            encoderLayerTensor(layer, "self_attn.out_proj.weight"), width, width);
        builder.addFloats(encoderLayerTensor(layer, "self_attn.out_proj.bias"),
                          {width},
                          next(width, 0.2f));

        builder.addFloats(encoderLayerTensor(layer, "final_layer_norm.weight"),
                          {width},
                          shiftedValues(width, seed++, 0.2f));
        builder.addFloats(encoderLayerTensor(layer, "final_layer_norm.bias"),
                          {width},
                          next(width, 0.2f));

        addProjection(
            encoderLayerTensor(layer, "fc1.weight"), shape.feedForwardWidth, width);
        builder.addFloats(encoderLayerTensor(layer, "fc1.bias"),
                          {shape.feedForwardWidth},
                          next(shape.feedForwardWidth, 0.2f));

        addProjection(
            encoderLayerTensor(layer, "fc2.weight"), width, shape.feedForwardWidth);
        builder.addFloats(
            encoderLayerTensor(layer, "fc2.bias"), {width}, next(width, 0.2f));
    }

    builder.addFloats(encoderTensor("layer_norm.weight"),
                      {width},
                      shiftedValues(width, seed++, 0.2f));
    builder.addFloats(encoderTensor("layer_norm.bias"), {width}, next(width, 0.2f));

    return builder.parse();
}

// ---------------------------------------------------------------------------
// The scalar reference: HuggingFace's WhisperEncoder.forward in doubles,
// written from the definition rather than out of the kernels' own references,
// since the chain is what this checks.
// ---------------------------------------------------------------------------

struct ReferenceLayer
{
    Vector<float> attentionNormWeight;
    Vector<float> attentionNormBias;
    Vector<float> queryWeight;
    Vector<float> queryBias;
    Vector<float> keyWeight;
    Vector<float> valueWeight;
    Vector<float> valueBias;
    Vector<float> attentionOutputWeight;
    Vector<float> attentionOutputBias;
    Vector<float> finalNormWeight;
    Vector<float> finalNormBias;
    Vector<float> feedForwardWeight;
    Vector<float> feedForwardBias;
    Vector<float> feedForwardOutputWeight;
    Vector<float> feedForwardOutputBias;
};

struct ReferenceModel
{
    Vector<float> firstConvolutionWeight;
    Vector<float> firstConvolutionBias;
    Vector<float> secondConvolutionWeight;
    Vector<float> secondConvolutionBias;
    Vector<float> positionalEmbedding;
    Vector<float> finalNormWeight;
    Vector<float> finalNormBias;
    std::vector<ReferenceLayer> layers;
};

// Read back through readFloats, which widens whatever the file holds — so the
// reference runs on exactly the numbers the kernels read, and a packed weight
// costs the comparison nothing.
inline ReferenceModel readReferenceModel(const SafeTensors& file,
                                         const EncoderShape& shape)
{
    auto model = ReferenceModel {};
    model.firstConvolutionWeight = file.readFloats(encoderTensor("conv1.weight"));
    model.firstConvolutionBias = file.readFloats(encoderTensor("conv1.bias"));
    model.secondConvolutionWeight = file.readFloats(encoderTensor("conv2.weight"));
    model.secondConvolutionBias = file.readFloats(encoderTensor("conv2.bias"));
    model.positionalEmbedding =
        file.readFloats(encoderTensor("embed_positions.weight"));
    model.finalNormWeight = file.readFloats(encoderTensor("layer_norm.weight"));
    model.finalNormBias = file.readFloats(encoderTensor("layer_norm.bias"));

    for (auto index = 0; index < shape.layers; ++index)
    {
        auto read = [&](std::string_view suffix)
        { return file.readFloats(encoderLayerTensor(index, suffix)); };

        model.layers.push_back({read("self_attn_layer_norm.weight"),
                                read("self_attn_layer_norm.bias"),
                                read("self_attn.q_proj.weight"),
                                read("self_attn.q_proj.bias"),
                                read("self_attn.k_proj.weight"),
                                read("self_attn.v_proj.weight"),
                                read("self_attn.v_proj.bias"),
                                read("self_attn.out_proj.weight"),
                                read("self_attn.out_proj.bias"),
                                read("final_layer_norm.weight"),
                                read("final_layer_norm.bias"),
                                read("fc1.weight"),
                                read("fc1.bias"),
                                read("fc2.weight"),
                                read("fc2.bias")});
    }

    return model;
}

inline double referenceGelu(double x)
{
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// [outputChannels, outputLength] channel-major, out of [inputChannels,
// inputLength] channel-major, with the padded taps skipped rather than read
// from a zero border.
inline std::vector<double> referenceConv1d(const std::vector<double>& input,
                                           int inputChannels,
                                           int inputLength,
                                           const Vector<float>& weight,
                                           const Vector<float>& bias,
                                           int outputChannels,
                                           int stride)
{
    constexpr auto kernelSize = EncoderShape::convolutionKernelSize;
    constexpr auto padding = EncoderShape::convolutionPadding;

    const auto outputLength =
        conv1dOutputLength(inputLength, kernelSize, stride, padding);

    auto output = std::vector<double>((std::size_t) (outputChannels * outputLength));

    for (auto channel = 0; channel < outputChannels; ++channel)
    {
        for (auto frame = 0; frame < outputLength; ++frame)
        {
            auto total = (double) bias[channel];

            for (auto source = 0; source < inputChannels; ++source)
            {
                for (auto tap = 0; tap < kernelSize; ++tap)
                {
                    const auto at = frame * stride + tap - padding;

                    if (at < 0 || at >= inputLength)
                        continue;

                    const auto weightAt =
                        (channel * inputChannels + source) * kernelSize + tap;

                    total += (double) weight[weightAt]
                             * input[(std::size_t) (source * inputLength + at)];
                }
            }

            output[(std::size_t) (channel * outputLength + frame)] = total;
        }
    }

    return output;
}

inline std::vector<double> referenceLayerNorm(const std::vector<double>& rows,
                                              int rowCount,
                                              int width,
                                              const Vector<float>& weight,
                                              const Vector<float>& bias)
{
    auto output = std::vector<double>(rows.size());

    for (auto row = 0; row < rowCount; ++row)
    {
        const auto base = (std::size_t) (row * width);
        auto mean = 0.0;

        for (auto column = 0; column < width; ++column)
            mean += rows[base + (std::size_t) column];

        mean /= width;
        auto variance = 0.0;

        for (auto column = 0; column < width; ++column)
        {
            const auto centred = rows[base + (std::size_t) column] - mean;
            variance += centred * centred;
        }

        const auto scale =
            1.0 / std::sqrt(variance / width + LayerNorm::whisperEpsilon);

        for (auto column = 0; column < width; ++column)
            output[base + (std::size_t) column] =
                (rows[base + (std::size_t) column] - mean) * scale * weight[column]
                + bias[column];
    }

    return output;
}

// nn.Linear's own indexing: weight is [outputWidth, innerCount], one output's
// whole input row contiguous. A null bias is k_proj's.
inline std::vector<double> referenceLinear(const std::vector<double>& input,
                                           int rowCount,
                                           int innerCount,
                                           const Vector<float>& weight,
                                           const Vector<float>* bias,
                                           int outputWidth)
{
    auto output = std::vector<double>((std::size_t) (rowCount * outputWidth));

    for (auto row = 0; row < rowCount; ++row)
    {
        for (auto column = 0; column < outputWidth; ++column)
        {
            auto total = bias == nullptr ? 0.0 : (double) (*bias)[column];

            for (auto step = 0; step < innerCount; ++step)
                total += input[(std::size_t) (row * innerCount + step)]
                         * (double) weight[column * innerCount + step];

            output[(std::size_t) (row * outputWidth + column)] = total;
        }
    }

    return output;
}

inline std::vector<double> referenceAttention(const std::vector<double>& queries,
                                              const std::vector<double>& keys,
                                              const std::vector<double>& values,
                                              const EncoderShape& shape)
{
    const auto width = shape.width;
    const auto positions = shape.positions();
    const auto headWidth = shape.headWidth();
    const auto scale = 1.0 / std::sqrt((double) headWidth);

    auto output = std::vector<double>((std::size_t) (positions * width));
    auto row = std::vector<double>((std::size_t) positions);

    for (auto head = 0; head < shape.heads; ++head)
    {
        const auto headColumn = head * headWidth;

        for (auto query = 0; query < positions; ++query)
        {
            auto largest = -std::numeric_limits<double>::infinity();

            for (auto key = 0; key < positions; ++key)
            {
                auto total = 0.0;

                for (auto channel = 0; channel < headWidth; ++channel)
                    total +=
                        queries[(std::size_t) (query * width + headColumn + channel)]
                        * keys[(std::size_t) (key * width + headColumn + channel)];

                row[(std::size_t) key] = scale * total;
                largest = std::max(largest, row[(std::size_t) key]);
            }

            auto total = 0.0;

            for (auto key = 0; key < positions; ++key)
            {
                row[(std::size_t) key] = std::exp(row[(std::size_t) key] - largest);
                total += row[(std::size_t) key];
            }

            for (auto channel = 0; channel < headWidth; ++channel)
            {
                auto weighted = 0.0;

                for (auto key = 0; key < positions; ++key)
                    weighted +=
                        row[(std::size_t) key]
                        * values[(std::size_t) (key * width + headColumn + channel)];

                output[(std::size_t) (query * width + headColumn + channel)] =
                    weighted / total;
            }
        }
    }

    return output;
}

inline std::vector<double> referenceEncode(const ReferenceModel& model,
                                           const EncoderShape& shape,
                                           const Vector<float>& mel)
{
    const auto width = shape.width;
    const auto positions = shape.positions();

    auto input = std::vector<double>((std::size_t) mel.size());

    for (auto index = 0; index < mel.size(); ++index)
        input[(std::size_t) index] = mel[index];

    auto convolved = referenceConv1d(input,
                                     shape.melBins,
                                     shape.inputFrames,
                                     model.firstConvolutionWeight,
                                     model.firstConvolutionBias,
                                     width,
                                     EncoderShape::firstConvolutionStride);

    for (auto& value: convolved)
        value = referenceGelu(value);

    auto projected = referenceConv1d(convolved,
                                     width,
                                     shape.convolutionFrames(),
                                     model.secondConvolutionWeight,
                                     model.secondConvolutionBias,
                                     width,
                                     EncoderShape::secondConvolutionStride);

    for (auto& value: projected)
        value = referenceGelu(value);

    // Channel-major out of the convolution, row-major into the transformer,
    // which is the permute the kernel's output strides do in the store — so the
    // reference does it by hand and the two have to agree.
    auto hidden = std::vector<double>((std::size_t) (positions * width));

    for (auto position = 0; position < positions; ++position)
        for (auto column = 0; column < width; ++column)
            hidden[(std::size_t) (position * width + column)] =
                projected[(std::size_t) (column * positions + position)]
                + model.positionalEmbedding[position * width + column];

    for (const auto& layer: model.layers)
    {
        const auto normalised = referenceLayerNorm(hidden,
                                                   positions,
                                                   width,
                                                   layer.attentionNormWeight,
                                                   layer.attentionNormBias);

        const auto queries = referenceLinear(normalised,
                                             positions,
                                             width,
                                             layer.queryWeight,
                                             &layer.queryBias,
                                             width);

        const auto keys = referenceLinear(
            normalised, positions, width, layer.keyWeight, nullptr, width);

        const auto values = referenceLinear(normalised,
                                            positions,
                                            width,
                                            layer.valueWeight,
                                            &layer.valueBias,
                                            width);

        const auto attended = referenceAttention(queries, keys, values, shape);

        const auto attentionOutput = referenceLinear(attended,
                                                     positions,
                                                     width,
                                                     layer.attentionOutputWeight,
                                                     &layer.attentionOutputBias,
                                                     width);

        for (auto index = 0u; index < hidden.size(); ++index)
            hidden[index] += attentionOutput[index];

        const auto renormalised = referenceLayerNorm(
            hidden, positions, width, layer.finalNormWeight, layer.finalNormBias);

        auto feedForward = referenceLinear(renormalised,
                                           positions,
                                           width,
                                           layer.feedForwardWeight,
                                           &layer.feedForwardBias,
                                           shape.feedForwardWidth);

        for (auto& value: feedForward)
            value = referenceGelu(value);

        const auto feedForwardOutput = referenceLinear(feedForward,
                                                       positions,
                                                       shape.feedForwardWidth,
                                                       layer.feedForwardOutputWeight,
                                                       &layer.feedForwardOutputBias,
                                                       width);

        for (auto index = 0u; index < hidden.size(); ++index)
            hidden[index] += feedForwardOutput[index];
    }

    return referenceLayerNorm(
        hidden, positions, width, model.finalNormWeight, model.finalNormBias);
}

// ---------------------------------------------------------------------------
// Running the GPU encoder, and saying how far off it was.
// ---------------------------------------------------------------------------

inline Vector<float> runEncoder(const EncoderShape& shape,
                                const SafeTensors& file,
                                const Vector<float>& mel)
{
    auto& device = eacp::GPU::Device::shared();

    const auto melBuffer = storageOf(mel);
    const auto output = outputFor(shape.elementCount());
    const auto weights = EncoderWeights {file, shape};

    auto encoder = Encoder {shape};
    encoder.prepare(device);

    auto commands = device.makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        encoder.encode(pass, melBuffer, weights, output);
    }

    commands.commit();

    return readBack(output, shape.elementCount());
}

// The same mixed absolute-and-relative measure isClose asserts on, reported as
// a number rather than only as a pass: a tolerance nobody has measured against
// is a tolerance that was guessed.
inline double worstError(const Vector<float>& actual,
                         const std::vector<double>& expected)
{
    auto worst = 0.0;

    for (auto index = 0; index < actual.size(); ++index)
    {
        const auto target = expected[(std::size_t) index];
        const auto error =
            std::abs((double) actual[index] - target) / (1.0 + std::abs(target));

        worst = std::max(worst, error);
    }

    return worst;
}

// Whisper's normalised mel occupies about [-1, 1] after the clamp and scale, so
// a synthetic one has to as well: an encoder fed numbers of the wrong size is
// an encoder whose saturating parts were never reached.
inline Vector<float> syntheticMel(const EncoderShape& shape)
{
    auto values = sized(shape.melElementCount());

    for (auto band = 0; band < shape.melBins; ++band)
    {
        for (auto frame = 0; frame < shape.inputFrames; ++frame)
        {
            const auto phase = 0.11 * band + 0.037 * frame;
            values[band * shape.inputFrames + frame] =
                (float) (std::sin(phase) + 0.5 * std::cos(2.3 * phase));
        }
    }

    return values;
}
} // namespace WSP::Testing
