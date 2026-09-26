#include "CoreMLNet.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace WSP
{
namespace
{
eacp::ML::Shape graphShape(const Vector<int>& extents)
{
    return eacp::ML::Shape {extents};
}

std::logic_error notInTheEncoder(std::string_view op)
{
    return std::logic_error {std::string {op}
                             + " is not part of the Core ML encoder; the "
                               "decoder's ops arrive with phase 4"};
}
} // namespace

CoreMLNet::CoreMLNet(Span<const int> positionsToUse,
                     const CoreMLNetOptions& optionsToUse)
    : netOptions(optionsToUse)
{
    for (auto count: positionsToUse)
        positionSet.add(count);

    if (positionSet.empty())
        throw std::logic_error {"a Core ML net is compiled for at least one "
                                "audio context"};
}

eacp::ML::Package CoreMLNet::build() const
{
    return program.build();
}

void CoreMLNet::retain(int) {}

void CoreMLNet::release(int) {}

Tensor CoreMLNet::make(eacp::ML::Tensor tensor, Layout layout)
{
    recorded.add(Value {tensor, layout});
    return Tensor {*this, recorded.size() - 1};
}

const CoreMLNet::Value& CoreMLNet::valueOf(const Tensor& handle) const
{
    if (!handle.isOwnedBy(*this) || handle.index() < 0
        || handle.index() >= recorded.size())
        throw std::logic_error {"a Core ML net was handed a tensor it did not "
                                "record"};

    return recorded[handle.index()];
}

std::string CoreMLNet::featureName(const Binding& binding,
                                   std::string_view fallback,
                                   int index)
{
    if (!binding.name.empty())
        return binding.name;

    return std::string {fallback} + "_" + std::to_string(index);
}

void CoreMLNet::requireUnbound(const Binding& binding)
{
    if (binding.range.isValid())
        throw std::logic_error {"a Core ML net addresses its inputs and outputs "
                                "by name; the binding's buffer would never be "
                                "read or written"};
}

// The mel, band-major, as [1, bands, frames] with the frame axis enumerated
// over the contexts. The shape the body passes is the window's, and only its
// band count is read: the frames are the context's.
Tensor CoreMLNet::input(const Binding& source, const Shape& shape, DType type)
{
    requireUnbound(source);

    if (!inputFeatures.empty() || shape.rank() != 2 || type == DType::int32)
        throw std::logic_error {"the Core ML net takes one band-major float input, "
                                "[bands, frames]: the mel"};

    const auto bands = shape[0];
    const auto largest = *std::max_element(positionSet.begin(), positionSet.end());

    if (shape[1] < netOptions.framesPerPosition * largest)
        throw std::logic_error {"the Core ML net's input holds "
                                + std::to_string(shape[1])
                                + " frames, fewer than its largest context of "
                                + std::to_string(largest) + " positions reads"};
    const auto shapeFor = [&](int positions)
    { return eacp::ML::Shape {1, bands, netOptions.framesPerPosition * positions}; };

    const auto name = featureName(source, "input", inputFeatures.size());
    inputFeatures.add(name);

    if (positionSet.size() == 1)
        return make(program.input(name, shapeFor(largest), eacp::ML::DType::float16),
                    Layout::bandMajorInput);

    auto enumerated = Vector<eacp::ML::Shape> {};

    for (auto positions: positionSet)
        enumerated.add(shapeFor(positions));

    return make(
        program.input(name, shapeFor(largest), enumerated, eacp::ML::DType::float16),
        Layout::bandMajorInput);
}

void CoreMLNet::output(const Tensor& value, const Binding& target)
{
    requireUnbound(target);

    if (!outputFeatures.empty())
        throw std::logic_error {"the Core ML net has one output: the encoder's "
                                "rows"};

    const auto rowsOut = rowsOf(value);
    const auto name = featureName(target, "output", outputFeatures.size());
    outputFeatures.add(name);

    program.output(rowsOut, name);
}

// A lazy prefix: the table goes into the program whole and is cut to the rows
// of the stream it is added to, at run time, since the context decides how
// many that is and count is only the window's.
Tensor CoreMLNet::rows(const Weight& table, int first, int)
{
    if (first != 0)
        throw std::logic_error {"the Core ML net reads a positional table from its "
                                "first row"};

    const auto largest = *std::max_element(positionSet.begin(), positionSet.end());

    if (table.dimension(0) < largest)
        throw ModelError {"the positional table '" + table.name + "' carries "
                          + std::to_string(table.dimension(0))
                          + " rows, and the Core ML encoder is compiled for "
                            "contexts of up to "
                          + std::to_string(largest)};

    return make(weightOf(table), Layout::positionalPrefix);
}

// conv wants [1, bands, frames], which the band-major mel already is: turning
// it frame-major is a relabelling here, not an op.
Tensor CoreMLNet::transpose(const Tensor& matrix)
{
    const auto& value = valueOf(matrix);

    if (value.layout != Layout::bandMajorInput)
        throw std::logic_error {"the Core ML net transposes only its band-major "
                                "input, into the frames a convolution reads"};

    return make(value.tensor, Layout::channelsFirst);
}

Tensor CoreMLNet::cached(const Cache&)
{
    throw notInTheEncoder("cached");
}

Tensor CoreMLNet::conv1d(const Tensor& frames,
                         const Weight& weight,
                         const Weight& bias,
                         int stride,
                         int padding,
                         Activation activation)
{
    const auto& value = valueOf(frames);

    if (value.layout != Layout::channelsFirst)
        throw std::logic_error {"the Core ML net convolves only the transposed "
                                "input and what a convolution made"};

    const auto convolved = program.conv(
        value.tensor, weightOf(weight), weightOf(bias), stride, padding);

    return make(activated(convolved, activation), Layout::channelsFirst);
}

Tensor CoreMLNet::linear(const Tensor& input,
                         const Weight& weight,
                         const Weight* bias,
                         Activation activation)
{
    const auto x = rowsOf(input);
    const auto projected =
        bias == nullptr ? program.linear(x, weightOf(weight))
                        : program.linear(x, weightOf(weight), weightOf(*bias));

    return make(activated(projected, activation), Layout::rows);
}

Tensor CoreMLNet::linearAdd(const Tensor& input,
                            const Weight& weight,
                            const Weight* bias,
                            const Tensor& stream)
{
    const auto projected = linear(input, weight, bias, Activation::none);
    return make(sum(rowsOf(stream), valueOf(projected).tensor), Layout::rows);
}

Tensor CoreMLNet::add(const Tensor& stream, const Tensor& addend)
{
    const auto& streamValue = valueOf(stream);
    const auto& addendValue = valueOf(addend);

    const auto streamIsTable = streamValue.layout == Layout::positionalPrefix;
    const auto addendIsTable = addendValue.layout == Layout::positionalPrefix;

    if (streamIsTable && addendIsTable)
        throw std::logic_error {"the Core ML net adds a positional table to a "
                                "stream, not to another table"};

    if (!streamIsTable && !addendIsTable)
        return make(sum(rowsOf(stream), rowsOf(addend)), Layout::rows);

    const auto rowsToAdd = rowsOf(streamIsTable ? addend : stream);
    const auto table = streamIsTable ? streamValue.tensor : addendValue.tensor;

    return make(sum(rowsToAdd, program.sliceLike(table, rowsToAdd)), Layout::rows);
}

Tensor CoreMLNet::layerNorm(const Tensor& input,
                            const Weight& weight,
                            const Weight& bias)
{
    return make(
        program.layerNorm(rowsOf(input), {-1}, weightOf(weight), weightOf(bias)),
        Layout::rows);
}

Tensor CoreMLNet::attention(const Tensor& queries,
                            const Tensor& keys,
                            const Tensor& values,
                            int heads,
                            bool causal)
{
    if (causal)
        throw notInTheEncoder("causal attention");

    const auto q = splitHeads(rowsOf(queries), heads);
    const auto k = splitHeads(rowsOf(keys), heads);
    const auto v = splitHeads(rowsOf(values), heads);

    const auto attended = netOptions.fusedAttention
                              ? program.scaledDotProductAttention(q, k, v, false)
                              : unfusedAttention(q, k, v);

    return make(mergeHeads(attended), Layout::rows);
}

Tensor CoreMLNet::embed(const Tensor&, const Weight&, const Weight&, int)
{
    throw notInTheEncoder("embed");
}

Tensor CoreMLNet::appendLinear(Cache&, const Tensor&, const Weight&, const Weight*)
{
    throw notInTheEncoder("appendLinear");
}

Cache CoreMLNet::makeCache(int, int)
{
    throw notInTheEncoder("makeCache");
}

// A channels-first value is the net's [frames, channels] rows seen the other
// way round, so it is turned back before anything but a convolution reads it.
eacp::ML::Tensor CoreMLNet::rowsOf(const Tensor& handle)
{
    const auto& value = valueOf(handle);

    switch (value.layout)
    {
        case Layout::rows:
            return value.tensor;

        case Layout::channelsFirst:
        {
            const auto channels = extentOf(value.tensor, 3, 1);
            const auto framesFirst = program.transpose(value.tensor, {0, 2, 1});
            return program.reshape(framesFirst,
                                   {eacp::ML::Shape::unknown, channels});
        }

        case Layout::bandMajorInput:
            throw std::logic_error {"the Core ML net reads its band-major input "
                                    "only through transpose"};

        case Layout::positionalPrefix:
            break;
    }

    throw std::logic_error {"the Core ML net reads a positional table only as "
                            "what add() adds to a stream"};
}

int CoreMLNet::extentOf(eacp::ML::Tensor value, int rank, int axis) const
{
    const auto& shape = program.shape(value);

    // A graph that has already failed reports its first error from
    // CoreMLEncoder::prepare as a ModelError; only a valid one misused is ours.
    if (!program.isValid())
        return 0;

    if (shape.rank() != rank)
        throw std::logic_error {"the Core ML net expected a value of rank "
                                + std::to_string(rank) + " and recorded one of "
                                + std::to_string(shape.rank())};

    return shape[axis];
}

// Every constant in the blob as fp16, which is what the engine runs: an F16
// file's bytes as they lie, and anything else narrowed from its values. For
// tiny.en, whose F32 values are all exact halves, the narrowing loses nothing.
eacp::ML::Tensor CoreMLNet::weightOf(const Weight& weight)
{
    if (!weight.name.empty())
        if (const auto* found = constants.getValue(weight.name))
            return *found;

    // A device tensor's fileBytes view the SafeTensors it came from, which
    // may be gone by now; a host tensor is only ever read while it lives.
    if (!weight.isHostOnly() || weight.fileBytes.empty())
        throw std::logic_error {"the Core ML net writes a host weight's bytes in "
                                "the file into its blob, and this weight is on "
                                "the device or has none; prepare the encoder "
                                "from host weights a SafeTensors made"};

    const auto name = weight.name.empty()
                          ? "weight_" + std::to_string(unnamedConstants++)
                          : weight.name;
    const auto shape = graphShape(weight.shape);

    const auto narrowed = [&]
    {
        const auto floats = readFileFloats(weight);
        return program.halfConstant(name, shape, floats);
    };

    const auto constant =
        weight.fileType == TensorType::F16
            ? program.constant(
                  name, shape, eacp::ML::DType::float16, weight.fileBytes)
            : narrowed();

    if (!weight.name.empty())
        constants.emplace(weight.name, constant);

    return constant;
}

eacp::ML::Tensor CoreMLNet::sum(eacp::ML::Tensor stream, eacp::ML::Tensor addend)
{
    const auto plus = [](const eacp::GPU::Float& a, const eacp::GPU::Float& b)
    { return a + b; };

    return program.apply(stream, addend, plus);
}

eacp::ML::Tensor CoreMLNet::activated(eacp::ML::Tensor value, Activation activation)
{
    return activation == Activation::gelu ? program.gelu(value) : value;
}

// [positions, width] rows as [heads, positions, headWidth], which is what the
// attention ops take: the graph's attention does not split heads itself.
eacp::ML::Tensor CoreMLNet::splitHeads(eacp::ML::Tensor rowsToSplit, int heads)
{
    const auto width = extentOf(rowsToSplit, 2, 1);

    if (heads <= 0 || width % heads != 0)
        throw std::logic_error {"attention was asked for a head count that does "
                                "not divide the width"};

    const auto perHead = program.reshape(
        rowsToSplit, {eacp::ML::Shape::unknown, heads, width / heads});

    return program.transpose(perHead, {1, 0, 2});
}

eacp::ML::Tensor CoreMLNet::mergeHeads(eacp::ML::Tensor perHead)
{
    const auto width = extentOf(perHead, 3, 0) * extentOf(perHead, 3, 2);
    const auto byRow = program.transpose(perHead, {1, 0, 2});

    return program.reshape(byRow, {eacp::ML::Shape::unknown, width});
}

// For an OS that loads specification 8 but not 9. The scale is the kernels'
// and the fused op's, 1/sqrt(headWidth), applied to the scores.
eacp::ML::Tensor CoreMLNet::unfusedAttention(eacp::ML::Tensor queries,
                                             eacp::ML::Tensor keys,
                                             eacp::ML::Tensor values)
{
    const auto headWidth = extentOf(queries, 3, 2);
    const auto scale = 1.0f / std::sqrt((float) headWidth);

    const auto scaled = [scale](const eacp::GPU::Float& score)
    { return score * scale; };

    const auto scores = program.matmul(queries, keys, false, true);
    const auto weights = program.softmax(program.apply(scores, scaled), -1);

    return program.matmul(weights, values);
}
} // namespace WSP
