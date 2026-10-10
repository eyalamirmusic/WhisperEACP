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

eacp::ML::Shape graphShape(const Shape& shape)
{
    auto extents = Vector<int> {};

    for (auto axis = 0; axis < shape.rank(); ++axis)
        extents.add(shape[axis]);

    return eacp::ML::Shape {extents};
}

// An additive mask's masked score, the eacp decoder step's own.
constexpr auto maskedScore = 10000.0f;
} // namespace

CoreMLNet::CoreMLNet(Span<const int> positionsToUse,
                     const CoreMLNetOptions& optionsToUse)
    : netOptions(optionsToUse)
{
    for (auto count: positionsToUse)
        positionSet.add(count);

    if (positionSet.empty())
        throw std::logic_error {"a Core ML encoder is compiled for at least one "
                                "audio context"};
}

CoreMLNet::CoreMLNet(const CoreMLNetOptions& optionsToUse)
    : netOptions(optionsToUse)
{
}

std::string CoreMLNet::cacheFeature(const Cache& cache)
{
    return "cache_" + std::to_string(cache.id);
}

std::string CoreMLNet::appendedRowsFeature(const Cache& cache)
{
    return cacheFeature(cache) + "_rows";
}

void CoreMLNet::requireFixedShape(std::string_view op) const
{
    if (!isFixedShape())
        throw std::logic_error {std::string {op}
                                + " is a decoder op, and the Core ML encoder's "
                                  "program has none"};
}

CoreMLNet::CacheInfo& CoreMLNet::cacheInfo(const Cache& cache)
{
    requireFixedShape("a cache");

    if (cache.id < 0 || cache.id >= caches.size())
        throw std::logic_error {"a Core ML net was handed a cache it did not "
                                "make"};

    return caches[cache.id];
}

eacp::ML::Tensor CoreMLNet::namedInput(const std::string& name,
                                       const eacp::ML::Shape& shape,
                                       eacp::ML::DType type)
{
    if (const auto* found = declaredInputs.getValue(name))
        return *found;

    const auto tensor = program.input(name, shape, type);
    declaredInputs.emplace(name, tensor);
    inputFeatures.add(name);

    return tensor;
}

void CoreMLNet::declareOutput(eacp::ML::Tensor value, const std::string& name)
{
    outputFeatures.add(name);
    program.output(value, name);
}

eacp::ML::Package CoreMLNet::build() const
{
    return program.build();
}

void CoreMLNet::retain(int) {}

void CoreMLNet::release(int) {}

Tensor CoreMLNet::make(eacp::ML::Tensor tensor, Layout layout, int maskedRows)
{
    recorded.add(Value {tensor, layout, maskedRows});
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

// The encoder's mel, band-major, as [1, bands, frames] with the frame axis
// enumerated over the contexts. The shape the body passes is the window's, and
// only its band count is read: the frames are the context's.
//
// A decoder's inputs are what the body says they are: fp16 rows, or int32
// token ids.
Tensor CoreMLNet::input(const Binding& source, const Shape& shape, DType type)
{
    requireUnbound(source);

    if (isFixedShape())
    {
        const auto name = featureName(source, "input", inputFeatures.size());

        if (type == DType::int32 && shape.rank() == 1
            && netOptions.embedsTokenByProduct)
            return make(namedInput(tokenDigitsFeature,
                                   {shape[0], 2},
                                   eacp::ML::DType::float16),
                        Layout::tokens);

        if (type == DType::int32 && shape.rank() == 1)
            return make(namedInput(name, graphShape(shape), eacp::ML::DType::int32),
                        Layout::tokens);

        if (type == DType::int32 || shape.rank() != 2)
            throw std::logic_error {"a Core ML decoder takes float rows or a list "
                                    "of token ids"};

        return make(namedInput(name, graphShape(shape), eacp::ML::DType::float16),
                    Layout::rows);
    }

    if (!inputFeatures.empty() || shape.rank() != 2 || type == DType::int32)
        throw std::logic_error {"the Core ML net takes one band-major float input, "
                                "[bands, frames]: the mel"};

    const auto bands = shape[0];
    const auto largest = largestContext();

    if (shape[1] < netOptions.framesPerPosition * largest)
        throw std::logic_error {"the Core ML net's input holds "
                                + std::to_string(shape[1])
                                + " frames, fewer than its largest context of "
                                + std::to_string(largest) + " positions reads"};

    const auto shapeFor = [&](int positions) -> eacp::ML::Shape
    {
        const auto frames = netOptions.framesPerPosition * positions;

        if (netOptions.framesMajorInput)
            return {1, frames, bands};

        return {1, bands, frames};
    };

    return make(
        contextInput(featureName(source, "input", inputFeatures.size()), shapeFor),
        Layout::bandMajorInput);
}

int CoreMLNet::largestContext() const
{
    return *std::max_element(positionSet.begin(), positionSet.end());
}

eacp::ML::Tensor
    CoreMLNet::contextInput(const std::string& name,
                            const std::function<eacp::ML::Shape(int)>& shapeFor)
{
    const auto largest = largestContext();
    inputFeatures.add(name);

    if (positionSet.size() == 1)
        return program.input(name, shapeFor(largest), eacp::ML::DType::float16);

    auto enumerated = Vector<eacp::ML::Shape> {};

    for (auto positions: positionSet)
        enumerated.add(shapeFor(positions));

    return program.input(
        name, shapeFor(largest), enumerated, eacp::ML::DType::float16);
}

void CoreMLNet::output(const Tensor& value, const Binding& target)
{
    requireUnbound(target);

    if (!isFixedShape() && !outputFeatures.empty())
        throw std::logic_error {"the Core ML encoder has one output: its rows"};

    const auto rowsOut = rowsOf(value);
    const auto name = featureName(target, "output", outputFeatures.size());

    if (netOptions.selection && name == netOptions.selection->logits)
        recordSelection(rowsOut);

    if (!netOptions.omittedOutputs.contains(name))
        declareOutput(rowsOut, name);
}

// The last row only: a step's logits are one row, and a prompt's last row is
// the distribution over what follows it.
void CoreMLNet::recordSelection(eacp::ML::Tensor logits)
{
    requireFixedShape("greedy selection");

    const auto rowCount = extentOf(logits, 2, 0);
    const auto vocabulary = extentOf(logits, 2, 1);

    const auto last =
        rowCount <= 1
            ? logits
            : program.slice(logits, {rowCount - 1, 0}, {rowCount, vocabulary});
    const auto suppression = namedInput(netOptions.selection->suppression,
                                        {1, vocabulary},
                                        eacp::ML::DType::float16);
    const auto scores = sum(last, suppression);

    declareOutput(netOptions.selection->onTheEngine ? argmaxDigits(scores)
                                                    : program.argmax(scores, -1),
                  netOptions.selection->token);
}

// [1, count] fp16: 0, 1, ... count - 1, every one exact while count <= 2049.
eacp::ML::Tensor CoreMLNet::counting(int count)
{
    const auto name = "counting_" + std::to_string(count);

    if (const auto* found = constants.getValue(name))
        return *found;

    auto values = Vector<float> {};

    for (auto index = 0; index < count; ++index)
        values.add((float) index);

    const auto tensor = program.halfConstant(name, {1, count}, values);
    constants.emplace(name, tensor);

    return tensor;
}

eacp::ML::Tensor CoreMLNet::oneHot(eacp::ML::Tensor index, int count)
{
    const auto isIndex =
        [](const eacp::GPU::Float& at, const eacp::GPU::Float& wanted)
    { return select(at == wanted, 1.0f, 0.0f); };

    return program.apply(counting(count), index, isIndex);
}

// [tokens, vocabulary]: per token, the outer product of its two digits'
// one-hot rows, laid out a radix to a row and cut to the vocabulary.
eacp::ML::Tensor CoreMLNet::oneHotTokens(eacp::ML::Tensor digits, int vocabulary)
{
    const auto tokens = extentOf(digits, 2, 0);
    const auto highCount = (vocabulary + tokenRadix - 1) / tokenRadix;
    const auto high = program.slice(digits, {0, 0}, {tokens, 1});
    const auto low = program.slice(digits, {0, 1}, {tokens, 2});

    const auto highColumns =
        program.reshape(oneHot(high, highCount), {tokens, highCount, 1});
    const auto lowRows =
        program.reshape(oneHot(low, tokenRadix), {tokens, 1, tokenRadix});
    const auto product = [](const eacp::GPU::Float& a, const eacp::GPU::Float& b)
    { return a * b; };
    const auto grid = program.apply(highColumns, lowRows, product);

    return program.slice(program.reshape(grid, {tokens, highCount * tokenRadix}),
                         {0, 0},
                         {tokens, vocabulary});
}

// The scores a radix to a row, padded with fp16's lowest; the lowest row whose
// maximum is the whole maximum, as highest - (highest - row) maximised; that
// row picked out by a one-hot product, exact since every other term is zero;
// and its lowest column holding the maximum, the same way.
eacp::ML::Tensor CoreMLNet::argmaxDigits(eacp::ML::Tensor scores)
{
    constexpr auto lowestHalf = -65504.0f;
    constexpr auto beyondAnyIndex = 2048.0f;

    const auto vocabulary = extentOf(scores, 2, 1);
    const auto rowCount = (vocabulary + tokenRadix - 1) / tokenRadix;
    const auto padding = rowCount * tokenRadix - vocabulary;

    auto grid = scores;

    if (padding > 0)
    {
        auto lowest = Vector<float> {};
        lowest.resize(padding, lowestHalf);

        grid = program.concat(
            {scores, program.halfConstant("argmax_padding", {1, padding}, lowest)},
            1);
    }

    grid = program.reshape(grid, {rowCount, tokenRadix});

    const auto rowMaxima = program.max(grid, 1, true);
    const auto best = program.max(rowMaxima, 0, true);

    const auto earliest = [](const eacp::GPU::Float& value,
                             const eacp::GPU::Float& target,
                             const eacp::GPU::Float& index)
    { return select(value == target, beyondAnyIndex - index, 0.0f); };

    const auto fromEarliest = [](const eacp::GPU::Float& score)
    { return beyondAnyIndex - score; };

    const auto rowIndices = program.reshape(counting(rowCount), {rowCount, 1});
    const auto row = program.apply(
        program.max(program.apply(rowMaxima, best, rowIndices, earliest), 0, true),
        fromEarliest);

    const auto pickedRow =
        program.matmul(program.reshape(oneHot(row, rowCount), {1, rowCount}), grid);

    const auto column = program.apply(
        program.max(
            program.apply(pickedRow, best, counting(tokenRadix), earliest), 1, true),
        fromEarliest);

    return program.concat({row, column}, 1);
}

// A lazy prefix: the table goes into the program whole and is cut to the rows
// of the stream it is added to, at run time, since the context decides how
// many that is and count is only the window's.
Tensor CoreMLNet::rows(const Weight& table, int first, int)
{
    if (isFixedShape())
        throw std::logic_error {"a Core ML decoder reads its positional table "
                                "through embed"};

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
// it frame-major is a relabelling here, not an op. A frames-major input is
// the one case that takes a transpose.
Tensor CoreMLNet::transpose(const Tensor& matrix)
{
    const auto& value = valueOf(matrix);

    if (value.layout != Layout::bandMajorInput)
        throw std::logic_error {"the Core ML net transposes only its mel input, "
                                "into the frames a convolution reads"};

    if (netOptions.framesMajorInput)
        return make(program.transpose(value.tensor, {0, 2, 1}),
                    Layout::channelsFirst);

    return make(value.tensor, Layout::channelsFirst);
}

Tensor CoreMLNet::cached(const Cache& cache)
{
    requireFixedShape("cached");

    const auto& info = cacheInfo(cache);

    if (info.filled)
        return make(*info.filled, Layout::wholeCache, info.capacity);

    const auto whole = namedInput(
        cacheFeature(cache), {info.capacity, info.width}, eacp::ML::DType::float16);

    return make(whole, Layout::wholeCache, info.capacity);
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
    const auto& keyValue = valueOf(keys);
    const auto isMasked = keyValue.layout == Layout::appendedCache
                          || keyValue.layout == Layout::wholeCache;

    if (causal != (keyValue.layout == Layout::appendedCache))
        throw std::logic_error {
            "the Core ML net attends causally only over a cache a step appended "
            "to, where the mask says which rows are earlier tokens, and "
            "without a mask only over rows or a whole cache"};

    if (valueOf(values).layout != keyValue.layout)
        throw std::logic_error {"the Core ML net attends over keys and values of "
                                "one kind"};

    const auto rowsIn = rowsOf(queries);
    const auto q = splitHeads(rowsIn, heads);
    const auto k = splitHeads(rowsOf(keys), heads);
    const auto v = splitHeads(rowsOf(values), heads);

    if (!isMasked)
        return make(
            mergeHeads(netOptions.fusedAttention
                           ? program.scaledDotProductAttention(q, k, v, false)
                           : unfusedAttention(q, k, v)),
            Layout::rows);

    const auto allowed = allowedKeys(keyValue, rowsIn);
    const auto attended = netOptions.fusedAttention
                              ? program.scaledDotProductAttention(q, k, v, allowed)
                              : maskedAttention(q, k, v, allowed);

    return make(mergeHeads(attended), Layout::rows);
}

// Which keys a query may attend to, 1 or 0: over a whole cache the cross
// mask, [1, keys]; over the rows a sequence's opening appended to empty caches
// each row's own and earlier ones, [rows, rows]; over a cache a step appended
// to the self mask's capacity rows and the step's own rows after them, always
// allowed, [1, keys]. A step past the opening attends from one token.
eacp::ML::Tensor CoreMLNet::allowedKeys(const Value& keys, eacp::ML::Tensor queries)
{
    if (keys.layout == Layout::wholeCache)
        return namedInput(
            crossMaskFeature, {1, keys.maskedRows}, eacp::ML::DType::float16);

    if (keys.maskedRows == 0)
        return causalMask(extentOf(queries, 2, 0));

    if (program.isValid() && extentOf(queries, 2, 0) != 1)
        throw std::logic_error {"a Core ML decoder step attends from one token; "
                                "only the program that opens a sequence takes "
                                "several"};

    const auto appended = extentOf(keys.tensor, 2, 0) - keys.maskedRows;
    const auto name =
        std::string {selfMaskFeature} + "_" + std::to_string(keys.maskedRows);

    if (const auto* found = masks.getValue(name))
        return *found;

    auto ownRows = Vector<float> {};
    ownRows.resize(std::max(appended, 1), 1.0f);

    const auto stored =
        namedInput(selfMaskFeature, {1, keys.maskedRows}, eacp::ML::DType::float16);
    const auto own = program.halfConstant(name + "_own", {1, appended}, ownRows);
    const auto allowed = program.concat({stored, own}, 1);

    masks.emplace(name, allowed);

    return allowed;
}

// [rows, rows], 1 on and below the diagonal: each of a prompt's rows attends
// to itself and the rows before it.
eacp::ML::Tensor CoreMLNet::causalMask(int rows)
{
    const auto name = "causal_mask_" + std::to_string(rows);

    if (const auto* found = masks.getValue(name))
        return *found;

    auto allowed = Vector<float> {};

    for (auto row = 0; row < rows; ++row)
        for (auto column = 0; column < rows; ++column)
            allowed.add(column <= row ? 1.0f : 0.0f);

    const auto mask = program.halfConstant(name, {rows, rows}, allowed);
    masks.emplace(name, mask);

    return mask;
}

eacp::ML::Tensor CoreMLNet::positionRows(const Weight& positionTable,
                                         int firstPosition,
                                         int count)
{
    return program.slice(weightOf(positionTable),
                         {firstPosition, 0},
                         {firstPosition + count, positionTable.dimension(1)});
}

// The tokens looked up and their positions added. A step embeds one token at
// the position input, since one program serves every step of a sequence and
// firstPosition, the one the body records with, is the run time's to give; the
// program that opens a sequence embeds the whole prompt at the positions it
// records with.
Tensor CoreMLNet::embed(const Tensor& tokens,
                        const Weight& tokenTable,
                        const Weight& positionTable,
                        int firstPosition)
{
    requireFixedShape("embed");

    const auto& value = valueOf(tokens);

    if (value.layout != Layout::tokens)
        throw std::logic_error {"the Core ML net embeds only token ids"};

    const auto tokenCount = program.isValid() ? program.shape(value.tensor)[0] : 1;

    if (!netOptions.opensSequence && tokenCount != 1)
        throw std::logic_error {"a Core ML decoder step embeds one token; only "
                                "the program that opens a sequence takes "
                                "several"};

    const auto token =
        netOptions.embedsTokenByProduct
            ? program.matmul(oneHotTokens(value.tensor, tokenTable.dimension(0)),
                             weightOf(tokenTable))
            : program.gather(weightOf(tokenTable), value.tensor, 0);

    const auto offset = [&]
    {
        if (netOptions.opensSequence)
            return positionRows(positionTable, firstPosition, tokenCount);

        if (!netOptions.embedsPositionByProduct)
            return program.gather(
                weightOf(positionTable),
                namedInput(positionFeature, {1}, eacp::ML::DType::int32),
                0);

        const auto position =
            namedInput(positionFeature, {1, 1}, eacp::ML::DType::float16);

        return program.matmul(oneHot(position, positionTable.dimension(0)),
                              weightOf(positionTable));
    }();

    return make(sum(token, offset), Layout::rows);
}

// A projection that fills the cache writes it out, and keeps it for what
// reads the cache later in the same program; a shorter one appends to the
// cache read in, or to an empty one in the program that opens a sequence, and
// writes out only the rows it made.
Tensor CoreMLNet::appendLinear(Cache& cache,
                               const Tensor& input,
                               const Weight& weight,
                               const Weight* bias)
{
    requireFixedShape("appendLinear");

    auto& info = cacheInfo(cache);
    const auto projected =
        valueOf(linear(input, weight, bias, Activation::none)).tensor;
    const auto count = extentOf(projected, 2, 0);

    if (count == info.capacity)
    {
        declareOutput(projected, cacheFeature(cache));
        info.filled = projected;
        cache.rows = count;

        return make(projected, Layout::rows);
    }

    if (program.isValid() && (count <= 0 || count > info.capacity))
        throw std::logic_error {"a cache of " + std::to_string(info.capacity)
                                + " rows was appended " + std::to_string(count)};

    if (netOptions.opensSequence)
    {
        declareOutput(projected, appendedRowsFeature(cache));
        cache.rows = count;

        return make(projected, Layout::appendedCache, 0);
    }

    const auto stored = namedInput(
        cacheFeature(cache), {info.capacity, info.width}, eacp::ML::DType::float16);
    declareOutput(projected, appendedRowsFeature(cache));
    cache.rows += count;

    return make(program.concat({stored, projected}, 0),
                Layout::appendedCache,
                info.capacity);
}

Cache CoreMLNet::makeCache(int capacityRows, int width)
{
    requireFixedShape("makeCache");

    if (capacityRows <= 0 || width <= 0)
        throw std::logic_error {"a cache needs rows and a width"};

    caches.add(CacheInfo {capacityRows, width, std::nullopt});

    return Cache {caches.size() - 1, 0};
}

// A channels-first value is the net's [frames, channels] rows seen the other
// way round, so it is turned back before anything but a convolution reads it.
eacp::ML::Tensor CoreMLNet::rowsOf(const Tensor& handle)
{
    const auto& value = valueOf(handle);

    switch (value.layout)
    {
        case Layout::rows:
        case Layout::appendedCache:
        case Layout::wholeCache:
            return value.tensor;

        case Layout::tokens:
            throw std::logic_error {"the Core ML net reads token ids only through "
                                    "embed"};

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

// The same with an additive mask in front of the softmax, for an OS that loads
// specification 8 but not 9's masked attention.
eacp::ML::Tensor CoreMLNet::maskedAttention(eacp::ML::Tensor queries,
                                            eacp::ML::Tensor keys,
                                            eacp::ML::Tensor values,
                                            eacp::ML::Tensor allowed)
{
    const auto headWidth = extentOf(queries, 3, 2);
    const auto scale = 1.0f / std::sqrt((float) headWidth);

    const auto scaledAndMasked =
        [scale](const eacp::GPU::Float& score, const eacp::GPU::Float& keep)
    { return score * scale + (keep - 1.0f) * maskedScore; };

    const auto scores = program.matmul(queries, keys, false, true);
    const auto weights =
        program.softmax(program.apply(scores, allowed, scaledAndMasked), -1);

    return program.matmul(weights, values);
}
} // namespace WSP
