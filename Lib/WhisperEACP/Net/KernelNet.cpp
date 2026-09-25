#include "KernelNet.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace WSP
{
using eacp::GPU::Buffer;
using eacp::GPU::BufferRange;
using eacp::GPU::BufferUsage;
using eacp::GPU::ComputePass;
using eacp::GPU::Device;

namespace
{
constexpr int floatBytes(int elementCount)
{
    return elementCount * (int) sizeof(float);
}

Buffer allocate(Device& device, int elementCount)
{
    return device.makeBuffer(floatBytes(elementCount), BufferUsage::Storage);
}

Buffer allocateZeroed(Device& device, int elementCount)
{
    auto zeroes = Vector<float> {};
    zeroes.resize(elementCount);

    for (auto index = 0; index < elementCount; ++index)
        zeroes[index] = 0.f;

    return device.makeBuffer(
        zeroes.data(), floatBytes(elementCount), BufferUsage::Storage);
}

// The tiled projection programs differ only in how they read the weight and in
// which tiling they compute the product with, and every one of them takes a
// TiledMatMulShape, so the bind is written once.
template <typename Program>
void dispatchLinear(Program& program,
                    ComputePass& pass,
                    const BufferRange& input,
                    const Buffer& weight,
                    const Buffer& bias,
                    const BufferRange& target,
                    int innerCount,
                    int outputWidth,
                    int rowCount,
                    bool gelu,
                    bool residual)
{
    program.a = input;
    program.b = weight;
    program.bias = bias;
    program.output = target;

    auto shape = TiledMatMulShape::forLinear(rowCount, innerCount, outputWidth);
    shape.gelu = gelu;
    shape.residual = residual;

    program.dispatch(pass, shape);
}

template <WeightStorage weightStorage>
void dispatchLinear(SplitLinearProgram<weightStorage>& program,
                    ComputePass& pass,
                    const BufferRange& input,
                    const Buffer& weight,
                    const Buffer& bias,
                    const BufferRange& target,
                    int innerCount,
                    int outputWidth,
                    int rowCount,
                    bool gelu,
                    bool residual)
{
    program.input = input;
    program.weights = weight;
    program.bias = bias;
    program.output = target;
    program.innerCount = (std::uint32_t) innerCount;
    program.outputWidth = (std::uint32_t) outputWidth;
    program.rowCount = (std::uint32_t) rowCount;
    program.gelu = gelu ? 1u : 0u;
    program.residual = residual ? 1u : 0u;

    program.dispatch(pass, outputWidth, rowCount);
}

constexpr int scoreColumnTiles(int keyCount)
{
    return (keyCount + AttentionScoresProduct::tile - 1)
           / AttentionScoresProduct::tile * AttentionScoresProduct::maximaPerRow;
}

float attentionScale(int headWidth)
{
    return 1.f / std::sqrt((float) headWidth);
}

int layerNormLanes(KernelProfile profile)
{
    return profile == KernelProfile::encoder ? LayerNorm::manyRowLanes
                                             : LayerNorm::singleRowLanes;
}

template <int maxSize>
void trackOnce(EA::StaticVector<const Buffer*, maxSize>& keys, const Buffer* key)
{
    if (keys.contains(key))
        return;

    if (keys.size() == maxSize)
        throw std::logic_error {"a kernel net tracks more buffers between two "
                                "barriers than it has room for"};

    keys.add(key);
}
} // namespace

KernelNet::KernelNet(KernelProfile profileToUse)
    : profile(profileToUse)
    , normalisation {layerNormLanes(profileToUse)}
{
    records.resize(maxRecords);
}

void KernelNet::prepare(Device& device, const KernelNetLayout& layoutToUse)
{
    gpu = &device;
    layout = layoutToUse;

    normalisation.prepare(device);
    projection.prepare(device);
    packedProjection.prepare(device);

    auto maxKeyCapacity = 0;

    for (auto& reservation: layout.attention)
        maxKeyCapacity = std::max(maxKeyCapacity, reservation.keyCapacity);

    if (profile == KernelProfile::encoder)
    {
        unfold.prepare(device);
        sum.prepare(device);
        windowScores.prepare(device);
        windowApply.prepare(device);
    }
    else
    {
        embedding.prepare(device);
        splitProjection.prepare(device);
        packedSplitProjection.prepare(device);
        splitLogits.prepare(device);
        packedSplitLogits.prepare(device);
        cachedScores.prepare(device);
        softmax.prepare(device);
        cachedApply.prepare(device);
        singleQueryAttention.prepare(
            device, layout.heads, layout.headWidth, maxKeyCapacity);
    }

    slots.clear();

    for (auto& reservation: layout.scratch)
        for (auto index = 0; index < reservation.count; ++index)
            slots.add(ScratchSlot {allocate(device, reservation.elements),
                                   reservation.elements});

    columns.reset();

    if (layout.columnElements > 0)
        columns.emplace(allocate(device, layout.columnElements));

    scoreScratch.clear();

    for (auto& reservation: layout.attention)
    {
        const auto alreadyReserved = [&](const ScoreScratch& scratch)
        { return scratch.keyCapacity == reservation.keyCapacity; };

        if (scoreScratch.findIf(alreadyReserved) != nullptr)
            continue;

        const auto scoreRows = layout.heads * reservation.queryCapacity;

        auto& scratch = scoreScratch.add(
            ScoreScratch {reservation.keyCapacity,
                          allocate(device, scoreRows * reservation.keyCapacity)});

        if (profile == KernelProfile::encoder)
            scratch.tileMaxima.emplace(allocate(
                device, scoreRows * scoreColumnTiles(reservation.keyCapacity)));
    }

    zeroBiases.clear();

    for (auto length: layout.zeroBiasLengths)
        zeroBiases.add(ZeroBias {allocateZeroed(device, length), length});

    caches.clear();
}

Cache KernelNet::makeCache(int capacityRows, int width)
{
    if (gpu == nullptr)
        throw std::logic_error {"a kernel net makes a cache once it is prepared"};

    caches.add(
        CacheStorage {allocate(*gpu, capacityRows * width), capacityRows, width});

    return {caches.size() - 1, 0};
}

void KernelNet::begin(ComputePass& passToUse)
{
    recordingPass = &passToUse;
    pending = {};
    readSinceBarrier.clear();
    writtenSinceBarrier.clear();
}

void KernelNet::end()
{
    flush();
    recordingPass = nullptr;

    for (auto& slot: slots)
        if (slot.holder >= 0)
            throw std::logic_error {"a value outlived the recording that made it"};
}

ComputePass& KernelNet::activePass()
{
    if (recordingPass == nullptr)
        throw std::logic_error {"a kernel net records between begin() and end()"};

    return *recordingPass;
}

void KernelNet::retain(int index)
{
    ++records[index].references;
}

void KernelNet::release(int index)
{
    auto& record = records[index];

    if (--record.references > 0)
        return;

    record.live = false;

    if (record.slot >= 0)
    {
        slots[record.slot].holder = -1;
        record.slot = -1;
    }

    if (record.base >= 0)
    {
        const auto base = record.base;
        record.base = -1;
        release(base);
    }
}

int KernelNet::makeRecord()
{
    flush();

    for (auto index = 0; index < records.size(); ++index)
        if (!records[index].live)
        {
            records[index] = Record {};
            records[index].live = true;
            return index;
        }

    throw std::logic_error {"a kernel net recording holds more values at once "
                            "than it has records for"};
}

KernelNet::Record& KernelNet::recordOf(const Tensor& value)
{
    if (!value.isValid())
        throw std::logic_error {"an op was given a tensor that holds no value"};

    return records[value.index()];
}

Tensor KernelNet::view(const Record& source)
{
    const auto index = makeRecord();

    records[index] = source;
    records[index].live = true;
    records[index].references = 0;

    return {*this, index};
}

// The lowest-numbered free buffer of exactly the capacity asked for, so a
// recording lands its values where the recording before it did. A capacity
// nothing was reserved at gets no buffer at all, which is only right for a
// value the next call hands to output().
int KernelNet::takeSlot(int elements)
{
    flush();

    auto reserved = false;

    for (auto index = 0; index < slots.size(); ++index)
    {
        if (slots[index].elements != elements)
            continue;

        reserved = true;

        if (slots[index].holder < 0)
            return index;
    }

    if (reserved)
        throw std::logic_error {"every scratch buffer of " + std::to_string(elements)
                                + " elements is holding a value"};

    return -1;
}

Tensor KernelNet::fresh(const Shape& shape, const Shape& capacity)
{
    const auto slot = takeSlot(capacity.elementCount());

    auto record = Record {};
    record.shape = shape;
    record.capacity = capacity;
    record.rowStride = shape.columns();
    record.slot = slot;

    if (slot >= 0)
        record.storage = BufferRange::of(slots[slot].buffer);

    auto value = view(record);

    if (slot >= 0)
        slots[slot].holder = value.index();

    return value;
}

bool KernelNet::makesNewValue(OpKind kind)
{
    return kind == OpKind::convolution || kind == OpKind::linear
           || kind == OpKind::layerNorm || kind == OpKind::attention
           || kind == OpKind::embed;
}

void KernelNet::flush()
{
    if (pending.kind == OpKind::none)
        return;

    const auto op = pending;
    pending = {};

    if (makesNewValue(op.kind) && records[op.result].storage.buffer == nullptr)
        throw std::logic_error {"a value with no scratch buffer of its capacity "
                                "has to be placed by output()"};

    run(op);
}

void KernelNet::run(const PendingOp& op)
{
    switch (op.kind)
    {
        case OpKind::convolution:
            runConvolution(op);
            break;
        case OpKind::linear:
        case OpKind::linearAdd:
        case OpKind::append:
            runProjection(op);
            break;
        case OpKind::add:
            runAdd(op);
            break;
        case OpKind::layerNorm:
            runLayerNorm(op);
            break;
        case OpKind::attention:
            runAttention(op);
            break;
        case OpKind::embed:
            runEmbed(op);
            break;
        case OpKind::none:
            break;
    }
}

void KernelNet::order(std::initializer_list<BufferKey> reads,
                      std::initializer_list<BufferKey> writes)
{
    auto hazard = false;

    for (auto read: reads)
        hazard = hazard || writtenSinceBarrier.contains(read);

    for (auto write: writes)
        hazard = hazard || writtenSinceBarrier.contains(write)
                 || readSinceBarrier.contains(write);

    if (hazard)
    {
        activePass().barrier();
        readSinceBarrier.clear();
        writtenSinceBarrier.clear();
    }

    for (auto read: reads)
        trackOnce(readSinceBarrier, read);

    for (auto write: writes)
        trackOnce(writtenSinceBarrier, write);
}

bool KernelNet::splitsProjection(int rowCount) const
{
    return profile == KernelProfile::decoder && rowCount <= splitRowLimit;
}

// Neither backend defines what a shader reading an unbound buffer gets, and a
// flag would only guard a read that must not happen at all, so a projection
// with no bias binds zeroes as long as its outputs: the shortest reserved
// buffer that is long enough.
const Buffer& KernelNet::zeroBiasFor(int length) const
{
    const ZeroBias* chosen = nullptr;

    for (auto& candidate: zeroBiases)
        if (candidate.length >= length
            && (chosen == nullptr || candidate.length < chosen->length))
            chosen = &candidate;

    if (chosen == nullptr)
        throw std::logic_error {"no zero bias was reserved for "
                                + std::to_string(length) + " outputs"};

    return chosen->buffer;
}

const KernelNet::ScoreScratch& KernelNet::scoresFor(int keyCapacity) const
{
    const auto reservedFor = [&](const ScoreScratch& scratch)
    { return scratch.keyCapacity == keyCapacity; };

    const auto* found = scoreScratch.findIf(reservedFor);

    if (found == nullptr)
        throw std::logic_error {"no attention scores were reserved for "
                                + std::to_string(keyCapacity) + " keys"};

    return *found;
}

Tensor KernelNet::input(const Binding& source, const Shape& shape, DType)
{
    flush();

    auto record = Record {};
    record.storage = source.range;
    record.shape = shape;
    record.capacity = source.capacity.rank() == 0 ? shape : source.capacity;
    record.rowStride = record.capacity.columns();

    return view(record);
}

void KernelNet::output(const Tensor& value, const Binding& target)
{
    auto& record = recordOf(value);

    const auto madeByPending =
        pending.result == value.index() && makesNewValue(pending.kind);

    if (madeByPending)
    {
        if (record.slot >= 0)
            slots[record.slot].holder = -1;

        record.slot = -1;
        record.storage = target.range;
        flush();
        return;
    }

    const auto alreadyThere = record.storage.buffer == target.range.buffer
                              && record.storage.offset == target.range.offset;

    if (!alreadyThere)
        throw std::logic_error {"the kernel backend places an output by pointing "
                                "the op just before it at the target"};

    flush();
}

Tensor KernelNet::rows(const Weight& table, int first, int count)
{
    flush();

    const auto width = table.dimension(1);
    const auto offset = floatBytes(first * width);

    auto record = Record {};
    record.storage = {&table.buffer, offset, table.buffer.size() - offset};
    record.shape = {count, width};
    record.capacity = record.shape;
    record.rowStride = width;

    return view(record);
}

Tensor KernelNet::transpose(const Tensor& matrix)
{
    flush();

    const auto& source = recordOf(matrix);

    auto record = Record {};
    record.storage = source.storage;
    record.shape = {source.shape[1], source.shape[0]};
    record.capacity = {source.capacity[1], source.capacity[0]};
    record.rowStride = source.columnStride;
    record.columnStride = source.rowStride;
    record.base = matrix.index();

    retain(matrix.index());

    return view(record);
}

Tensor KernelNet::cached(const Cache& cache)
{
    flush();

    const auto& storage = caches[cache.id];
    const auto width = storage.width;

    auto record = Record {};
    record.storage = BufferRange::of(storage.buffer);
    record.shape = {cache.rows, width};
    record.capacity = {storage.capacity, width};
    record.rowStride = width;

    return view(record);
}

Tensor KernelNet::conv1d(const Tensor& frames,
                         const Weight& weight,
                         const Weight& bias,
                         int stride,
                         int padding,
                         Activation activation)
{
    flush();

    const auto& source = recordOf(frames);
    const auto outputs = weight.dimension(0);
    const auto taps = weight.dimension(2);

    const auto length = conv1dOutputLength(source.shape[0], taps, stride, padding);
    const auto capacity =
        conv1dOutputLength(source.capacity[0], taps, stride, padding);

    auto result = fresh({length, outputs}, {capacity, outputs});

    auto op = PendingOp {};
    op.kind = OpKind::convolution;
    op.result = result.index();
    op.first = frames.index();
    op.weight = &weight;
    op.bias = &bias;
    op.stride = stride;
    op.padding = padding;
    op.activation = activation;
    pending = op;

    return result;
}

Tensor KernelNet::linear(const Tensor& input,
                         const Weight& weight,
                         const Weight* bias,
                         Activation activation)
{
    flush();

    const auto& source = recordOf(input);
    const auto outputs = weight.dimension(0);

    auto result =
        fresh({source.shape.rows(), outputs}, {source.capacity.rows(), outputs});

    auto op = PendingOp {};
    op.kind = OpKind::linear;
    op.result = result.index();
    op.first = input.index();
    op.weight = &weight;
    op.bias = bias;
    op.activation = activation;
    pending = op;

    return result;
}

Tensor KernelNet::linearAdd(const Tensor& input,
                            const Weight& weight,
                            const Weight* bias,
                            const Tensor& stream)
{
    flush();

    auto op = PendingOp {};
    op.kind = OpKind::linearAdd;
    op.result = stream.index();
    op.first = input.index();
    op.weight = &weight;
    op.bias = bias;
    pending = op;

    return stream;
}

Tensor KernelNet::add(const Tensor& stream, const Tensor& addend)
{
    flush();

    auto op = PendingOp {};
    op.kind = OpKind::add;
    op.result = stream.index();
    op.first = stream.index();
    op.second = addend.index();
    pending = op;

    return stream;
}

Tensor KernelNet::layerNorm(const Tensor& input,
                            const Weight& weight,
                            const Weight& bias)
{
    flush();

    const auto& source = recordOf(input);
    auto result = fresh(source.shape, source.capacity);

    auto op = PendingOp {};
    op.kind = OpKind::layerNorm;
    op.result = result.index();
    op.first = input.index();
    op.weight = &weight;
    op.bias = &bias;
    pending = op;

    return result;
}

Tensor KernelNet::attention(const Tensor& queries,
                            const Tensor& keys,
                            const Tensor& values,
                            int heads,
                            bool causal)
{
    flush();

    const auto& source = recordOf(queries);
    auto result = fresh(source.shape, source.capacity);

    auto op = PendingOp {};
    op.kind = OpKind::attention;
    op.result = result.index();
    op.first = queries.index();
    op.second = keys.index();
    op.third = values.index();
    op.heads = heads;
    op.causal = causal;
    pending = op;

    return result;
}

Tensor KernelNet::embed(const Tensor& tokens,
                        const Weight& tokenTable,
                        const Weight& positionTable,
                        int firstPosition)
{
    flush();

    const auto& source = recordOf(tokens);
    const auto width = tokenTable.dimension(1);

    auto result = fresh({source.shape[0], width}, {source.capacity[0], width});

    auto op = PendingOp {};
    op.kind = OpKind::embed;
    op.result = result.index();
    op.first = tokens.index();
    op.weight = &tokenTable;
    op.positionTable = &positionTable;
    op.position = firstPosition;
    pending = op;

    return result;
}

// Appending is a bind, not a copy: the projection writes straight into the
// rows the cache holds next, as a byte range into its buffer, and whatever
// attends to the cache afterwards reads it from row zero. Without the ranged
// bind a step would project into a scratch row and copy it into place, a
// dispatch per layer per token whose only work is a memcpy.
Tensor KernelNet::appendLinear(Cache& cache,
                               const Tensor& input,
                               const Weight& weight,
                               const Weight* bias)
{
    flush();

    const auto& storage = caches[cache.id];
    const auto width = storage.width;
    const auto count = recordOf(input).shape.rows();

    if (cache.rows + count > storage.capacity)
        throw std::logic_error {"a cache was appended to past its capacity"};

    auto op = PendingOp {};
    op.kind = OpKind::append;
    op.first = input.index();
    op.weight = &weight;
    op.bias = bias;
    op.target = {
        &storage.buffer, floatBytes(cache.rows * width), floatBytes(count * width)};

    cache.rows += count;

    auto result = cached(cache);
    op.result = result.index();
    pending = op;

    return result;
}

// A convolution is its windows unfolded into rows, over an input addressed by
// its two strides, and one projection of them against the weight as the file
// lays it out, [out, in, k] being [out, in * k].
void KernelNet::runConvolution(const PendingOp& op)
{
    const auto& input = records[op.first];
    const auto& output = records[op.result];
    const auto channels = input.shape[1];
    const auto taps = op.weight->dimension(2);
    const auto outputLength = output.shape[0];

    unfold.input = input.storage;
    unfold.columns = *columns;
    unfold.inputChannelCount = (std::uint32_t) channels;
    unfold.inputLength = (std::uint32_t) input.shape[0];
    unfold.kernelSize = (std::uint32_t) taps;
    unfold.stride = (std::uint32_t) op.stride;
    unfold.padding = (std::uint32_t) op.padding;
    unfold.inputChannelStride = (std::uint32_t) input.columnStride;
    unfold.inputFrameStride = (std::uint32_t) input.rowStride;

    order({input.storage.buffer}, {&*columns});
    activePass().dispatch(unfold, channels * taps, outputLength);

    dispatchProjection(BufferRange::of(*columns),
                       outputLength,
                       channels * taps,
                       *op.weight,
                       op.bias->buffer,
                       output.storage,
                       op.activation == Activation::gelu,
                       false);
}

void KernelNet::runProjection(const PendingOp& op)
{
    const auto& input = records[op.first];
    const auto target =
        op.kind == OpKind::append ? op.target : records[op.result].storage;
    const auto& bias =
        op.bias != nullptr ? op.bias->buffer : zeroBiasFor(op.weight->dimension(0));

    dispatchProjection(input.storage,
                       input.shape.rows(),
                       input.shape.columns(),
                       *op.weight,
                       bias,
                       target,
                       op.activation == Activation::gelu,
                       op.kind == OpKind::linearAdd);
}

// The one place a weight's storage decides anything: a tensor the loader left
// packed goes to the program that reads packed halves, and one it uploaded as
// floats to the program that subscripts floats. The row count decides the
// other axis in the decoder profile, a step's few rows taking the split form
// and the vocabulary projection its own split count.
void KernelNet::dispatchProjection(const BufferRange& input,
                                   int rowCount,
                                   int innerCount,
                                   const Weight& weight,
                                   const Buffer& bias,
                                   const BufferRange& target,
                                   bool gelu,
                                   bool residual)
{
    const auto outputWidth = weight.dimension(0);
    auto& pass = activePass();

    if (residual)
        order({input.buffer, target.buffer}, {target.buffer});
    else
        order({input.buffer}, {target.buffer});

    const auto vocabularyWide = outputWidth > layout.widestLayerOutput;

    const auto dispatchWith = [&](auto& program)
    {
        dispatchLinear(program,
                       pass,
                       input,
                       weight.buffer,
                       bias,
                       target,
                       innerCount,
                       outputWidth,
                       rowCount,
                       gelu,
                       residual);
    };

    if (splitsProjection(rowCount) && weight.isPackedHalf())
        dispatchWith(vocabularyWide ? packedSplitLogits : packedSplitProjection);
    else if (splitsProjection(rowCount))
        dispatchWith(vocabularyWide ? splitLogits : splitProjection);
    else if (weight.isPackedHalf())
        dispatchWith(packedProjection);
    else
        dispatchWith(projection);
}

void KernelNet::runAdd(const PendingOp& op)
{
    const auto& stream = records[op.first];
    const auto& addend = records[op.second];

    sum.output = stream.storage;
    sum.addend = addend.storage;

    order({stream.storage.buffer, addend.storage.buffer}, {stream.storage.buffer});
    activePass().dispatch(sum, stream.shape.elementCount());
}

void KernelNet::runLayerNorm(const PendingOp& op)
{
    const auto& input = records[op.first];
    const auto& output = records[op.result];

    normalisation.input = input.storage;
    normalisation.weight = op.weight->buffer;
    normalisation.bias = op.bias->buffer;
    normalisation.output = output.storage;
    normalisation.rowLength = (std::uint32_t) input.shape.columns();

    order({input.storage.buffer}, {output.storage.buffer});
    normalisation.dispatchRows(activePass(), input.shape.rows());
}

void KernelNet::runAttention(const PendingOp& op)
{
    if (profile == KernelProfile::encoder)
        runWindowAttention(op);
    else
        runCachedAttention(op);
}

// The scores are a product of the queries against the keys, one batch per head
// over the head's columns, in the same layout as the projections: a key row is
// contiguous along the head dimension exactly as a weight row is along its
// inputs.
void KernelNet::runWindowAttention(const PendingOp& op)
{
    const auto& queries = records[op.first];
    const auto& keys = records[op.second];
    const auto& values = records[op.third];
    const auto& output = records[op.result];

    const auto width = queries.shape.columns();
    const auto headWidth = width / op.heads;
    const auto queryCount = queries.shape.rows();
    const auto keyCount = keys.shape.rows();

    const auto& scratch = scoresFor(keys.capacity.rows());
    const auto& scores = scratch.scores;
    const auto& maxima = *scratch.tileMaxima;
    auto& pass = activePass();

    windowScores.a = queries.storage;
    windowScores.b = keys.storage;
    windowScores.bias = zeroBiasFor(keyCount);
    windowScores.output = scores;
    windowScores.tileMaxima = maxima;

    order({queries.storage.buffer, keys.storage.buffer}, {&scores, &maxima});
    windowScores.dispatch(
        pass,
        TiledMatMulShape::forAttentionScores(queryCount,
                                             keyCount,
                                             op.heads,
                                             headWidth,
                                             width,
                                             attentionScale(headWidth),
                                             op.causal));

    windowApply.a = scores;
    windowApply.b = values.storage;
    windowApply.bias = zeroBiasFor(headWidth);
    windowApply.output = output.storage;
    windowApply.rowMaxima = maxima;

    order({&scores, &maxima, values.storage.buffer}, {output.storage.buffer});
    windowApply.dispatch(pass,
                         TiledMatMulShape::forAttentionApply(
                             queryCount, keyCount, op.heads, headWidth, width));
}

// scores, softmax, apply over whichever keys and values a cache holds, the
// softmax normalising the scores where they lie. A single query, which is every
// step after the prompt, takes the fused pair instead: the keys split across
// groups and joined, the scores never leaving shared memory. The query stands
// at the last position there, so a causal mask masks nothing.
void KernelNet::runCachedAttention(const PendingOp& op)
{
    const auto& queries = records[op.first];
    const auto& keys = records[op.second];
    const auto& values = records[op.third];
    const auto& output = records[op.result];

    const auto width = queries.shape.columns();
    const auto headWidth = width / op.heads;
    const auto queryCount = queries.shape.rows();
    const auto keyCount = keys.shape.rows();
    auto& pass = activePass();

    if (queryCount == 1)
    {
        order({queries.storage.buffer, keys.storage.buffer, values.storage.buffer},
              {output.storage.buffer});

        singleQueryAttention.encode(pass,
                                    *queries.storage.buffer,
                                    *keys.storage.buffer,
                                    *values.storage.buffer,
                                    *output.storage.buffer,
                                    width,
                                    headWidth,
                                    keyCount,
                                    attentionScale(headWidth));
        return;
    }

    const auto& scores = scoresFor(keys.capacity.rows()).scores;
    const auto scoreRows = op.heads * queryCount;

    cachedScores.queries = queries.storage;
    cachedScores.keys = keys.storage;
    cachedScores.scores = scores;
    cachedScores.modelWidth = (std::uint32_t) width;
    cachedScores.headWidth = (std::uint32_t) headWidth;
    cachedScores.queryCount = (std::uint32_t) queryCount;
    cachedScores.keyCount = (std::uint32_t) keyCount;
    cachedScores.causal = op.causal ? 1u : 0u;
    cachedScores.scale = attentionScale(headWidth);

    order({queries.storage.buffer, keys.storage.buffer}, {&scores});
    pass.dispatch(cachedScores, keyCount, scoreRows);

    softmax.values = scores;
    softmax.rowLength = (std::uint32_t) keyCount;

    order({&scores}, {&scores});
    softmax.dispatchRows(pass, scoreRows);

    cachedApply.probabilities = scores;
    cachedApply.values = values.storage;
    cachedApply.output = output.storage;
    cachedApply.modelWidth = (std::uint32_t) width;
    cachedApply.headWidth = (std::uint32_t) headWidth;
    cachedApply.queryCount = (std::uint32_t) queryCount;
    cachedApply.keyCount = (std::uint32_t) keyCount;

    order({&scores, values.storage.buffer}, {output.storage.buffer});
    cachedApply.dispatch(pass, width, queryCount);
}

void KernelNet::runEmbed(const PendingOp& op)
{
    const auto& tokens = records[op.first];
    const auto& output = records[op.result];
    const auto width = output.shape.columns();

    embedding.tokens = tokens.storage;
    embedding.tokenTable = op.weight->buffer;
    embedding.positionTable = op.positionTable->buffer;
    embedding.output = output.storage;
    embedding.width = (std::uint32_t) width;
    embedding.firstPosition = (std::uint32_t) op.position;

    order({tokens.storage.buffer}, {output.storage.buffer});
    activePass().dispatch(embedding, width, tokens.shape[0]);
}
} // namespace WSP
