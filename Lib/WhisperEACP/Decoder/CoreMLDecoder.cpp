#include "CoreMLDecoder.h"

#if EACP_HAS_COREML

#include <chrono>
#include <stdexcept>
#include <string>

namespace WSP
{
namespace
{
using Clock = std::chrono::steady_clock;

double secondsSince(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

constexpr auto encoderRowsFeature = "encoder_rows";
constexpr auto tokensFeature = "tokens";
constexpr auto hiddenFeature = "hidden_states";
constexpr auto logitsFeature = "logits";

void addCaches(Vector<Cache>& caches, Net& net, int layers, int rows, int width)
{
    caches.clear();

    for (auto index = 0; index < layers; ++index)
        caches.add(net.makeCache(rows, width));
}

// The order Decoder::prepare makes them in, so the two programs, each with a
// net of its own, number every cache alike and name its feature alike.
DecoderCaches makeCaches(Net& net, const DecoderShape& shape)
{
    auto caches = DecoderCaches {};
    const auto width = shape.width;

    addCaches(caches.selfKeys, net, shape.layers, shape.maxPositions, width);
    addCaches(caches.selfValues, net, shape.layers, shape.maxPositions, width);
    addCaches(caches.crossKeys, net, shape.layers, shape.crossPositions, width);
    addCaches(caches.crossValues, net, shape.layers, shape.crossPositions, width);

    return caches;
}

Vector<Cache> selfCachesOf(const DecoderCaches& caches)
{
    auto all = caches.selfKeys;

    for (const auto& cache: caches.selfValues)
        all.add(cache);

    return all;
}

Vector<Cache> crossCachesOf(const DecoderCaches& caches)
{
    auto all = caches.crossKeys;

    for (const auto& cache: caches.crossValues)
        all.add(cache);

    return all;
}

Binding named(const Shape& capacity, const std::string& name)
{
    return Binding {{}, capacity, name};
}

eacp::ML::MultiArray makeArray(const eacp::ML::Shape& shape, eacp::ML::DType type)
{
    auto array = eacp::ML::MultiArray::create(shape, type);

    if (!array.isValid())
        throw ModelError {"Core ML could not make the decoder's arrays"};

    return array;
}

void requireRecorded(const CoreMLNet& net, std::string_view what)
{
    if (!net.isValid())
        throw ModelError {"the Core ML decoder's " + std::string {what}
                          + " could not be recorded: " + net.errors()[0]};
}
} // namespace

CoreMLDecoder::CoreMLDecoder(const DecoderShape& shapeToUse)
    : decoderShape(shapeToUse)
{
}

bool CoreMLDecoder::isPrepared() const
{
    return prepared && startModel.isLoaded() && stepModel.isLoaded();
}

bool CoreMLDecoder::wasCacheHit() const
{
    return startModel.wasCacheHit() && stepModel.wasCacheHit();
}

eacp::ML::Result CoreMLDecoder::prepare(const DecoderWeights& hostWeights,
                                        const CoreMLDecoderOptions& options)
{
    const auto programs = record(hostWeights, options);
    const auto start = Clock::now();

    auto result = startModel.load(programs.start, programs.modelOptions);

    if (result.ok)
        result = stepModel.load(programs.step, programs.modelOptions);

    loadSeconds = secondsSince(start);

    if (result.ok)
        adopt(programs);

    return result;
}

eacp::Threads::Async<eacp::ML::Result>
    CoreMLDecoder::prepareAsync(const DecoderWeights& hostWeights,
                                const CoreMLDecoderOptions& options)
{
    const auto programs = record(hostWeights, options);
    const auto promise = eacp::Threads::AsyncPromise<eacp::ML::Result> {};
    const auto start = Clock::now();

    const auto settle = [this, promise, start](const eacp::ML::Result& result)
    {
        loading = false;
        loadSeconds = secondsSince(start);
        promise.resolve(result);
    };

    const auto fail = [settle](const std::string& error)
    { settle(eacp::ML::Result::failure(error)); };

    // this outlives both loads: the models are members, and destroying one
    // abandons its Async.
    const auto stepLoaded = [this, settle, programs](const eacp::ML::Result& result)
    {
        if (result.ok)
            adopt(programs);

        settle(result);
    };

    const auto startLoaded =
        [this, settle, fail, stepLoaded, programs](const eacp::ML::Result& result)
    {
        if (!result.ok)
        {
            settle(result);
            return;
        }

        stepModel.loadAsync(programs.step, programs.modelOptions)
            .then(stepLoaded, fail);
    };

    loading = true;
    startModel.loadAsync(programs.start, programs.modelOptions)
        .then(startLoaded, fail);

    return promise.get();
}

CoreMLDecoder::Programs CoreMLDecoder::record(const DecoderWeights& hostWeights,
                                              const CoreMLDecoderOptions& options)
{
    if (loading)
        throw std::logic_error {"the Core ML decoder is prepared again only once "
                                "its prepareAsync has resolved"};

    unprepare();

    if (!(hostWeights.shape == decoderShape))
        throw ModelError {"the weights were loaded against a different decoder "
                          "shape than this decoder was built for"};

    if (hostWeights.placement != WeightPlacement::Host)
        throw std::logic_error {"the Core ML decoder writes host weights into its "
                                "blob; load them with WeightPlacement::Host"};

    auto netOptions = CoreMLNetOptions {};
    netOptions.fusedAttention = options.fusedAttention;

    auto startNet = CoreMLNet {netOptions};
    auto startCaches = makeCaches(startNet, decoderShape);

    recordSequenceStart(
        startNet,
        decoderShape,
        hostWeights,
        startCaches,
        named({decoderShape.crossPositions, decoderShape.width}, encoderRowsFeature),
        decoderShape.crossPositions);

    requireRecorded(startNet, "sequence start");

    netOptions.selection = CoreMLSelection {};
    netOptions.selection->onTheEngine = options.selectsOnTheEngine;
    netOptions.embedsTokenByProduct = options.embedsTokenByProduct;
    netOptions.embedsPositionByProduct = options.embedsPositionByProduct;
    netOptions.omittedOutputs.add(hiddenFeature);

    if (!options.keepsLogits)
        netOptions.omittedOutputs.add(logitsFeature);

    auto stepNet = CoreMLNet {netOptions};
    caches = makeCaches(stepNet, decoderShape);

    recordDecoderStep(stepNet,
                      decoderShape,
                      hostWeights,
                      caches,
                      named({decoderShape.maxPositions}, tokensFeature),
                      1,
                      0,
                      named({}, hiddenFeature),
                      named({}, logitsFeature));

    requireRecorded(stepNet, "step");

    auto modelOptions = eacp::ML::Options {};
    modelOptions.units = options.units;
    modelOptions.cacheDirectory = options.cacheDirectory;

    return {startNet.build(),
            stepNet.build(),
            modelOptions,
            options.fusedAttention,
            options.keepsLogits,
            options.embedsTokenByProduct,
            options.embedsPositionByProduct,
            options.selectsOnTheEngine};
}

// Every array a sequence touches, made once and bound once: the start's
// outputs are the very arrays the step reads its cross caches from, and the
// step's token and rows come back into arrays of their own.
void CoreMLDecoder::adopt(const Programs& programs)
{
    using eacp::ML::DType;

    const auto width = decoderShape.width;
    const auto half = DType::float16;
    const auto selection = CoreMLSelection {};

    fusedAttention = programs.fusedAttention;
    keepsLogits = programs.keepsLogits;
    embedsTokenByProduct = programs.embedsTokenByProduct;
    embedsPositionByProduct = programs.embedsPositionByProduct;
    selectsOnTheEngine = programs.selectsOnTheEngine;

    startInputs = {};
    startOutputs = {};
    stepInputs = {};
    stepOutputs = {};
    selfArrays.clear();
    appendedRows.clear();

    encoderRows = makeArray({decoderShape.crossPositions, width}, half);
    tokenIn = embedsTokenByProduct ? makeArray({1, 2}, half)
                                   : makeArray({1}, DType::int32);
    positionIn = embedsPositionByProduct ? makeArray({1, 1}, half)
                                         : makeArray({1}, DType::int32);
    selfMask = makeArray({1, decoderShape.maxPositions}, half);
    crossMask = makeArray({1, decoderShape.crossPositions}, half);
    nextToken =
        selectsOnTheEngine ? makeArray({1, 2}, half) : makeArray({1}, DType::int32);

    startInputs[encoderRowsFeature] = encoderRows;

    for (const auto& cache: crossCachesOf(caches))
    {
        const auto array = makeArray({decoderShape.crossPositions, width}, half);
        startOutputs[CoreMLNet::cacheFeature(cache)] = array;
        stepInputs[CoreMLNet::cacheFeature(cache)] = array;
    }

    for (const auto& cache: selfCachesOf(caches))
    {
        selfArrays.add(makeArray({decoderShape.maxPositions, width}, half));
        appendedRows.add(makeArray({1, width}, half));

        stepInputs[CoreMLNet::cacheFeature(cache)] = selfArrays.back();
        stepOutputs[CoreMLNet::appendedRowsFeature(cache)] = appendedRows.back();
    }

    stepInputs[embedsTokenByProduct ? CoreMLNet::tokenDigitsFeature
                                    : tokensFeature] = tokenIn;
    stepInputs[CoreMLNet::positionFeature] = positionIn;
    stepInputs[CoreMLNet::selfMaskFeature] = selfMask;
    stepInputs[CoreMLNet::crossMaskFeature] = crossMask;
    stepOutputs[selection.token] = nextToken;

    if (keepsLogits)
    {
        logits = makeArray({1, decoderShape.vocabularySize}, half);
        stepOutputs[logitsFeature] = logits;
    }

    allowedRows.clear();
    allowedRows.resize(decoderShape.maxPositions, 0.0f);

    decodedPositions = 0;
    sequenceOpen = false;
    prepared = true;
}

void CoreMLDecoder::unprepare()
{
    prepared = false;
    sequenceOpen = false;
}

void CoreMLDecoder::requireReady() const
{
    if (loading)
        throw std::logic_error {"the Core ML decoder is still loading"};

    if (!isPrepared())
        throw std::logic_error {"the Core ML decoder runs after prepare()"};
}

eacp::ML::MultiArray CoreMLDecoder::makeSuppression(Span<const float> mask) const
{
    auto row = Vector<float> {};
    row.resize(decoderShape.vocabularySize, 0.0f);

    for (auto index = 0; index < mask.size() && index < row.size(); ++index)
        row[index] = mask[index] != 0.0f ? suppressedScore : 0.0f;

    auto array =
        makeArray({1, decoderShape.vocabularySize}, eacp::ML::DType::float16);
    array.fromFloats(row);

    return array;
}

void CoreMLDecoder::beginSequence(const eacp::GPU::Buffer& rows,
                                  int crossPositionCount)
{
    requireReady();

    if (rows.size() < decoderShape.crossElementCount() * (int) sizeof(float))
        throw std::logic_error {"the encoder rows buffer is smaller than the "
                                "decoder's cross positions"};

    encoderRows.copyFrom(rows, eacp::ML::DType::float32);
    openSequence(crossPositionCount);
}

void CoreMLDecoder::beginSequence(const eacp::ML::MultiArray& rows,
                                  int crossPositionCount)
{
    requireReady();

    if (rows.columns() != decoderShape.width || rows.rows() < crossPositionCount)
        throw std::logic_error {"the encoder rows array holds fewer rows than the "
                                "sequence attends to"};

    encoderRows.copyRows(rows, 0, 0, crossPositionCount);
    openSequence(crossPositionCount);
}

void CoreMLDecoder::openSequence(int crossPositionCount)
{
    const auto rows =
        crossPositionCount == 0 ? decoderShape.crossPositions : crossPositionCount;

    if (rows < 0 || rows > decoderShape.crossPositions)
        throw ModelError {"a decoder built for "
                          + std::to_string(decoderShape.crossPositions)
                          + " encoder rows was asked to attend to "
                          + std::to_string(crossPositionCount)};

    auto crossAllowed = Vector<float> {};
    crossAllowed.resize(decoderShape.crossPositions, 0.0f);

    for (auto index = 0; index < rows; ++index)
        crossAllowed[index] = 1.0f;

    crossMask.fromFloats(crossAllowed);

    for (auto& allowed: allowedRows)
        allowed = 0.0f;

    selfMask.fromFloats(allowedRows);

    const auto start = Clock::now();
    const auto result = startModel.predict(startInputs, startOutputs);
    sequenceStartSeconds = secondsSince(start);
    stepSeconds = 0.0;

    if (!result.ok)
        throw ModelError {"Core ML refused the decoder's sequence start: "
                          + result.error};

    decodedPositions = 0;
    sequenceOpen = true;
}

int CoreMLDecoder::step(int token, const eacp::ML::MultiArray& suppression)
{
    requireReady();

    if (!sequenceOpen)
        throw std::logic_error {"a Core ML decoder step comes after "
                                "beginSequence"};

    if (decodedPositions >= decoderShape.maxPositions)
        throw ModelError {"this step would carry the sequence past the "
                          + std::to_string(decoderShape.maxPositions)
                          + " positions this decoder was built for"};

    constexpr auto radix = CoreMLNet::tokenRadix;

    const auto tokenValues =
        embedsTokenByProduct
            ? Array<float, 2> {(float) (token / radix), (float) (token % radix)}
            : Array<float, 2> {(float) token, 0.0f};
    const auto positionValue = (float) decodedPositions;
    tokenIn.fromFloats({tokenValues.data(), tokenIn.elementCount()});
    positionIn.fromFloats({&positionValue, 1});
    stepInputs[CoreMLSelection {}.suppression] = suppression;

    const auto start = Clock::now();
    const auto result = stepModel.predict(stepInputs, stepOutputs);
    stepSeconds += secondsSince(start);

    if (!result.ok)
        throw ModelError {"Core ML refused a decoder step: " + result.error};

    for (auto index = 0; index < selfArrays.size(); ++index)
        selfArrays[index].copyRows(appendedRows[index], 0, decodedPositions, 1);

    allowedRows[decodedPositions] = 1.0f;
    selfMask.fromFloats({allowedRows.data(), decodedPositions + 1});
    ++decodedPositions;

    const auto chosen = nextToken.toFloats();

    if (selectsOnTheEngine)
        return (int) chosen[0] * radix + (int) chosen[1];

    return (int) chosen[0];
}

Vector<float> CoreMLDecoder::lastLogits() const
{
    if (!keepsLogits)
        throw std::logic_error {"the Core ML decoder keeps its logits only under "
                                "CoreMLDecoderOptions::keepsLogits"};

    return logits.toFloats();
}
} // namespace WSP

#endif
