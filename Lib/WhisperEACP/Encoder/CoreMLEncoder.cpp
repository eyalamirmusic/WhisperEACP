#include "CoreMLEncoder.h"

#if EACP_HAS_COREML

#include <WhisperEACP/Net/CoreMLMel.h>

#include <array>
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

constexpr auto framesPerPosition =
    EncoderShape::firstConvolutionStride * EncoderShape::secondConvolutionStride;
} // namespace

CoreMLEncoder::CoreMLEncoder(const EncoderShape& shapeToUse,
                             Span<const int> positionsToUse)
    : encoderShape(shapeToUse)
{
    for (auto count: positionsToUse)
    {
        if (count <= 0 || count > encoderShape.positions())
            throw std::logic_error {"a Core ML encoder's contexts are between 1 "
                                    "and the window's positions"};

        positionSet.add(count);
    }

    if (positionSet.empty())
        throw std::logic_error {"a Core ML encoder is compiled for at least one "
                                "audio context"};
}

eacp::ML::Result CoreMLEncoder::prepare(const EncoderWeights& hostWeights,
                                        const CoreMLEncoderOptions& options)
{
    const auto program = record(hostWeights, options);

    const auto start = Clock::now();
    auto result = model.load(program.package, program.modelOptions);

    if (result.ok)
        result = loadMel(program);

    loadSeconds = secondsSince(start);

    if (result.ok)
        adopt(program.features);

    return result;
}

eacp::ML::Result CoreMLEncoder::loadMel(const RecordedProgram& program)
{
    if (!program.melPackage)
        return eacp::ML::Result::success();

    return melModel.load(*program.melPackage, program.modelOptions);
}

eacp::Threads::Async<eacp::ML::Result>
    CoreMLEncoder::prepareAsync(const EncoderWeights& hostWeights,
                                const CoreMLEncoderOptions& options)
{
    const auto program = record(hostWeights, options);
    const auto features = program.features;
    const auto promise = eacp::Threads::AsyncPromise<eacp::ML::Result> {};
    const auto start = Clock::now();

    // this outlives both for encodeAsync's reason: the models are members,
    // and destroying one abandons its load's Async.
    const auto settle =
        [this, promise, features, start](const eacp::ML::Result& result)
    {
        loading = false;
        loadSeconds = secondsSince(start);

        if (result.ok)
            adopt(features);

        promise.resolve(result);
    };

    const auto fail = [settle](const std::string& error)
    { settle(eacp::ML::Result::failure(error)); };

    const auto loaded = [this, settle, fail, program](const eacp::ML::Result& result)
    {
        if (!result.ok || !program.melPackage)
        {
            settle(result);
            return;
        }

        melModel.loadAsync(*program.melPackage, program.modelOptions)
            .then(settle, fail);
    };

    loading = true;
    model.loadAsync(program.package, program.modelOptions).then(loaded, fail);

    return promise.get();
}

CoreMLEncoder::RecordedProgram
    CoreMLEncoder::record(const EncoderWeights& hostWeights,
                          const CoreMLEncoderOptions& options)
{
    if (isAnyInFlight())
        throw std::logic_error {"the Core ML encoder is prepared again only once "
                                "every encodeAsync has resolved"};

    if (loading)
        throw std::logic_error {"the Core ML encoder is prepared again only once "
                                "its prepareAsync has resolved"};

    unprepare();

    if (!(hostWeights.shape == encoderShape))
        throw ModelError {"the weights were loaded against a different encoder "
                          "shape than this encoder was built for"};

    auto netOptions = CoreMLNetOptions {};
    netOptions.fusedAttention = options.fusedAttention;
    netOptions.framesPerPosition = framesPerPosition;
    netOptions.framesMajorInput = options.frontEnd.has_value();

    auto net = CoreMLNet {positionSet, netOptions};

    // The Core ML net addresses features by name and reads no buffer, so the
    // bindings carry names and nothing else.
    const auto mel =
        Binding {{}, Shape {encoderShape.melBins, encoderShape.inputFrames}, "mel"};
    const auto rows = Binding {{}, {}, "rows"};

    recordEncoder(
        net, encoderShape, hostWeights, mel, rows, encoderShape.inputFrames);

    if (!net.isValid())
        throw ModelError {"the Core ML encoder could not be recorded: "
                          + net.errors()[0]};

    auto modelOptions = eacp::ML::Options {};
    modelOptions.units = options.units;
    modelOptions.cacheDirectory = options.cacheDirectory;

    auto program = RecordedProgram {net.build(),
                                    modelOptions,
                                    {net.inputNames()[0],
                                     net.outputNames()[0],
                                     options.fusedAttention,
                                     std::nullopt},
                                    std::nullopt};

    if (options.frontEnd)
    {
        const auto& frontEnd = *options.frontEnd;

        if (frontEnd.shape.melCount != encoderShape.melBins
            || frontEnd.shape.frameCount < encoderShape.inputFrames)
            throw std::logic_error {"the Core ML front end makes a mel of another "
                                    "shape than the encoder reads"};

        const auto mel = recordLogMelProgram(frontEnd.shape, frontEnd.filterBank);

        if (!mel.isValid())
            throw ModelError {"the Core ML mel could not be recorded: "
                              + mel.errors()[0]};

        program.melPackage = mel.build();
        program.features.melShape = frontEnd.shape;
    }

    return program;
}

void CoreMLEncoder::adopt(const ProgramFeatures& features)
{
    inputName = features.inputName;
    outputName = features.outputName;
    fusedAttention = features.fusedAttention;
    computesMel = features.melShape.has_value();
    samplesIn = {};
    melOut = {};

    if (computesMel)
    {
        melShape = *features.melShape;
        samplesIn = eacp::ML::MultiArray::create(logMelInputShape(melShape),
                                                 eacp::ML::DType::float16);
        melOut = eacp::ML::MultiArray::create(
            {melShape.frameCount, melShape.melCount}, eacp::ML::DType::float16);

        if (!samplesIn.isValid() || !melOut.isValid())
            throw ModelError {"Core ML could not make the mel model's arrays"};
    }

    plan.reset();
    prepared = true;
}

void CoreMLEncoder::unprepare()
{
    prepared = false;
    members.clear();
    plan.reset();
}

bool CoreMLEncoder::isAnyInFlight() const
{
    const auto isInFlight = [](const Member& member) { return member.inFlight; };
    return members.findIf(isInFlight) != nullptr;
}

CoreMLEncoder::Member& CoreMLEncoder::memberFor(int positionCount)
{
    if (!isPrepared())
        throw std::logic_error {"the Core ML encoder runs after prepare()"};

    if (!positionSet.contains(positionCount))
        throw ModelError {"the Core ML encoder is compiled for "
                          + std::to_string(positionSet.size())
                          + " audio contexts, and " + std::to_string(positionCount)
                          + " is not one of them"};

    const auto isCount = [positionCount](const Member& member)
    { return member.positions == positionCount; };

    if (auto* found = members.findIf(isCount))
    {
        if (found->inFlight)
            throw std::logic_error {"the Core ML encoder is already running an "
                                    "encodeAsync at "
                                    + std::to_string(positionCount)
                                    + " positions, and its arrays are in use"};

        return *found;
    }

    const auto frames = framesPerPosition * positionCount;
    const auto inputShape = computesMel
                                ? eacp::ML::Shape {1, frames, encoderShape.melBins}
                                : eacp::ML::Shape {1, encoderShape.melBins, frames};

    auto member = Member {};
    member.positions = positionCount;
    member.mel = eacp::ML::MultiArray::create(inputShape, eacp::ML::DType::float16);
    member.rows = eacp::ML::MultiArray::create({positionCount, encoderShape.width},
                                               eacp::ML::DType::float16);

    if (!member.mel.isValid() || !member.rows.isValid())
        throw ModelError {"Core ML could not make the arrays for a context of "
                          + std::to_string(positionCount) + " positions"};

    members.add(member);
    return members.back();
}

// The first 2n frames of each band, each band a window's frames apart in the
// buffer; the copy waits for the GPU work that wrote the mel.
CoreMLEncoder::Member& CoreMLEncoder::filledFromMel(const eacp::GPU::Buffer& mel,
                                                    int positionCount)
{
    if (computesMel)
        throw std::logic_error {"this Core ML encoder computes its own mel, and "
                                "is run from the samples"};

    auto& member = memberFor(positionCount);
    const auto bandStride = (size_t) encoderShape.inputFrames * sizeof(float);

    if (mel.size() < encoderShape.melElementCount() * (int) sizeof(float))
        throw std::logic_error {"the mel buffer is smaller than the encoder's "
                                "window"};

    member.mel.copyFrom(mel, 0, bandStride, eacp::ML::DType::float32);

    return member;
}

// The samples array is the model's one, so no context may still be reading it.
CoreMLEncoder::Member& CoreMLEncoder::filledFromSamples(Span<const float> samples,
                                                        int positionCount)
{
    if (!computesMel)
        throw std::logic_error {"this Core ML encoder reads a mel, and is run "
                                "from the mel buffer"};

    if (samples.size() != melShape.sampleCount)
        throw std::logic_error {"a Core ML encoder with a front end reads the "
                                "whole window, "
                                + std::to_string(melShape.sampleCount)
                                + " samples, and was handed "
                                + std::to_string(samples.size())};

    if (isAnyInFlight())
        throw std::logic_error {"the Core ML encoder's mel arrays are in use by "
                                "an encodeAsync"};

    auto& member = memberFor(positionCount);

    const auto start = Clock::now();
    samplesIn.fromFloats(samples);
    sampleCopySeconds = secondsSince(start);

    return member;
}

// The mel predicted on the caller's thread and handed over.
CoreMLEncoder::Member& CoreMLEncoder::melFromSamples(Span<const float> samples,
                                                     int positionCount)
{
    auto& member = filledFromSamples(samples, positionCount);

    auto inputs = eacp::ML::Inputs {};
    inputs[samplesFeature] = samplesIn;

    auto outputs = eacp::ML::Outputs {};
    outputs[melFeature] = melOut;

    const auto start = Clock::now();
    const auto result = melModel.predict(inputs, outputs);
    melSeconds = secondsSince(start);

    if (!result.ok)
        throw ModelError {"Core ML refused the mel's prediction: " + result.error};

    copyMelInto(member);

    return member;
}

// The context's frames are the first rows of the frames-major mel, so the
// hand-over is one row copy, fp16 to fp16.
void CoreMLEncoder::copyMelInto(Member& member)
{
    member.mel.copyRows(melOut, 0, 0, framesPerPosition * member.positions);
}

CoreMLEncoder::Member* CoreMLEncoder::findMember(int positionCount)
{
    const auto isCount = [positionCount](const Member& member)
    { return member.positions == positionCount; };

    return members.findIf(isCount);
}

eacp::ML::Inputs CoreMLEncoder::inputsFor(const Member& member) const
{
    auto inputs = eacp::ML::Inputs {};
    inputs[inputName] = member.mel;

    return inputs;
}

// MultiArray::copyTo copies nothing into a buffer that is too small, which
// would leave the caller reading stale rows.
void CoreMLEncoder::requireRowsFit(const eacp::GPU::Buffer& rows,
                                   int positionCount) const
{
    const auto needed =
        (size_t) positionCount * (size_t) encoderShape.width * sizeof(float);

    if ((size_t) rows.size() < needed)
        throw std::logic_error {"the rows buffer is smaller than "
                                + std::to_string(positionCount)
                                + " rows of the encoder's width"};
}

void CoreMLEncoder::encode(const eacp::GPU::Buffer& mel,
                           eacp::GPU::Buffer& rows,
                           int positionCount)
{
    requireRowsFit(rows, positionCount);
    predictInto(filledFromMel(mel, positionCount), &rows);
}

void CoreMLEncoder::encode(const eacp::GPU::Buffer& mel, int positionCount)
{
    predictInto(filledFromMel(mel, positionCount), nullptr);
}

eacp::Threads::Async<eacp::ML::Result> CoreMLEncoder::encodeAsync(
    const eacp::GPU::Buffer& mel, eacp::GPU::Buffer& rows, int positionCount)
{
    requireRowsFit(rows, positionCount);
    return predictAsyncInto(filledFromMel(mel, positionCount), &rows);
}

eacp::Threads::Async<eacp::ML::Result>
    CoreMLEncoder::encodeAsync(const eacp::GPU::Buffer& mel, int positionCount)
{
    return predictAsyncInto(filledFromMel(mel, positionCount), nullptr);
}

void CoreMLEncoder::encode(Span<const float> samples,
                           eacp::GPU::Buffer& rows,
                           int positionCount)
{
    requireRowsFit(rows, positionCount);
    predictInto(melFromSamples(samples, positionCount), &rows);
}

void CoreMLEncoder::encode(Span<const float> samples, int positionCount)
{
    predictInto(melFromSamples(samples, positionCount), nullptr);
}

eacp::Threads::Async<eacp::ML::Result> CoreMLEncoder::encodeAsync(
    Span<const float> samples, eacp::GPU::Buffer& rows, int positionCount)
{
    requireRowsFit(rows, positionCount);
    return predictAsyncFromSamples(filledFromSamples(samples, positionCount), &rows);
}

eacp::Threads::Async<eacp::ML::Result>
    CoreMLEncoder::encodeAsync(Span<const float> samples, int positionCount)
{
    return predictAsyncFromSamples(filledFromSamples(samples, positionCount),
                                   nullptr);
}

// The mel on its model's queue, the hand-over on the main thread, and the
// encoder on its own queue after it. The context is marked in flight from the
// start, so the shared mel arrays are not touched by another encode until the
// whole chain has settled.
eacp::Threads::Async<eacp::ML::Result>
    CoreMLEncoder::predictAsyncFromSamples(Member& member, eacp::GPU::Buffer* rows)
{
    const auto positionCount = member.positions;
    const auto promise = eacp::Threads::AsyncPromise<eacp::ML::Result> {};

    auto inputs = eacp::ML::Inputs {};
    inputs[samplesFeature] = samplesIn;

    auto outputs = eacp::ML::Outputs {};
    outputs[melFeature] = melOut;

    const auto fail = [this, promise, positionCount](const std::string& error)
    {
        finishAsync(positionCount);
        promise.resolve(eacp::ML::Result::failure(error));
    };

    const auto settle = [promise](const eacp::ML::Result& result)
    { promise.resolve(result); };

    const auto settleFailure = [promise](const std::string& error)
    { promise.resolve(eacp::ML::Result::failure(error)); };

    const auto encodeMel = [this, fail, settle, settleFailure, positionCount, rows](
                               const eacp::ML::Prediction& done)
    {
        melSeconds = done.predictSeconds;
        auto* waiting = findMember(positionCount);

        if (!done.ok || waiting == nullptr)
        {
            fail(done.error);
            return;
        }

        copyMelInto(*waiting);
        predictAsyncInto(*waiting, rows).then(settle, settleFailure);
    };

    member.inFlight = true;
    melModel.predictAsync(inputs, outputs).then(encodeMel, fail);

    return promise.get();
}

const eacp::ML::MultiArray& CoreMLEncoder::rowsAt(int positionCount) const
{
    const auto isCount = [positionCount](const Member& member)
    { return member.positions == positionCount; };

    const auto* found = members.findIf(isCount);

    if (found == nullptr || found->inFlight)
        throw std::logic_error {"the Core ML encoder has no finished rows at "
                                + std::to_string(positionCount) + " positions"};

    return found->rows;
}

void CoreMLEncoder::predictInto(Member& member, eacp::GPU::Buffer* rows)
{
    auto outputs = eacp::ML::Outputs {};
    outputs[outputName] = member.rows;

    const auto start = Clock::now();
    const auto result = model.predict(inputsFor(member), outputs);
    predictSeconds = secondsSince(start);

    if (!result.ok)
        throw ModelError {"Core ML refused the encoder's prediction at "
                          + std::to_string(member.positions)
                          + " positions: " + result.error};

    if (rows != nullptr)
        member.rows.copyTo(*rows, eacp::ML::DType::float32);
}

eacp::Threads::Async<eacp::ML::Result>
    CoreMLEncoder::predictAsyncInto(Member& member, eacp::GPU::Buffer* rows)
{
    const auto positionCount = member.positions;
    const auto inputs = inputsFor(member);

    auto outputs = eacp::ML::Outputs {};
    outputs[outputName] = member.rows;

    const auto promise = eacp::Threads::AsyncPromise<eacp::ML::Result> {};
    const auto output = member.rows;
    auto* target = rows;

    // this outlives both: the model is a member, and destroying it abandons
    // the prediction's Async, so neither runs after the encoder is gone.
    const auto copyOut = [this, promise, output, target, positionCount](
                             const eacp::ML::Prediction& done)
    {
        predictSeconds = done.predictSeconds;
        finishAsync(positionCount);

        if (done.ok && target != nullptr)
            output.copyTo(*target, eacp::ML::DType::float32);

        promise.resolve(eacp::ML::Result {done.ok, done.error});
    };

    const auto fail = [this, promise, positionCount](const std::string& error)
    {
        finishAsync(positionCount);
        promise.resolve(eacp::ML::Result::failure(error));
    };

    member.inFlight = true;
    model.predictAsync(inputs, outputs).then(copyOut, fail);

    return promise.get();
}

void CoreMLEncoder::finishAsync(int positionCount)
{
    const auto isCount = [positionCount](const Member& member)
    { return member.positions == positionCount; };

    if (auto* member = members.findIf(isCount))
        member->inFlight = false;
}

const eacp::ML::ComputePlan& CoreMLEncoder::computePlan()
{
    if (!plan)
        plan = isPrepared() ? model.computePlan() : eacp::ML::ComputePlan {};

    return *plan;
}

std::string describePlacement(const eacp::ML::ComputePlan& plan,
                              const std::string& indent)
{
    using Device = eacp::ML::ComputePlan::Device;

    if (plan.isEmpty())
        return indent
               + "no plan: no model is loaded, or this OS has no "
                 "MLComputePlan\n";

    constexpr auto devices =
        std::array {Device::neuralEngine, Device::gpu, Device::cpu, Device::unknown};

    const auto countOn = [&plan](Device device)
    {
        auto count = 0;

        for (const auto& op: plan.ops)
            count += op.device == device ? 1 : 0;

        return count;
    };

    auto text = indent + std::to_string(plan.ops.size()) + " ops:";

    for (auto device: devices)
        text += " " + std::to_string(countOn(device)) + " on "
                + eacp::ML::toString(device) + ",";

    text.back() = '\n';

    struct Group
    {
        std::string type;
        Device device = Device::unknown;
        int count = 0;
    };

    auto offTheEngine = Vector<Group> {};

    for (const auto& op: plan.ops)
    {
        if (op.device == Device::neuralEngine)
            continue;

        const auto isSame = [&op](const Group& group)
        { return group.type == op.type && group.device == op.device; };

        if (auto* group = offTheEngine.findIf(isSame))
            ++group->count;
        else
            offTheEngine.add(Group {op.type, op.device, 1});
    }

    for (const auto& group: offTheEngine)
        text += indent + "  off the engine: " + std::to_string(group.count) + " "
                + group.type + " on " + eacp::ML::toString(group.device) + "\n";

    return text;
}
} // namespace WSP

#endif
