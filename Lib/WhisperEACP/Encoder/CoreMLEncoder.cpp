#include "CoreMLEncoder.h"

#if EACP_HAS_COREML

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
    if (isAnyInFlight())
        throw std::logic_error {"the Core ML encoder is prepared again only once "
                                "every encodeAsync has resolved"};

    unprepare();

    if (!(hostWeights.shape == encoderShape))
        throw ModelError {"the weights were loaded against a different encoder "
                          "shape than this encoder was built for"};

    auto net = CoreMLNet {
        positionSet, CoreMLNetOptions {options.fusedAttention, framesPerPosition}};

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

    const auto start = Clock::now();
    const auto result = model.load(net.build(), modelOptions);
    loadSeconds = secondsSince(start);

    if (!result.ok)
        return result;

    inputName = net.inputNames()[0];
    outputName = net.outputNames()[0];
    fusedAttention = options.fusedAttention;
    prepared = true;

    return result;
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

    auto member = Member {};
    member.positions = positionCount;
    member.mel = eacp::ML::MultiArray::create(
        {1, encoderShape.melBins, framesPerPosition * positionCount},
        eacp::ML::DType::float16);
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
void CoreMLEncoder::fillInput(Member& member, const eacp::GPU::Buffer& mel)
{
    const auto bandStride = (size_t) encoderShape.inputFrames * sizeof(float);

    if (mel.size() < encoderShape.melElementCount() * (int) sizeof(float))
        throw std::logic_error {"the mel buffer is smaller than the encoder's "
                                "window"};

    member.mel.copyFrom(mel, 0, bandStride, eacp::ML::DType::float32);
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
    auto& member = memberFor(positionCount);
    requireRowsFit(rows, positionCount);
    fillInput(member, mel);

    auto inputs = eacp::ML::Inputs {};
    inputs[inputName] = member.mel;

    auto outputs = eacp::ML::Outputs {};
    outputs[outputName] = member.rows;

    const auto start = Clock::now();
    const auto result = model.predict(inputs, outputs);
    predictSeconds = secondsSince(start);

    if (!result.ok)
        throw ModelError {"Core ML refused the encoder's prediction at "
                          + std::to_string(positionCount)
                          + " positions: " + result.error};

    member.rows.copyTo(rows, eacp::ML::DType::float32);
}

eacp::Threads::Async<eacp::ML::Result> CoreMLEncoder::encodeAsync(
    const eacp::GPU::Buffer& mel, eacp::GPU::Buffer& rows, int positionCount)
{
    auto& member = memberFor(positionCount);
    requireRowsFit(rows, positionCount);
    fillInput(member, mel);

    auto inputs = eacp::ML::Inputs {};
    inputs[inputName] = member.mel;

    auto outputs = eacp::ML::Outputs {};
    outputs[outputName] = member.rows;

    const auto promise = eacp::Threads::AsyncPromise<eacp::ML::Result> {};
    const auto output = member.rows;
    auto* target = &rows;

    // this outlives both: the model is a member, and destroying it abandons
    // the prediction's Async, so neither runs after the encoder is gone.
    const auto copyOut = [this, promise, output, target, positionCount](
                             const eacp::ML::Prediction& done)
    {
        predictSeconds = done.predictSeconds;
        finishAsync(positionCount);

        if (done.ok)
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
