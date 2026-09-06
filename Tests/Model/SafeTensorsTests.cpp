#include "Common.h"

#include <eacp/GPU/GPU.h>

#include <cmath>
#include <cstdint>

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;

namespace
{
const auto singleTensorHeader = std::string {
    R"({"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}})"};

Vector<std::uint8_t> singleTensorFile()
{
    return assemble(singleTensorHeader,
                    toBytes<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

SafeTensors parseHeaderOnly(const std::string& header)
{
    return SafeTensors::fromBytes(assemble(header));
}
} // namespace

auto tSingleTensor = test("Model/SafeTensors/singleTensor") = []
{
    const auto file = SafeTensors::fromBytes(singleTensorFile());

    check(file.tensors().size() == 1);
    check(file.contains("weight"));
    check(!file.contains("bias"));

    const auto& weight = file.info("weight");
    check(weight.name == "weight");
    check(weight.type == TensorType::F32);
    check(weight.rank() == 2);
    check(weight.dimension(0) == 2);
    check(weight.dimension(1) == 3);
    check(weight.elementCount() == 6);
    check(weight.byteCount == 24);
    check(file.rawBytes(weight).size() == 24);

    const auto values = file.readFloats("weight");
    check(values.size() == 6);

    for (auto index = 0; index < values.size(); ++index)
        check(nearlyEqual(values[index], static_cast<float>(index + 1)));
};

auto tReadIntoDestination = test("Model/SafeTensors/readIntoDestination") = []
{
    const auto file = SafeTensors::fromBytes(singleTensorFile());

    auto destination = Vector<float> {};
    destination.resize(6);
    file.readFloats("weight", destination);

    check(nearlyEqual(destination[5], 6.0f));
    check(throwsModelError(
        [&]
        {
            auto tooSmall = Vector<float> {};
            tooSmall.resize(5);
            file.readFloats("weight", tooSmall);
        }));
};

auto tMultipleTensorsAndMetadata =
    test("Model/SafeTensors/multipleTensorsAndMetadata") = []
{
    const auto header =
        std::string {R"({"__metadata__":{"format":"pt"},)"
                     R"("b":{"dtype":"F32","shape":[2],"data_offsets":[8,16]},)"
                     R"("a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})"};

    const auto file = SafeTensors::fromBytes(
        assemble(header, toBytes<float>({1.0f, 2.0f, 3.0f, 4.0f})));

    check(file.tensors().size() == 2);
    check(file.metadata().size() == 1);
    check(file.metadata().at("format") == "pt");
    check(!file.contains("__metadata__"));

    const auto names = file.names();
    check(names.size() == 2);
    check(names[0] == "a");
    check(names[1] == "b");

    check(nearlyEqual(file.readFloats("a")[0], 1.0f));
    check(nearlyEqual(file.readFloats("b")[0], 3.0f));
};

auto tEmptyTensorIsLegal = test("Model/SafeTensors/emptyTensorIsLegal") = []
{
    const auto file = parseHeaderOnly(
        R"({"empty":{"dtype":"F32","shape":[0,4],"data_offsets":[0,0]}})");

    check(file.info("empty").elementCount() == 0);
    check(file.readFloats("empty").empty());
};

// The widening reference is IEEE 754 binary16, not our own implementation: each
// pattern below is the value the standard defines for those sixteen bits.
auto tHalfWidensToFloat = test("Model/SafeTensors/halfWidensToFloat") = []
{
    const auto bits = toBytes<std::uint16_t>(
        {0x0000, 0x8000, 0x3C00, 0xC000, 0x3555, 0x0001, 0x03FF, 0x7BFF});

    const auto file = SafeTensors::fromBytes(assemble(
        R"({"w":{"dtype":"F16","shape":[8],"data_offsets":[0,16]}})", bits));

    const auto values = file.readFloats("w");

    check(values[0] == 0.0f);
    check(values[1] == 0.0f);
    check(std::signbit(values[1]));
    check(values[2] == 1.0f);
    check(values[3] == -2.0f);
    check(nearlyEqual(values[4], 0.333251953125f, 1.0e-9f));
    check(nearlyEqual(values[5], 5.9604644775390625e-8f, 1.0e-12f));
    check(nearlyEqual(values[6], 6.0975551605224609e-5f, 1.0e-10f));
    check(values[7] == 65504.0f);
};

auto tHalfInfinitiesAndNaN = test("Model/SafeTensors/halfInfinitiesAndNaN") = []
{
    const auto bits = toBytes<std::uint16_t>({0x7C00, 0xFC00, 0x7E00});

    const auto file = SafeTensors::fromBytes(
        assemble(R"({"w":{"dtype":"F16","shape":[3],"data_offsets":[0,6]}})", bits));

    const auto values = file.readFloats("w");

    check(std::isinf(values[0]) && values[0] > 0.0f);
    check(std::isinf(values[1]) && values[1] < 0.0f);
    check(std::isnan(values[2]));
};

auto tBfloatWidensToFloat = test("Model/SafeTensors/bfloatWidensToFloat") = []
{
    const auto bits =
        toBytes<std::uint16_t>({0x0000, 0x3F80, 0xC000, 0x4049, 0x7F80});

    const auto file = SafeTensors::fromBytes(assemble(
        R"({"w":{"dtype":"BF16","shape":[5],"data_offsets":[0,10]}})", bits));

    const auto values = file.readFloats("w");

    check(values[0] == 0.0f);
    check(values[1] == 1.0f);
    check(values[2] == -2.0f);
    check(nearlyEqual(values[3], 3.140625f, 1.0e-7f));
    check(std::isinf(values[4]) && values[4] > 0.0f);
};

auto tDoubleNarrowsToFloat = test("Model/SafeTensors/doubleNarrowsToFloat") = []
{
    const auto file = SafeTensors::fromBytes(
        assemble(R"({"w":{"dtype":"F64","shape":[2],"data_offsets":[0,16]}})",
                 toBytes<double>({0.5, -0.25})));

    const auto values = file.readFloats("w");

    check(values[0] == 0.5f);
    check(values[1] == -0.25f);
};

auto tIntegerTensorsAreNotFloats =
    test("Model/SafeTensors/integerTensorsAreNotFloats") = []
{
    const auto file = SafeTensors::fromBytes(
        assemble(R"({"w":{"dtype":"I32","shape":[2],"data_offsets":[0,8]}})",
                 toBytes<std::int32_t>({1, 2})));

    check(file.info("w").type == TensorType::I32);
    check(file.rawBytes("w").size() == 8);
    check(throwsModelError([&] { return file.readFloats("w"); }));
};

auto tUnknownTensorLookup = test("Model/SafeTensors/unknownTensorLookup") = []
{
    const auto file = SafeTensors::fromBytes(singleTensorFile());

    check(file.find("nope") == nullptr);
    check(throwsModelError([&] { return file.info("nope"); }));
    check(throwsModelError([&] { return file.rawBytes("nope"); }));
};

auto tShorterThanLengthPrefix =
    test("Model/SafeTensors/shorterThanLengthPrefix") = []
{
    auto truncated = Vector<std::uint8_t> {};
    truncated.resize(7);

    check(throwsModelError([&] { return SafeTensors::fromBytes(truncated); }));
};

auto tHeaderLengthPastEnd = test("Model/SafeTensors/headerLengthPastEnd") = []
{
    const auto header = std::string {R"({})"};

    check(throwsModelError(
        [&]
        {
            return SafeTensors::fromBytes(
                assembleWithLength(header.size() + 1, header, {}));
        }));

    check(throwsModelError(
        [&]
        {
            return SafeTensors::fromBytes(
                assembleWithLength(~std::uint64_t {}, header, {}));
        }));
};

auto tHeaderIsNotJson = test("Model/SafeTensors/headerIsNotJson") = []
{
    check(throwsModelError([] { return parseHeaderOnly("{not json"); }));
    check(throwsModelError([] { return parseHeaderOnly("[1,2,3]"); }));
    check(throwsModelError([] { return parseHeaderOnly(""); }));
};

auto tEntryIsNotAnObject = test("Model/SafeTensors/entryIsNotAnObject") = []
{ check(throwsModelError([] { return parseHeaderOnly(R"({"w":42})"); })); };

auto tMissingFields = test("Model/SafeTensors/missingFields") = []
{
    check(throwsModelError(
        []
        { return parseHeaderOnly(R"({"w":{"shape":[1],"data_offsets":[0,4]}})"); }));

    check(throwsModelError(
        []
        {
            return parseHeaderOnly(R"({"w":{"dtype":"F32","data_offsets":[0,4]}})");
        }));

    check(throwsModelError(
        [] { return parseHeaderOnly(R"({"w":{"dtype":"F32","shape":[1]}})"); }));
};

auto tUnknownDtype = test("Model/SafeTensors/unknownDtype") = []
{
    check(throwsModelError(
        []
        {
            return parseHeaderOnly(
                R"({"w":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]}})");
        }));

    check(throwsModelError(
        []
        {
            return parseHeaderOnly(
                R"({"w":{"dtype":32,"shape":[1],"data_offsets":[0,4]}})");
        }));
};

auto tMalformedOffsets = test("Model/SafeTensors/malformedOffsets") = []
{
    const auto blob = toBytes<float>({1.0f, 2.0f});

    check(throwsModelError(
        [&]
        {
            return SafeTensors::fromBytes(assemble(
                R"({"w":{"dtype":"F32","shape":[1],"data_offsets":[4]}})", blob));
        }));

    check(throwsModelError(
        [&]
        {
            return SafeTensors::fromBytes(assemble(
                R"({"w":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}})", blob));
        }));

    check(throwsModelError(
        [&]
        {
            return SafeTensors::fromBytes(assemble(
                R"({"w":{"dtype":"F32","shape":[1],"data_offsets":[-4,0]}})", blob));
        }));

    check(throwsModelError(
        [&]
        {
            return SafeTensors::fromBytes(assemble(
                R"({"w":{"dtype":"F32","shape":[1],"data_offsets":[0,1.5]}})",
                blob));
        }));
};

auto tRangeLeavesTheBlob = test("Model/SafeTensors/rangeLeavesTheBlob") = []
{
    check(throwsModelError(
        [&]
        {
            return SafeTensors::fromBytes(
                assemble(R"({"w":{"dtype":"F32","shape":[4],)"
                         R"("data_offsets":[0,16]}})",
                         toBytes<float>({1.0f, 2.0f})));
        }));
};

auto tShapeDisagreesWithRange =
    test("Model/SafeTensors/shapeDisagreesWithRange") = []
{
    const auto blob = toBytes<float>({1.0f, 2.0f});

    check(throwsModelError(
        [&]
        {
            return SafeTensors::fromBytes(assemble(
                R"({"w":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}})", blob));
        }));

    check(throwsModelError(
        [&]
        {
            return SafeTensors::fromBytes(assemble(
                R"({"w":{"dtype":"F16","shape":[2],"data_offsets":[0,8]}})", blob));
        }));
};

auto tNegativeDimension = test("Model/SafeTensors/negativeDimension") = []
{
    check(throwsModelError(
        []
        {
            return parseHeaderOnly(
                R"({"w":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}})");
        }));

    check(throwsModelError(
        []
        {
            return parseHeaderOnly(
                R"({"w":{"dtype":"F32","shape":"2","data_offsets":[0,0]}})");
        }));
};

auto tMetadataMustBeStrings = test("Model/SafeTensors/metadataMustBeStrings") = []
{
    check(throwsModelError(
        [] { return parseHeaderOnly(R"({"__metadata__":{"format":7}})"); }));

    check(throwsModelError([] { return parseHeaderOnly(R"({"__metadata__":7})"); }));
};

auto tMissingFileIsAnError = test("Model/SafeTensors/missingFileIsAnError") = []
{
    check(throwsModelError(
        [] { return SafeTensors::fromFile("no/such/model.safetensors"); }));
};

auto tTensorToGpuBuffer = test("Model/SafeTensors/tensorToGpuBuffer") = []
{
    auto& device = eacp::GPU::Device::shared();

    if (!device.isValid())
        return;

    const auto file = SafeTensors::fromBytes(
        assemble(R"({"w":{"dtype":"F16","shape":[4],"data_offsets":[0,8]}})",
                 toBytes<std::uint16_t>({0x3C00, 0xC000, 0x0000, 0x4000})));

    const auto buffer = file.makeBuffer("w");
    check(buffer.isValid());
    check(buffer.size() == 16);

    auto readBack = Vector<float> {};
    readBack.resize(4);
    buffer.read(readBack.data(), buffer.size());

    check(readBack[0] == 1.0f);
    check(readBack[1] == -2.0f);
    check(readBack[2] == 0.0f);
    check(readBack[3] == 2.0f);
};

// The F32 path is the other branch of makeBuffer: the blob is already the
// layout a kernel binds, so it goes to the device without a widened copy.
auto tFloatTensorToGpuBuffer = test("Model/SafeTensors/floatTensorToGpuBuffer") = []
{
    if (!eacp::GPU::Device::shared().isValid())
        return;

    const auto file = SafeTensors::fromBytes(singleTensorFile());
    const auto buffer = file.makeBuffer("weight");

    check(buffer.isValid());
    check(buffer.size() == 24);

    auto readBack = Vector<float> {};
    readBack.resize(6);
    buffer.read(readBack.data(), buffer.size());

    for (auto index = 0; index < readBack.size(); ++index)
        check(readBack[index] == static_cast<float>(index + 1));
};
