#include "Argmax.h"

#include <algorithm>
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
constexpr auto elementsPerThread = 8;
} // namespace

int Argmax::groupsPerRow(int rowLength)
{
    const auto perGroup = ComputeProgram::groupWidth * elementsPerThread;
    return std::max(1, (rowLength + perGroup - 1) / perGroup);
}

void Argmax::prepare(Device& device, int rowCount, int rowLength)
{
    partialStage.prepare(device);
    finalStage.prepare(device);

    const auto candidates = rowCount * groupsPerRow(rowLength);

    partialValues.emplace(
        device.makeBuffer(candidates * (int) sizeof(float), BufferUsage::Storage));
    partialIndices.emplace(device.makeBuffer(
        candidates * (int) sizeof(std::uint32_t), BufferUsage::Storage));

    rowCapacity = rowCount;
    lengthCapacity = rowLength;
}

void Argmax::prepare(int rowCount, int rowLength)
{
    prepare(Device::shared(), rowCount, rowLength);
}

void Argmax::encode(ComputePass& pass,
                    const BufferRange& logits,
                    const Buffer& mask,
                    const BufferRange& indices,
                    int rowCount,
                    int rowLength)
{
    if (rowCount > rowCapacity || rowLength > lengthCapacity)
        throw std::invalid_argument {
            "Argmax was prepared for " + std::to_string(rowCapacity) + " rows of "
            + std::to_string(lengthCapacity) + " and asked to scan "
            + std::to_string(rowCount) + " of " + std::to_string(rowLength)};

    const auto groups = groupsPerRow(rowLength);

    partialStage.logits = logits;
    partialStage.mask = mask;
    partialStage.partialValues = *partialValues;
    partialStage.partialIndices = *partialIndices;
    partialStage.rowLength = (std::uint32_t) rowLength;
    partialStage.groupsPerRow = (std::uint32_t) groups;

    partialStage.dispatchRows(pass, rowCount * groups);

    finalStage.partialValues = *partialValues;
    finalStage.partialIndices = *partialIndices;
    finalStage.result = indices;
    finalStage.partialsPerRow = (std::uint32_t) groups;

    finalStage.dispatchRows(pass, rowCount);
}
} // namespace WSP
