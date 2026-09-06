#include "TensorType.h"

#include <array>

namespace WSP
{
namespace
{
struct TensorTypeEntry
{
    TensorType type;
    std::string_view name;
    int bytes;
    bool floatingPoint;
};

constexpr auto tensorTypes =
    std::array {TensorTypeEntry {TensorType::Bool, "BOOL", 1, false},
                TensorTypeEntry {TensorType::U8, "U8", 1, false},
                TensorTypeEntry {TensorType::I8, "I8", 1, false},
                TensorTypeEntry {TensorType::U16, "U16", 2, false},
                TensorTypeEntry {TensorType::I16, "I16", 2, false},
                TensorTypeEntry {TensorType::F16, "F16", 2, true},
                TensorTypeEntry {TensorType::BF16, "BF16", 2, true},
                TensorTypeEntry {TensorType::U32, "U32", 4, false},
                TensorTypeEntry {TensorType::I32, "I32", 4, false},
                TensorTypeEntry {TensorType::F32, "F32", 4, true},
                TensorTypeEntry {TensorType::U64, "U64", 8, false},
                TensorTypeEntry {TensorType::I64, "I64", 8, false},
                TensorTypeEntry {TensorType::F64, "F64", 8, true}};

constexpr bool tensorTypesAreInEnumOrder()
{
    for (auto index = std::size_t {}; index < tensorTypes.size(); ++index)
        if (tensorTypes[index].type != static_cast<TensorType>(index))
            return false;

    return true;
}

static_assert(tensorTypesAreInEnumOrder());

const TensorTypeEntry& entryFor(TensorType type)
{
    return tensorTypes[static_cast<std::size_t>(type)];
}
} // namespace

int bytesPerElement(TensorType type)
{
    return entryFor(type).bytes;
}

bool isFloatingPoint(TensorType type)
{
    return entryFor(type).floatingPoint;
}

std::string_view tensorTypeName(TensorType type)
{
    return entryFor(type).name;
}

std::optional<TensorType> findTensorType(std::string_view name)
{
    for (const auto& entry: tensorTypes)
        if (entry.name == name)
            return entry.type;

    return {};
}
} // namespace WSP
