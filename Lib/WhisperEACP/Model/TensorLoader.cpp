#include "TensorLoader.h"

#include <utility>

namespace WSP
{
namespace
{
std::string shapeText(const TensorInfo& tensor)
{
    auto text = std::string {"["};

    for (auto axis = 0; axis < tensor.rank(); ++axis)
        text += (axis == 0 ? "" : ", ") + std::to_string(tensor.dimension(axis));

    return text + "]";
}
} // namespace

const TensorInfo& TensorLoader::require(const std::string& name) const
{
    if (const auto* found = file.find(name))
        return *found;

    throw ModelError {"the " + std::string {component} + " needs a tensor named '"
                      + name + "', and the file has none"};
}

void TensorLoader::checkShape(const TensorInfo& tensor, TensorShape expected) const
{
    auto wrong = [&]
    {
        auto text = std::string {"["};
        auto axis = 0;

        for (auto extent: expected)
            text += (axis++ == 0 ? "" : ", ") + std::to_string(extent);

        throw ModelError {"tensor '" + tensor.name + "' is " + shapeText(tensor)
                          + ", and the " + std::string {component}
                          + " was built for " + text + "]"};
    };

    if (tensor.rank() != (int) expected.size())
        wrong();

    auto axis = 0;

    for (auto extent: expected)
        if (tensor.dimension(axis++) != extent)
            wrong();
}

void TensorLoader::checkPositionalWidth(const TensorInfo& tensor, int width) const
{
    if (tensor.rank() != 2 || tensor.dimension(1) != width)
        throw ModelError {"tensor '" + tensor.name + "' is " + shapeText(tensor)
                          + ", and the " + std::string {component}
                          + " was built for [positions, " + std::to_string(width)
                          + "]"};
}

TensorBuffer TensorLoader::loadFloatTensor(const std::string& name,
                                           TensorShape expected) const
{
    checkShape(require(name), expected);

    return file.makeFloatBuffer(name);
}

TensorBuffer TensorLoader::loadProjectionWeight(const std::string& name,
                                                TensorShape expected,
                                                WeightPacking packing) const
{
    checkShape(require(name), expected);

    if (packing == WeightPacking::ExactHalf)
        if (auto packed = file.makeExactHalfBuffer(name))
            return std::move(*packed);

    return file.makeBuffer(name);
}
} // namespace WSP
