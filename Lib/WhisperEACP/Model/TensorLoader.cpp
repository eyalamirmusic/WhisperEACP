#include "TensorLoader.h"

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

void TensorLoader::rejectPackedHalf(const TensorBuffer& loaded,
                                    const std::string& name,
                                    std::string_view reader) const
{
    if (loaded.isPackedHalf())
        throw ModelError {"tensor '" + name + "' is fp16, and this "
                          + std::string {component} + " binds it to "
                          + std::string {reader}
                          + ", which has no packed-half read: ship the tensor "
                            "as F32"};
}

TensorBuffer TensorLoader::loadFloatTensor(const std::string& name,
                                           TensorShape expected,
                                           std::string_view reader) const
{
    checkShape(require(name), expected);
    auto loaded = file.makeBuffer(name);
    rejectPackedHalf(loaded, name, reader);

    return loaded;
}

TensorBuffer TensorLoader::loadProjectionWeight(const std::string& name,
                                                TensorShape expected) const
{
    checkShape(require(name), expected);
    return file.makeBuffer(name);
}
} // namespace WSP
