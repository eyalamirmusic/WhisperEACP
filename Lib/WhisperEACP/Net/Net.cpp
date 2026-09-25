#include "Net.h"

namespace WSP
{
Shape::Shape(std::initializer_list<int> extentsToUse)
{
    for (auto extent: extentsToUse)
        if (count < maxRank)
            extents[count++] = extent;
}

int Shape::rows() const
{
    auto product = 1;

    for (auto axis = 0; axis + 1 < count; ++axis)
        product *= extents[axis];

    return product;
}

int Shape::columns() const
{
    return count == 0 ? 0 : extents[count - 1];
}

int Shape::elementCount() const
{
    return count == 0 ? 0 : rows() * columns();
}

Tensor::Tensor(Net& ownerToUse, int indexToUse)
    : owner(&ownerToUse)
    , id(indexToUse)
{
    retain();
}

Tensor::Tensor(const Tensor& other)
    : owner(other.owner)
    , id(other.id)
{
    retain();
}

Tensor::Tensor(Tensor&& other) noexcept
    : owner(other.owner)
    , id(other.id)
{
    other.owner = nullptr;
    other.id = -1;
}

Tensor& Tensor::operator=(const Tensor& other)
{
    if (this != &other)
    {
        other.retain();
        release();

        owner = other.owner;
        id = other.id;
    }

    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if (this != &other)
    {
        release();

        owner = other.owner;
        id = other.id;
        other.owner = nullptr;
        other.id = -1;
    }

    return *this;
}

Tensor::~Tensor()
{
    release();
}

void Tensor::retain() const
{
    if (owner != nullptr)
        owner->retain(id);
}

void Tensor::release() const
{
    if (owner != nullptr)
        owner->release(id);
}
} // namespace WSP
