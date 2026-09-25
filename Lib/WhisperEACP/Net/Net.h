#pragma once

#include <WhisperEACP/Model/SafeTensors.h>

#include <initializer_list>

namespace WSP
{
// The net's sequence written once, against whichever backend runs it. A body
// in Encoder.cpp or Decoder.cpp calls these; KernelNet dispatches each as the
// compute kernels the tree is written out of, and a graph backend records each
// as an op of a program compiled ahead of time.
//
// The ops sit above matmul and softmax on purpose: the kernel backend folds
// the softmax into the attention apply and the GELU into a projection's store,
// and a graph backend lowers attention to one fused op. Ops at the matmul level
// would take that freedom from both.

enum class Activation
{
    none,
    gelu
};

enum class DType
{
    float32,
    float16,
    int32
};

using Weight = TensorBuffer;

class Shape
{
public:
    static constexpr auto maxRank = 4;

    Shape() = default;
    Shape(std::initializer_list<int> extentsToUse);

    int rank() const { return count; }
    int operator[](int axis) const { return extents[axis]; }

    int rows() const;
    int columns() const;
    int elementCount() const;

    friend bool operator==(const Shape&, const Shape&) = default;

private:
    Array<int, maxRank> extents {};
    int count = 0;
};

// Storage outside the net: the mel, the encoder rows, the logits, the token
// slots. capacity is the whole of what range holds, row-major, when that is
// more than the shape a run reads out of it — the mel's 3000 frames a shorter
// context reads the first few of, or the token slots a step reads one of.
struct Binding
{
    eacp::GPU::BufferRange range;
    Shape capacity;
};

// Rows a sequence keeps between recordings, which a projection appends to in
// place: a decoder's keys and values. rows is how many are filled.
struct Cache
{
    int id = -1;
    int rows = 0;

    void clear() { rows = 0; }
};

class Net;

// A value inside one recording. Copies share it, and the backend reclaims what
// holds it once the last copy is gone, so a body scopes its intermediates the
// way it scopes any other value.
class Tensor
{
public:
    Tensor() = default;
    Tensor(Net& ownerToUse, int indexToUse);

    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    bool isValid() const { return owner != nullptr; }
    int index() const { return id; }

private:
    void retain() const;
    void release() const;

    Net* owner = nullptr;
    int id = -1;
};

class Net
{
public:
    virtual ~Net() = default;

    virtual Tensor input(const Binding& source, const Shape& shape, DType type) = 0;
    virtual void output(const Tensor& value, const Binding& target) = 0;

    virtual Tensor rows(const Weight& table, int first, int count) = 0;
    virtual Tensor transpose(const Tensor& matrix) = 0;
    virtual Tensor cached(const Cache& cache) = 0;

    virtual Tensor conv1d(const Tensor& frames,
                          const Weight& weight,
                          const Weight& bias,
                          int stride,
                          int padding,
                          Activation activation) = 0;

    virtual Tensor linear(const Tensor& input,
                          const Weight& weight,
                          const Weight* bias,
                          Activation activation) = 0;

    // The residual: the projection of input added into stream, which is
    // consumed, and what comes back is the sum.
    virtual Tensor linearAdd(const Tensor& input,
                             const Weight& weight,
                             const Weight* bias,
                             const Tensor& stream) = 0;

    virtual Tensor add(const Tensor& stream, const Tensor& addend) = 0;

    virtual Tensor
        layerNorm(const Tensor& input, const Weight& weight, const Weight& bias) = 0;

    virtual Tensor attention(const Tensor& queries,
                             const Tensor& keys,
                             const Tensor& values,
                             int heads,
                             bool causal) = 0;

    virtual Tensor embed(const Tensor& tokens,
                         const Weight& tokenTable,
                         const Weight& positionTable,
                         int firstPosition) = 0;

    // A projection written into the rows the cache holds next, and the cache
    // with them in it.
    virtual Tensor appendLinear(Cache& cache,
                                const Tensor& input,
                                const Weight& weight,
                                const Weight* bias) = 0;

    virtual Cache makeCache(int capacityRows, int width) = 0;

protected:
    friend class Tensor;

    virtual void retain(int index) = 0;
    virtual void release(int index) = 0;
};
} // namespace WSP
