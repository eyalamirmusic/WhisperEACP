#include "Common.h"

// The GPU plumbing the kernel tests already have — buffers up, floats back, a
// spread of inputs — rather than a second copy of it here. This is the one
// place the two modules meet: a weight loaded by Model/ bound to a kernel from
// Kernels/, which is what says the packed buffer and the kernel that reads it
// agree about what is in it.
#include "../Kernels/Common.h"

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;

namespace
{
// Bit patterns fp16 holds exactly — a normal exponent, both signs, and a
// mantissa walking the whole field. Widening one is lossless, so the reference
// below is the same arithmetic the GPU does and the tolerance is float32
// accumulation's rather than the format's.
std::uint16_t halfPattern(int index)
{
    const auto sign = index % 2 == 0 ? 0x0000u : 0x8000u;
    const auto mantissa = static_cast<unsigned>((index * 137) % 1024);

    return static_cast<std::uint16_t>(sign | 0x3800u | mantissa);
}

Vector<std::uint8_t> packedHalves(int count)
{
    auto bytes = Vector<std::uint8_t> {};
    bytes.resize(count * 2);

    for (auto index = 0; index < count; ++index)
    {
        const auto bits = halfPattern(index);
        std::memcpy(bytes.data() + index * 2, &bits, sizeof(bits));
    }

    return bytes;
}

SafeTensors halfWeightFile(int elementCount)
{
    const auto header = R"({"weight":{"dtype":"F16","shape":[)"
                        + std::to_string(elementCount) + R"(],"data_offsets":[0,)"
                        + std::to_string(elementCount * 2) + R"(]}})";

    return SafeTensors::fromBytes(assemble(header, packedHalves(elementCount)));
}

Vector<float> matMulReference(const Vector<float>& a,
                              const Vector<float>& b,
                              const Vector<float>& bias,
                              int rowCount,
                              int innerCount,
                              int columnCount)
{
    auto expected = sized(rowCount * columnCount);

    for (auto row = 0; row < rowCount; ++row)
    {
        for (auto column = 0; column < columnCount; ++column)
        {
            auto total = 0.0;

            for (auto step = 0; step < innerCount; ++step)
                total += (double) a[row * innerCount + step]
                         * b[step * columnCount + column];

            expected[row * columnCount + column] = (float) (total + bias[column]);
        }
    }

    return expected;
}

// The whole path, from the file's bytes to the numbers the kernel produced: the
// weight goes up packed, `b` is the same tensor widened on the CPU, and the two
// have to agree because fp16 to fp32 is exact either side.
void checkHalfWeightMatMul(int rowCount, int innerCount, int columnCount)
{
    const auto file = halfWeightFile(innerCount * columnCount);
    const auto weight = file.makeBuffer("weight");

    check(weight.isPackedHalf());

    auto a = spreadValues(rowCount * innerCount, 314159u, 2.f);
    auto bias = spreadValues(columnCount, 161803u, 1.f);
    auto b = file.readFloats("weight");

    auto aBuffer = storageOf(a);
    auto biasBuffer = storageOf(bias);
    auto output = outputFor(rowCount * columnCount);

    auto kernel = HalfWeightMatMul {};
    kernel.a = aBuffer;
    kernel.b = weight.buffer;
    kernel.bias = biasBuffer;
    kernel.output = output;
    kernel.innerCount = (unsigned) innerCount;
    kernel.columnCount = (unsigned) columnCount;

    auto result = runOverGrid(kernel, output, columnCount, rowCount);
    auto expected = matMulReference(a, b, bias, rowCount, innerCount, columnCount);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
}
} // namespace

// 21 halves, which is 42 bytes: the weight buffer is a padded one, and the last
// column's thread reads the half in the word the padding completes.
auto tHalfWeightMatMulOddCount = test("Model/HalfWeightMatMul/oddHalfCount") = []
{
    if (!eacp::GPU::Device::shared().isValid())
        return;

    const auto file = halfWeightFile(21);
    check(file.makeBuffer("weight").buffer.size() == 44);

    checkHalfWeightMatMul(5, 7, 3);
};

// An even count, so the blob goes up untouched, over a shape none of whose
// extents is a multiple of the 8 x 8 dispatch group.
auto tHalfWeightMatMulWiderShape = test("Model/HalfWeightMatMul/widerShape") = []
{
    if (!eacp::GPU::Device::shared().isValid())
        return;

    const auto file = halfWeightFile(41 * 18);
    check(file.makeBuffer("weight").buffer.size() == 41 * 18 * 2);

    checkHalfWeightMatMul(13, 41, 18);
};
