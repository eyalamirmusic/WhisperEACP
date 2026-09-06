#include "Common.h"

using namespace nano;
using namespace WSP;
using namespace eacp::GPU;

namespace
{
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

// The shape a bias-free projection binds, since both backends reject a dispatch
// whose shader declares a buffer nothing was bound to.
Vector<float> noBias(int columnCount)
{
    auto values = sized(columnCount);

    for (auto i = 0; i < columnCount; ++i)
        values[i] = 0.f;

    return values;
}

// None of the three equal, none a multiple of the 8 x 8 dispatch group, so a
// kernel that confused a row stride for a column stride or leaned on the grid
// landing exactly on the matrix fails here rather than on the first real model.
constexpr auto rowCount = 5;
constexpr auto innerCount = 7;
constexpr auto columnCount = 3;
} // namespace

auto tMatMulMatchesCpu = test("Kernels/matMulMatchesCpu") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto a = spreadValues(rowCount * innerCount, 314159u, 2.f);
    auto b = spreadValues(innerCount * columnCount, 271828u, 3.f);
    auto bias = spreadValues(columnCount, 161803u, 1.f);

    auto aBuffer = storageOf(a);
    auto bBuffer = storageOf(b);
    auto biasBuffer = storageOf(bias);
    auto output = outputFor(rowCount * columnCount);

    auto kernel = MatMul {};
    kernel.a = aBuffer;
    kernel.b = bBuffer;
    kernel.bias = biasBuffer;
    kernel.output = output;
    kernel.innerCount = (unsigned) innerCount;
    kernel.columnCount = (unsigned) columnCount;

    auto result = runOverGrid(kernel, output, columnCount, rowCount);
    auto expected = matMulReference(a, b, bias, rowCount, innerCount, columnCount);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
};

// A shape wide enough that a thread reaching one element past its row, or an
// accumulator that never got reset between output elements, shows up as a
// number rather than as a crash.
auto tMatMulWiderShapeWithoutBias = test("Kernels/matMulWiderShapeWithoutBias") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto rows = 13;
    constexpr auto inner = 41;
    constexpr auto columns = 17;

    auto a = spreadValues(rows * inner, 4242u, 1.5f);
    auto b = spreadValues(inner * columns, 2424u, 1.5f);
    auto bias = noBias(columns);

    auto aBuffer = storageOf(a);
    auto bBuffer = storageOf(b);
    auto biasBuffer = storageOf(bias);
    auto output = outputFor(rows * columns);

    auto kernel = MatMul {};
    kernel.a = aBuffer;
    kernel.b = bBuffer;
    kernel.bias = biasBuffer;
    kernel.output = output;
    kernel.innerCount = (unsigned) inner;
    kernel.columnCount = (unsigned) columns;

    auto result = runOverGrid(kernel, output, columns, rows);
    auto expected = matMulReference(a, b, bias, rows, inner, columns);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
};

// Multiplying by the identity is the one product whose answer is the input, so
// a transposed index or an off-by-one in either stride is visible by eye.
auto tMatMulByIdentityIsTheInput = test("Kernels/matMulByIdentityIsTheInput") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto rows = 6;
    constexpr auto width = 9;

    auto a = spreadValues(rows * width, 555u, 5.f);
    auto identity = sized(width * width);

    for (auto row = 0; row < width; ++row)
        for (auto column = 0; column < width; ++column)
            identity[row * width + column] = row == column ? 1.f : 0.f;

    auto bias = noBias(width);

    auto aBuffer = storageOf(a);
    auto identityBuffer = storageOf(identity);
    auto biasBuffer = storageOf(bias);
    auto output = outputFor(rows * width);

    auto kernel = MatMul {};
    kernel.a = aBuffer;
    kernel.b = identityBuffer;
    kernel.bias = biasBuffer;
    kernel.output = output;
    kernel.innerCount = (unsigned) width;
    kernel.columnCount = (unsigned) width;

    auto result = runOverGrid(kernel, output, width, rows);

    for (auto i = 0; i < result.size(); ++i)
        check(result[i] == a[i]);
};

// One pipeline, two shapes, two dispatches — what a model with thirty blocks of
// differing widths needs, and what a kernel with a shape compiled into it
// cannot do.
auto tMatMulOneProgramTwoShapes = test("Kernels/matMulOneProgramTwoShapes") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto secondRows = 4;
    constexpr auto secondInner = 11;
    constexpr auto secondColumns = 6;

    auto firstA = spreadValues(rowCount * innerCount, 1001u, 2.f);
    auto firstB = spreadValues(innerCount * columnCount, 1002u, 2.f);
    auto firstBias = spreadValues(columnCount, 1003u, 1.f);

    auto secondA = spreadValues(secondRows * secondInner, 2001u, 2.f);
    auto secondB = spreadValues(secondInner * secondColumns, 2002u, 2.f);
    auto secondBias = spreadValues(secondColumns, 2003u, 1.f);

    auto firstABuffer = storageOf(firstA);
    auto firstBBuffer = storageOf(firstB);
    auto firstBiasBuffer = storageOf(firstBias);
    auto firstOutput = outputFor(rowCount * columnCount);

    auto secondABuffer = storageOf(secondA);
    auto secondBBuffer = storageOf(secondB);
    auto secondBiasBuffer = storageOf(secondBias);
    auto secondOutput = outputFor(secondRows * secondColumns);

    auto kernel = MatMul {};
    kernel.prepare();

    auto commands = device.makeCommandBuffer();

    {
        auto pass = commands.beginCompute();

        kernel.a = firstABuffer;
        kernel.b = firstBBuffer;
        kernel.bias = firstBiasBuffer;
        kernel.output = firstOutput;
        kernel.innerCount = (unsigned) innerCount;
        kernel.columnCount = (unsigned) columnCount;
        pass.dispatch(kernel, columnCount, rowCount);

        kernel.a = secondABuffer;
        kernel.b = secondBBuffer;
        kernel.bias = secondBiasBuffer;
        kernel.output = secondOutput;
        kernel.innerCount = (unsigned) secondInner;
        kernel.columnCount = (unsigned) secondColumns;
        pass.dispatch(kernel, secondColumns, secondRows);
    }

    commands.commit();

    auto firstResult = readBack(firstOutput, rowCount * columnCount);
    auto secondResult = readBack(secondOutput, secondRows * secondColumns);

    auto firstExpected = matMulReference(
        firstA, firstB, firstBias, rowCount, innerCount, columnCount);

    auto secondExpected = matMulReference(
        secondA, secondB, secondBias, secondRows, secondInner, secondColumns);

    for (auto i = 0; i < firstResult.size(); ++i)
        check(isClose(firstResult[i], firstExpected[i], 1e-5));

    for (auto i = 0; i < secondResult.size(); ++i)
        check(isClose(secondResult[i], secondExpected[i], 1e-5));
};
