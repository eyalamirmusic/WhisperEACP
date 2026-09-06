#pragma once

#include <WhisperEACP/Kernels/Kernels.h>

#include <eacp/Core/App/App.h>
#include <eacp/GPU/GPU.h>

#include <NanoTest/NanoTest.h>

#include <cmath>
#include <random>

namespace WSP
{
// The host-side plumbing every kernel test repeats: floats up, floats back, and
// a spread of inputs worth running a kernel over. Nothing here is a fixture —
// the reference each test asserts against is scalar C++ in the test body, which
// is what makes the same assertion catch a divergence between MSL and HLSL.
inline Vector<float> sized(int elementCount)
{
    auto values = Vector<float>();
    values.resize(elementCount);
    return values;
}

inline eacp::GPU::Buffer storageOf(const Vector<float>& values)
{
    return eacp::GPU::Device::shared().makeBuffer(values.data(),
                                                  (int) sizeof(float)
                                                      * values.size(),
                                                  eacp::GPU::BufferUsage::Storage);
}

inline eacp::GPU::Buffer outputFor(int elementCount)
{
    return eacp::GPU::Device::shared().makeBuffer((int) sizeof(float)
                                                  * elementCount);
}

inline Vector<float> readBack(const eacp::GPU::Buffer& buffer, int elementCount)
{
    auto values = sized(elementCount);
    buffer.read(values.data(), (int) sizeof(float) * elementCount);
    return values;
}

// Values in [-range, range], negatives included, from a generator whose
// sequence the standard specifies rather than an implementation: the same
// numbers reach the GPU and the reference on every machine, so a disagreement
// is the kernel.
inline Vector<float> spreadValues(int count, unsigned seed, float range)
{
    auto engine = std::mt19937 {seed};
    auto values = sized(count);

    for (auto i = 0; i < count; ++i)
        values[i] = range * ((float) (engine() % 2001u) / 1000.f - 1.f);

    return values;
}

inline bool isClose(float actual, double expected, double tolerance)
{
    return std::abs((double) actual - expected)
           <= tolerance * (1.0 + std::abs(expected));
}

template <typename Kernel>
Vector<float> runOverRows(Kernel& kernel,
                          const eacp::GPU::Buffer& output,
                          int rowCount,
                          int outputElements)
{
    kernel.prepare();

    auto commands = eacp::GPU::Device::shared().makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        pass.dispatch(kernel, rowCount);
    }

    commands.commit();
    return readBack(output, outputElements);
}

template <typename Kernel>
Vector<float> runOverGrid(Kernel& kernel,
                          const eacp::GPU::Buffer& output,
                          int width,
                          int height)
{
    kernel.prepare();

    auto commands = eacp::GPU::Device::shared().makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        pass.dispatch(kernel, width, height);
    }

    commands.commit();
    return readBack(output, width * height);
}
} // namespace WSP
