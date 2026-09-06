#include "Common.h"

// The toolchain, end to end: a kernel authored as a struct, compiled for
// whichever backend this is, dispatched over a storage buffer, and read back.
// Nothing here is about Whisper — it is the check that eacp's compute path is
// reachable from this project at all, so a failure in a real layer later can be
// read as ours rather than as a broken link line or a missing device.
//
// It is also the shape every layer's test takes: build inputs, dispatch, assert
// against a reference the CPU computed independently.

using namespace nano;
using namespace eacp;
using namespace eacp::GPU;

namespace
{
struct ScaleKernel final : ComputeProgram
{
    ScaleKernel() { compile(); }

    void define() override
    {
        auto i = threadId();
        write(output, i, input[i] * scale);
    }

    Uniform<InputBuffer> input;
    Uniform<OutputBuffer> output;
    Uniform<Float> scale;

    EACP_SHADER(input, output, scale)
};

constexpr auto elementCount = 8;
} // namespace

auto tScaleKernelRuns = test("Compute/scaleKernelRuns") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    float source[elementCount] = {};

    for (auto i = 0; i < elementCount; ++i)
        source[i] = (float) i;

    auto input = device.makeBuffer(source, sizeof(source), BufferUsage::Storage);
    auto output = device.makeBuffer(sizeof(source));

    auto kernel = ScaleKernel {};
    kernel.input = input;
    kernel.output = output;
    kernel.scale = 3.0f;
    kernel.prepare();

    auto commands = device.makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        pass.dispatch(kernel, elementCount);
    }

    commands.commit();

    float result[elementCount] = {};
    output.read(result, sizeof(result));

    for (auto i = 0; i < elementCount; ++i)
        check(result[i] == (float) i * 3.0f);
};

// The property a whole transformer rests on: one compiled program, re-pointed
// at different buffers and re-dispatched. ComputePass::dispatch re-runs the
// resource bind and re-packs the uniform block on every call, so a model's
// thirty-odd blocks need one pipeline per op type rather than one each.
auto tOneProgramManyDispatches = test("Compute/oneProgramManyDispatches") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    float source[elementCount] = {};

    for (auto i = 0; i < elementCount; ++i)
        source[i] = 1.0f;

    auto input = device.makeBuffer(source, sizeof(source), BufferUsage::Storage);
    auto doubled = device.makeBuffer(sizeof(source));
    auto tripled = device.makeBuffer(sizeof(source));

    auto kernel = ScaleKernel {};
    kernel.input = input;
    kernel.prepare();

    auto commands = device.makeCommandBuffer();

    {
        auto pass = commands.beginCompute();

        kernel.output = doubled;
        kernel.scale = 2.0f;
        pass.dispatch(kernel, elementCount);

        kernel.output = tripled;
        kernel.scale = 3.0f;
        pass.dispatch(kernel, elementCount);
    }

    commands.commit();

    float twos[elementCount] = {};
    float threes[elementCount] = {};
    doubled.read(twos, sizeof(twos));
    tripled.read(threes, sizeof(threes));

    for (auto i = 0; i < elementCount; ++i)
    {
        check(twos[i] == 2.0f);
        check(threes[i] == 3.0f);
    }
};
