#include "MelTestSupport.h"

using namespace nano;
using namespace eacp::GPU;

namespace
{
constexpr auto groupWidth = WSP::MaxReduceKernel::groupWidth;

int groupsFor(int elementCount)
{
    return (elementCount + groupWidth - 1) / groupWidth;
}

// The rounds the front-end records, run on their own: each pass reduces its
// input to one value per threadgroup, and the next reads what it wrote.
float runMaxReduce(Device& device, const std::vector<float>& values)
{
    const auto source = melTest::upload(device, values);
    const auto first = melTest::allocate(device, groupsFor((int) values.size()));
    const auto second =
        melTest::allocate(device, groupsFor(groupsFor((int) values.size())));

    auto kernel = WSP::MaxReduceKernel {};
    kernel.prepare(device);

    auto commands = device.makeCommandBuffer();

    const auto* input = &source;
    const auto* output = &first;
    auto remaining = (int) values.size();

    while (remaining > 1)
    {
        const auto groups = groupsFor(remaining);

        kernel.values = *input;
        kernel.partials = *output;
        kernel.count = (std::uint32_t) remaining;
        kernel.stride = (std::uint32_t) (groups * groupWidth);

        {
            auto pass = commands.beginCompute();
            pass.dispatch(kernel, groups * groupWidth);
        }

        input = output;
        output = output == &first ? &second : &first;
        remaining = groups;
    }

    commands.commit();

    return melTest::download(*input, 1)[0];
}

std::vector<float> pseudoRandom(int count, unsigned seed)
{
    auto values = std::vector<float>((std::size_t) count);
    auto state = seed;

    for (auto& value: values)
    {
        state = state * 1664525u + 1013904223u;
        value = (float) ((state >> 8) % 1000001u) * 1e-5f - 10.0f;
    }

    return values;
}

std::vector<float>
    runNormalise(Device& device, const std::vector<float>& values, float peak)
{
    const auto valueBuffer = melTest::upload(device, values);
    const auto peakBuffer = melTest::upload(device, std::vector<float> {peak});
    const auto output = melTest::allocate(device, (int) values.size());

    auto kernel = WSP::MelNormaliseKernel {};
    kernel.values = valueBuffer;
    kernel.peak = peakBuffer;
    kernel.normalised = output;
    kernel.prepare(device);

    auto commands = device.makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        pass.dispatch(kernel, (int) values.size());
    }

    commands.commit();

    return melTest::download(output, (int) values.size());
}
} // namespace

// Three rounds at this size, which is the point: the maximum of a spectrogram
// is not something one dispatch can know, and a partial that never reaches the
// last round is the way that goes wrong.
auto tMaxReduceFindsTheGlobalMaximum =
    test("Mel/maxReduceFindsTheGlobalMaximum") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto values = pseudoRandom(240000, 4242u);
    values[(std::size_t) 199999] = 7.5f;

    check(groupsFor(groupsFor((int) values.size())) > 1);
    check(runMaxReduce(device, values) == 7.5f);
};

auto tMaxReduceHandlesAPartialGroup = test("Mel/maxReduceHandlesAPartialGroup") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    for (const auto count: {1, 7, groupWidth, groupWidth + 1, 3 * groupWidth - 5})
    {
        auto values = pseudoRandom(count, 99u);
        values[(std::size_t) (count / 2)] = 3.25f;

        check(runMaxReduce(device, values) == 3.25f);
    }
};

auto tNormaliseShiftsAndScales = test("Mel/normaliseShiftsAndScales") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    const auto values = std::vector<float> {0.5f, -2.0f, -6.0f, -9.0f, -30.0f};
    const auto peak = 0.5f;

    const auto normalised = runNormalise(device, values, peak);

    for (auto i = 0; i < (int) values.size(); ++i)
    {
        const auto floored = std::max((double) values[(std::size_t) i], peak - 8.0);

        check(melTest::close(
            normalised[(std::size_t) i], (floored + 4.0) / 4.0, 1e-6));
    }

    check(melTest::close(normalised[0], (0.5 + 4.0) / 4.0, 1e-6));
    check(melTest::close(normalised[4], (-7.5 + 4.0) / 4.0, 1e-6));
};

// The eight-decade floor is taken against the loudest cell of the whole
// spectrogram, never against a neighbourhood of it: a quiet stretch beside a
// loud one is clamped by what happened elsewhere, and a per-group floor would
// leave it untouched.
auto tNormaliseFloorsAgainstTheGlobalMaximum =
    test("Mel/normaliseFloorsAgainstTheGlobalMaximum") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto count = 5000;

    auto values = std::vector<float>((std::size_t) count, -9.0f);

    for (auto i = 0; i < 64; ++i)
        values[(std::size_t) i] = -1.0f;

    values[(std::size_t) (count - 1)] = 4.0f;

    const auto peak = runMaxReduce(device, values);
    check(peak == 4.0f);

    const auto normalised = runNormalise(device, values, peak);

    check(melTest::close(normalised[0], (-1.0 + 4.0) / 4.0, 1e-6));
    check(melTest::close(normalised[100], (-4.0 + 4.0) / 4.0, 1e-6));
    check(melTest::close(normalised[(std::size_t) (count - 1)], 2.0, 1e-6));
};
