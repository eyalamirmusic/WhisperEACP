#pragma once

#include "KernelTypes.h"

namespace WSP
{
// The exact GELU, which is what HuggingFace's Whisper is trained and shipped
// with: x * 0.5 * (1 + erf(x / sqrt(2))). Not the tanh form the reference
// implementations settle for, which is an approximation of this one and about
// three orders of magnitude further from it — GeluTests measures both.
//
// erf is eacp's intrinsic. It is a float32 error function rather than the
// libm-grade one the name has on the CPU, since neither shading language has
// erf and the EDSL emits the approximation as a helper, but its error lands
// under what a float can resolve near one.
inline Float exactGelu(const Float& x)
{
    constexpr auto inverseRootTwo = 0.70710678118654752f;

    return 0.5f * x * (1.f + erf(x * inverseRootTwo));
}

// Elementwise and in place, one thread per element: dispatch(kernel,
// elementCount). The activation replaces what it was computed from, since
// nothing in a transformer reads a pre-activation twice.
struct Gelu final : ComputeProgram
{
    Gelu() { compile(); }

    void define() override
    {
        auto at = threadId();
        write(values, at, exactGelu(values[at]));
    }

    Uniform<OutputBuffer> values;

    EACP_SHADER(values)
};
} // namespace WSP
