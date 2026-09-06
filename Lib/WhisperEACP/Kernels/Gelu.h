#pragma once

#include "KernelTypes.h"

namespace WSP
{
// The EDSL has no erf and no tanh — its intrinsic set is abs, atan2, ceil,
// clamp, cos, exp, floor, fract, log, log2, max, min, mix, pow, rsqrt, sign,
// sin, smoothstep, sqrt and step — so the error function GELU is defined
// through is built out of exp instead: Abramowitz & Stegun 7.1.26, a
// fifth-order polynomial in 1 / (1 + 0.3275911 |x|) against a Gaussian. Its
// absolute error stays under 1.5e-7 across the whole line, which is float32's
// own resolution, and about a thousandth of what the tanh approximation
// whisper.cpp settles for costs. GeluTests measures both.
//
// The gap belongs in eacp rather than here: an erf intrinsic — or a tanh one,
// which the other approximation would need — makes this a single call, and both
// backends already have both functions.
inline Float approximateErf(const Float& x)
{
    auto magnitude = abs(x);
    auto t = 1.f / (1.f + 0.3275911f * magnitude);

    auto series =
        t
        * (0.254829592f
           + t
                 * (-0.284496736f
                    + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));

    return sign(x) * (1.f - series * exp(-(magnitude * magnitude)));
}

// The exact GELU, which is what HuggingFace's Whisper is trained and shipped
// with: x * 0.5 * (1 + erf(x / sqrt(2))).
inline Float exactGelu(const Float& x)
{
    constexpr auto inverseRootTwo = 0.70710678118654752f;

    return 0.5f * x * (1.f + approximateErf(x * inverseRootTwo));
}

// Elementwise, one thread per element: dispatch(kernel, elementCount).
struct Gelu final : ComputeProgram
{
    Gelu() { compile(); }

    void define() override
    {
        auto at = threadId();
        write(output, at, exactGelu(input[at]));
    }

    Uniform<InputBuffer> input;
    Uniform<OutputBuffer> output;

    EACP_SHADER(input, output)
};
} // namespace WSP
