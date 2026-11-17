#include "Half.h"

#include <bit>
#include <cstring>

namespace WSP
{
std::uint16_t packHalf(float value)
{
    auto bits = std::uint32_t {};
    std::memcpy(&bits, &value, sizeof(bits));

    const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
    const auto rawExponent = (bits >> 23) & 0xFFu;
    const auto mantissa = bits & 0x7FFFFFu;

    if (rawExponent == 0xFFu)
        return static_cast<std::uint16_t>(sign | 0x7C00u
                                          | (mantissa != 0u ? 0x200u : 0u));

    const auto exponent = static_cast<std::int32_t>(rawExponent) - 127 + 15;

    if (exponent >= 0x1F)
        return static_cast<std::uint16_t>(sign | 0x7C00u);

    // Below the smallest subnormal's halfway point nothing survives the shift.
    if (exponent < -10)
        return sign;

    const auto roundUp = [](std::uint32_t result,
                            std::uint32_t dropped,
                            std::uint32_t half) -> std::uint32_t
    {
        const auto atLeastHalf = (dropped & half) != 0u;
        const auto aboveHalf = (dropped & (half - 1u)) != 0u;

        return result + (atLeastHalf && (aboveHalf || (result & 1u)) ? 1u : 0u);
    };

    if (exponent <= 0)
    {
        const auto shift = static_cast<std::uint32_t>(14 - exponent);
        const auto full = mantissa | 0x800000u;
        const auto dropped = full & ((1u << shift) - 1u);
        const auto rounded = roundUp(full >> shift, dropped, 1u << (shift - 1u));

        return static_cast<std::uint16_t>(sign | rounded);
    }

    const auto packed =
        (static_cast<std::uint32_t>(exponent) << 10) | (mantissa >> 13);
    const auto rounded = roundUp(packed, mantissa & 0x1FFFu, 0x1000u);

    return static_cast<std::uint16_t>(sign | rounded);
}

float halfToFloat(std::uint16_t bits)
{
    const auto sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const auto exponent = static_cast<std::uint32_t>((bits >> 10) & 0x1Fu);
    const auto mantissa = static_cast<std::uint32_t>(bits & 0x03FFu);

    if (exponent == 0x1Fu)
        return std::bit_cast<float>(sign | 0x7F800000u | (mantissa << 13));

    if (exponent == 0)
    {
        constexpr auto smallestSubnormal = 1.0f / (1 << 24);
        const auto magnitude = static_cast<float>(mantissa) * smallestSubnormal;

        return sign != 0 ? -magnitude : magnitude;
    }

    constexpr auto biasDifference = std::uint32_t {127 - 15};
    const auto rebiased = (exponent + biasDifference) << 23;

    return std::bit_cast<float>(sign | rebiased | (mantissa << 13));
}
} // namespace WSP
