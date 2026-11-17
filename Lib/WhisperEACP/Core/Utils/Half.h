#pragma once

#include <cstdint>

namespace WSP
{
// float32 to IEEE binary16, round to nearest even, overflowing to infinity —
// the same rule Metal's own narrowing follows, so a value packed here is the
// value a shader would have packed. Written out rather than taken from _Float16
// because MSVC has no such type; the loader narrows with it, and the kernel
// tests pack their halves with it so they see exactly what the loader ships.
std::uint16_t packHalf(float value);

// IEEE binary16 bits widened to float32, exactly.
float halfToFloat(std::uint16_t bits);
} // namespace WSP
