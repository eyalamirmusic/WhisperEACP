#include "Window.h"

#include <cmath>
#include <numbers>

namespace WSP
{
float periodicHannValue(int index, int length)
{
    const auto turn = 2.0 * std::numbers::pi * (double) index / (double) length;
    return (float) (0.5 * (1.0 - std::cos(turn)));
}

Vector<float> periodicHannWindow(int length)
{
    auto window = Vector<float>(length);

    for (auto i = 0; i < length; ++i)
        window[i] = periodicHannValue(i, length);

    return window;
}
} // namespace WSP
