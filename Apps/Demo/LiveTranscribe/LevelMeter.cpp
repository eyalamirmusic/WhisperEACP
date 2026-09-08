#include "LevelMeter.h"

#include <algorithm>
#include <cmath>

namespace LiveTranscribe
{
namespace
{
constexpr auto floorDb = -60.f;

// A second at the timer's 30 Hz, and then a twenty-fifth of the bar a tick,
// which reads as a fall rather than as a disappearance.
constexpr auto peakHoldTicks = 30;
constexpr auto peakFallPerTick = 0.04f;

constexpr auto peakTickWidth = 2.f;

// Half a point at any width a meter is given, which is the resolution a repaint
// can actually show.
bool differsOnScreen(float a, float b)
{
    return std::abs(a - b) > 0.002f;
}

UI::Gradient meterGradient(const UI::Rect& bounds)
{
    auto gradient = UI::Gradient {};

    gradient.start = {bounds.x, bounds.y};
    gradient.end = {bounds.right(), bounds.y};

    gradient.stops.add(UI::GradientStop {{0.30f, 0.80f, 0.48f, 1.f}, 0.f});
    gradient.stops.add(UI::GradientStop {{0.88f, 0.78f, 0.32f, 1.f}, 0.72f});
    gradient.stops.add(UI::GradientStop {{0.94f, 0.36f, 0.32f, 1.f}, 1.f});

    return gradient;
}
} // namespace

float meterPosition(float linear)
{
    if (linear <= 0.f)
        return 0.f;

    const auto decibels = 20.f * std::log10(linear);

    return std::clamp((decibels - floorDb) / -floorDb, 0.f, 1.f);
}

void LevelMeter::setLevel(const WSP::CaptureLevel& level)
{
    const auto previousFill = fillPosition;
    const auto previousPeak = peakPosition;

    fillPosition = meterPosition(level.rms);
    advancePeak(meterPosition(level.peak));

    if (differsOnScreen(fillPosition, previousFill)
        || differsOnScreen(peakPosition, previousPeak))
        repaint();
}

void LevelMeter::advancePeak(float peak)
{
    if (peak >= peakPosition)
    {
        peakPosition = peak;
        ticksHoldingPeak = peakHoldTicks;

        return;
    }

    if (ticksHoldingPeak > 0)
    {
        --ticksHoldingPeak;
        return;
    }

    peakPosition = std::max(peak, peakPosition - peakFallPerTick);
}

void LevelMeter::paint(UI::Graphics& g)
{
    const auto& theme = UI::defaultTheme();
    const auto bounds = getLocalBounds();
    const auto corner = std::min(bounds.h * 0.5f, 4.f);

    g.setColour(theme.panel);
    g.fillRoundedRect(bounds, corner);

    if (fillPosition > 0.f)
    {
        g.setGradient(meterGradient(bounds));
        g.fillRoundedRect(bounds.withWidth(bounds.w * fillPosition), corner);
        g.clearGradient();
    }

    if (peakPosition > 0.f)
    {
        const auto left =
            std::min(bounds.w * peakPosition, bounds.w - peakTickWidth);

        g.setColour(theme.text);
        g.fillRect({bounds.x + left, bounds.y, peakTickWidth, bounds.h});
    }

    g.setColour(theme.outline);
    g.drawRoundedRect(bounds, corner);
}
} // namespace LiveTranscribe
