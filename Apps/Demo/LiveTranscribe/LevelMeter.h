#pragma once

#include <WhisperEACP/Audio/Capture.h>

#include <eacp/UI/UI.h>

namespace LiveTranscribe
{
namespace UI = eacp::UI;

// What the microphone is hearing: the RMS of the last block as a fill that runs
// green through yellow to red across the bar, and the peak as a tick that holds
// for about a second and then falls back to it.
//
// Both are placed on a dB scale rather than a linear one, since a linear meter
// spends nine tenths of its width on the last 20 dB and shows speech as a
// twitch at the left edge.
class LevelMeter final : public UI::Component
{
public:
    void setLevel(const WSP::CaptureLevel& level);

    void paint(UI::Graphics& g) override;

private:
    void advancePeak(float peak);

    float fillPosition = 0.f;
    float peakPosition = 0.f;
    int ticksHoldingPeak = 0;
};

// Where a linear amplitude sits on the bar: 0 at -60 dBFS and below, 1 at full
// scale.
float meterPosition(float linear);
} // namespace LiveTranscribe
