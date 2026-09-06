#include "Format.h"

namespace WSP
{
bool canCaptureNatively(const MakeASound::DeviceInfo& device)
{
    return device.hasChannels(true)
           && MakeASound::deviceSupportsSampleRate(device, sampleRate);
}
} // namespace WSP
