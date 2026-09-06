#pragma once

#include <WhisperEACP/Core/Core.h>

namespace WSP
{
// The periodic Hann window, 0.5 * (1 - cos(2*pi*n / length)) — the variant
// torch.hann_window and WhisperFeatureExtractor use. The symmetric one divides
// by length - 1 instead and is a different window: close enough everywhere to
// pass an eyeball and wrong enough to move the last decimal of every mel bin.
float periodicHannValue(int index, int length);

Vector<float> periodicHannWindow(int length);
} // namespace WSP
