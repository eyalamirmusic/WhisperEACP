#pragma once

#include <WhisperEACP/Core/Core.h>

namespace WSP
{
// The real DFT as a matrix: row 2 * bin holds cos(2 pi bin n / N) over the N
// taps, row 2 * bin + 1 holds -sin of the same, for the N / 2 + 1 bins of a
// real spectrum. Laid out [2 * bins, N] row-major, which is nn.Linear's
// [out, in], so the STFT of every frame at once is one tiled product of the
// [frames, N] windowed frames against it, and the spectrum comes out
// [frames, 2 * bins] with each bin's real and imaginary parts side by side.
//
// Computed in double and stored as float, once per prepare(): 402 x 400 for
// Whisper's 400-point window, 643 kB.
Vector<float> dftBasis(int fftLength);
} // namespace WSP
