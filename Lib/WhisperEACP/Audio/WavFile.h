#pragma once

#include <WhisperEACP/Core/Core.h>

#include <filesystem>
#include <stdexcept>

namespace WSP
{
// Every way a WAV file can fail to be the audio Whisper's front-end accepts:
// not a RIFF/WAVE file at all, a truncated or reversed chunk, an encoding this
// reader has no decode for, and — the one that is a policy rather than a
// defect — a sample rate that is not 16 kHz.
//
// Its own type rather than ModelError, because Audio/ sits below Model/ and a
// file of samples is not a file of weights.
class WavError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// A RIFF/WAVE file read as the mono 16 kHz float samples Format.h fixes, which
// is the only thing above this has a use for.
//
// MakeASound has no file decoder to borrow — it opens capture devices and hands
// out the buffers they fill, and the miniaudio that does carry `ma_decoder` is
// PRIVATE to its target (makeasound Lib/CMakeLists.txt) — so this is written
// here rather than delegated. It is deliberately the smallest reader that is
// honest about what it refuses:
//
//   * 16-bit signed PCM (format 1) and 32-bit IEEE float (format 3), including
//     both spelled through WAVE_FORMAT_EXTENSIBLE, which is what a recorder
//     writes once it has more than two channels or names a channel mask.
//   * A multi-channel file yields its first channel. Mixing channels down is a
//     decision about the signal, and a caller that meant to keep the others has
//     no way to say so through this.
//   * **No resampling.** A file at any rate but 16 kHz is a WavError naming the
//     rate it holds, because the mel front-end's window, hop and filterbank are
//     all written against that number and a resampler is a module that does not
//     exist yet.
//
// Chunks are walked rather than assumed to be in any order: jfk.wav itself
// carries a LIST/INFO between `fmt ` and `data`, so a reader that took the
// third chunk to be the samples would read metadata as audio.
Vector<float> readWavFile(const std::filesystem::path& path);
} // namespace WSP
