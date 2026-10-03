#pragma once

#include <WhisperEACP/Mel/MelShape.h>

#include <eacp/ML/Graph/Graph.h>

namespace WSP
{
// Whisper's log-mel front end recorded into an eacp::ML graph, so that it runs
// where the rest of the program runs rather than as a Metal command buffer in
// front of it. Every op is one the Neural Engine takes: slices, reshapes,
// concatenations, matrix products, elementwise arithmetic and two reductions.
//
// The input is the window's samples, shape().sampleCount of them, as
// [sampleCount / hopLength, hopLength] fp16: one hop to a row, which keeps
// every axis of every tensor inside what the engine will place, where a
// [1, 480000] vector would not be. The output is the normalised mel,
// [melCount, frameCount] band major, the [bands, frames] a convolution reads,
// or [frameCount, melCount] frames major, where a context's frames are the
// first rows and so one MultiArray::copyRows away from an encoder's input.
//
// The STFT is a product rather than a gather. Frame t is centred on sample
// hop * t, so it reads the tail of one row, whole rows, and the head of a
// last one; the product of the hop rows against the columns of the
// Hann-windowed DFT basis those taps meet, one product per row offset, summed,
// is every frame's spectrum at once. The reflect padding is the few rows
// before the first and after the last, made by a product of the edge rows
// against a permutation, exact in any precision.
//
// **fp16 is the precision, and the range is the constraint.** Whisper floors
// the log-mel eight decades below its loudest cell, and fp16 holds about nine
// between its smallest normal value and its largest. So the samples are scaled
// to a peak of one first - the frame's power is then at most (fftLength / 2)^2,
// 40000 for Whisper, under fp16's 65504 - and the filterbank to a largest row
// sum of one, which bounds the mel by the same. The log is taken in that
// scaled domain and the scale's own log taken back off before the floor
// (log10 1e-10, HF's) and the clamp, so the result is the unscaled one's.
//
// A window of silence stays finite: the gain is capped at quietestPeak's
// reciprocal, and every cell lands on HF's -1.5.
enum class MelLayout
{
    bandMajor,
    framesMajor
};

eacp::ML::Tensor recordLogMel(eacp::ML::Graph& graph,
                              eacp::ML::Tensor samples,
                              const MelShape& shape,
                              Span<const float> filterBank,
                              MelLayout layout = MelLayout::bandMajor);

// The shape of the samples input recordLogMel reads.
eacp::ML::Shape logMelInputShape(const MelShape& shape);

// A program of the mel alone: samplesFeature in, melFeature out, frames major.
// What CoreMLEncoder runs in front of an encoder that reads frames-major mel.
eacp::ML::Graph recordLogMelProgram(const MelShape& shape,
                                    Span<const float> filterBank);

inline constexpr auto samplesFeature = "samples";
inline constexpr auto melFeature = "mel";

// The quietest peak the samples are scaled up from: 2^-14, fp16's smallest
// normal value, which caps the gain at 16384.
inline constexpr auto quietestPeak = 6.103515625e-5f;
} // namespace WSP
