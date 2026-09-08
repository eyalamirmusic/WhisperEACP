#pragma once

#include "MelKernels.h"
#include "MelShape.h"

#include <WhisperEACP/Kernels/TiledMatMul.h>

#include <optional>

namespace WSP
{
// 30 s of 16 kHz mono PCM in, an 80 x 3000 log-mel spectrogram out, recorded
// into the compute pass the caller opened.
//
// The filterbank is a parameter of encode() and not state of this class: it is
// published as a matrix in the model's preprocessor_config.json, so the loader
// that reads it owns the buffer and nothing here builds one.
//
// The STFT is a framing and a product: every windowed frame is gathered into a
// row, and the rows go through the tiled kernel against the DFT basis, so the
// transform runs at the speed of a matrix multiply rather than a per-bin loop.
// The window, the basis, the frames, the spectrum, the power and the
// un-normalised mel are this object's own, sized once by prepare() and reused
// by every encode().
class MelSpectrogram
{
public:
    explicit MelSpectrogram(const MelShape& shapeToUse = {});

    void prepare(eacp::GPU::Device& device);
    void prepare();

    const MelShape& shape() const { return melShape; }

    // samples holds shape().sampleCount floats, filterBank the
    // shape().filterElementCount() of the published matrix in row-major order,
    // and output receives shape().melElementCount() floats, mel band major.
    void encode(eacp::GPU::ComputePass& pass,
                const eacp::GPU::Buffer& samples,
                const eacp::GPU::Buffer& filterBank,
                const eacp::GPU::Buffer& output);

private:
    void encodeSpectrum(eacp::GPU::ComputePass& pass,
                        const eacp::GPU::Buffer& samples);

    void encodeProjection(eacp::GPU::ComputePass& pass,
                          const eacp::GPU::Buffer& filterBank);

    const eacp::GPU::Buffer& encodePeak(eacp::GPU::ComputePass& pass);

    MelShape melShape;

    StftFramesKernel framing;
    TiledLinear transform;
    SpectrumPowerKernel squaring;
    MelProjectKernel projection;
    MaxReduceKernel reduction;
    MelNormaliseKernel normalisation;

    std::optional<eacp::GPU::Buffer> window;
    std::optional<eacp::GPU::Buffer> basis;
    std::optional<eacp::GPU::Buffer> zeroBias;
    std::optional<eacp::GPU::Buffer> frames;
    std::optional<eacp::GPU::Buffer> spectrum;
    std::optional<eacp::GPU::Buffer> power;
    std::optional<eacp::GPU::Buffer> logMel;
    std::optional<eacp::GPU::Buffer> partials;
    std::optional<eacp::GPU::Buffer> partialsOfPartials;
};
} // namespace WSP
