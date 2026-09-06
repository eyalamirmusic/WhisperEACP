#pragma once

#include "MelKernels.h"
#include "MelShape.h"

#include <optional>

namespace WSP
{
// 30 s of 16 kHz mono PCM in, an 80 x 3000 log-mel spectrogram out, in one
// recorded command buffer.
//
// The filterbank is a parameter of encode() and not state of this class: it is
// published as a matrix in the model's preprocessor_config.json, so the loader
// that reads it owns the buffer and nothing here builds one.
//
// The window, the power spectrum and the un-normalised mel are this object's
// own, sized once by prepare() and reused by every encode().
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
    void encode(eacp::GPU::CommandBuffer& commands,
                const eacp::GPU::Buffer& samples,
                const eacp::GPU::Buffer& filterBank,
                const eacp::GPU::Buffer& output);

private:
    void encodeSpectrum(eacp::GPU::CommandBuffer& commands,
                        const eacp::GPU::Buffer& samples);

    void encodeProjection(eacp::GPU::CommandBuffer& commands,
                          const eacp::GPU::Buffer& filterBank);

    const eacp::GPU::Buffer& encodePeak(eacp::GPU::CommandBuffer& commands);

    MelShape melShape;

    StftPowerKernel spectrum;
    MelProjectKernel projection;
    MaxReduceKernel reduction;
    MelNormaliseKernel normalisation;

    std::optional<eacp::GPU::Buffer> window;
    std::optional<eacp::GPU::Buffer> power;
    std::optional<eacp::GPU::Buffer> logMel;
    std::optional<eacp::GPU::Buffer> partials;
    std::optional<eacp::GPU::Buffer> partialsOfPartials;
};
} // namespace WSP
