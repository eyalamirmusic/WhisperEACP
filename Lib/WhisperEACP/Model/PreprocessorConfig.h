#pragma once

#include <WhisperEACP/Core/Core.h>
#include <WhisperEACP/Model/ModelError.h>

#include <eacp/GPU/Buffer/Buffer.h>

#include <filesystem>
#include <string_view>

namespace WSP
{
// preprocessor_config.json from the HF repo, which carries the mel filterbank
// already built — an 80 x 201 matrix for tiny.en, verified against the real
// file. That is why nothing in this project reconstructs librosa's filterbank
// or lifts one out of another project.
//
// The rest of the file restates the audio contract: sampling_rate, n_fft,
// hop_length, feature_size, chunk_length, n_samples and nb_max_frames. Those
// stay compiled-in constants in Audio/Format.h; this is what to check them
// against, and Tests/Model does exactly that.
struct PreprocessorConfig
{
    int sampleRate = 0;
    int fftSize = 0;
    int hopSize = 0;
    int melBins = 0;
    int chunkSeconds = 0;
    int windowSamples = 0;
    int maxFrames = 0;

    // Row-major, melBins rows of melFilterColumns each. HF has serialised this
    // matrix both ways round across transformers versions, so the loader
    // transposes a frequency-major file into this layout rather than making
    // every reader guess.
    Vector<float> melFilters;

    int melFilterColumns() const;
    Span<const float> melFilterRow(int bin) const;

    // The filterbank as the Mel front-end binds it: one float32 storage buffer,
    // melBins x melFilterColumns, row-major.
    eacp::GPU::Buffer makeMelFilterBuffer() const;

    static PreprocessorConfig fromFile(const std::filesystem::path& path);
    static PreprocessorConfig fromJson(std::string_view text);
};
} // namespace WSP
