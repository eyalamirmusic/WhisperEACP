#include "Common.h"

#include <iostream>

// The front-end, isolated through whisper.cpp's own encoder and decoder.
//
// whisper.cpp exposes neither its mel nor its encoder output, but it accepts a
// mel (whisper_set_mel) and returns logits (whisper_decode plus
// whisper_get_logits). So the same PCM is run twice: once entirely inside the
// reference, and once with our 80 x 3000 mel substituted for its own.
// Everything after the mel is whisper.cpp both times, so a disagreement in the
// logit row is the front-end's and nothing else's.
//
// What differs between the two front-ends by design, read out of
// log_mel_spectrogram in whisper.cpp's src/whisper.cpp rather than remembered:
//
//  * The head padding is the same reflection about the first sample — 200
//    samples, torch.stft's center=True — a reverse_copy there and the fold in
//    Mel/MelKernels.h here.
//  * The tail padding is not. HF reflects about the last sample; whisper.cpp
//    appends 30 s of zeros and frames into them. At a 30 s input only the
//    frame at index 2999 reads past the end at all, so both signals below are
//    zero-filled to exactly 480000 samples, where a reflection of zeros and a
//    pad of zeros are the same thing. That is what makes this like for like
//    rather than a comparison with a known discrepancy inside it.
//  * whisper.cpp computes 6000 frames for a 30 s input, because its zero tail
//    is part of the signal it frames. Frames 3000 and up are digital silence
//    at the log floor, which is the smallest value the spectrogram can hold,
//    so they cannot move the maximum the normalisation is taken against; and
//    the encoder reads the first 2 * n_audio_ctx = 3000 frames either way.
//  * The normalisation itself is the same: the maximum over the whole
//    spectrogram, floored eight decades below it, then (x + 4) / 4. So is the
//    periodic Hann window, and so is the 1e-10 floor before log10.
//  * The filterbank is not merely equivalent but identical: the 80 x 201
//    matrix in the GGML file and the one in preprocessor_config.json agree bit
//    for bit in all 16080 entries, so the format conversion CLAUDE.md says to
//    rule out first is ruled out for this stage.
//
// The arithmetic underneath is not the same and is not meant to be: whisper.cpp
// runs a radix-2 FFT in float over a 400-entry sine table, ours is a naive DFT
// per bin on the GPU.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;
using namespace eacp::GPU;

namespace
{
// The two tokens whisper.cpp's own greedy decoder opens an English
// transcription with. Two is enough for the row to depend on the whole encoder
// output through cross-attention, which is what makes a logit row a probe of
// the mel at all.
std::vector<whisper_token> transcriptionPrompt(whisper_context& context)
{
    return {whisper_token_sot(&context), whisper_token_not(&context)};
}

struct LogitComparison
{
    std::vector<float> ourMel;
    std::vector<float> fromReferenceMel;
    std::vector<float> fromOurMel;
    float difference = 0.0f;
    float resolution = 0.0f;
};

LogitComparison compareThroughOracle(Device& device,
                                     whisper_context& context,
                                     const std::vector<float>& samples)
{
    const auto prompt = transcriptionPrompt(context);

    auto comparison = LogitComparison {};
    comparison.fromReferenceMel = oracleLogitsForPcm(context, samples, prompt);
    comparison.ourMel = ourMelSpectrogram(device, samples);
    comparison.fromOurMel = oracleLogitsForMel(context, comparison.ourMel, prompt);
    comparison.difference = maximumAbsoluteDifference(comparison.fromReferenceMel,
                                                      comparison.fromOurMel);
    comparison.resolution = oracleLogitResolution(
        context, comparison.ourMel, comparison.fromOurMel, prompt);

    return comparison;
}

// The whole spectrogram is asserted to be finite and inside the two-wide band
// the clamp-and-scale leaves, before a single logit is looked at: a mel with a
// NaN in it reaches the encoder as one, and the logit comparison would report
// that as a front-end disagreement without saying where.
void checkMelIsWellFormed(const std::vector<float>& mel)
{
    const auto shape = MelShape {};
    check((int) mel.size() == shape.melElementCount(), "mel element count");

    auto smallest = mel.front();
    auto largest = mel.front();

    for (const auto value: mel)
    {
        check(std::isfinite(value), "mel is finite");
        smallest = std::min(smallest, value);
        largest = std::max(largest, value);
    }

    check(largest - smallest <= 2.0f + 1e-4f, "mel spans no more than two");
}

// The claim, and the two numbers behind it. A difference the reference cannot
// tell apart from a last-bit change in its own input is a difference this tier
// has nothing to say about. The factor of two is margin for a machine or a
// whisper.cpp release that rounds elsewhere, and the absolute ceiling is there
// so that a reference which had somehow gone insensitive could not pass the
// first assertion by making its own resolution enormous.
void checkAgainstTheOracle(std::string_view label, const LogitComparison& comparison)
{
    std::cout << "  " << label << ": max |dlogit| = " << comparison.difference
              << ", one ulp of mel moves it " << comparison.resolution << ", argmax "
              << argmaxOf(comparison.fromReferenceMel) << " vs "
              << argmaxOf(comparison.fromOurMel) << "\n";

    check(!comparison.fromReferenceMel.empty(), "the reference decoded");
    check(comparison.fromOurMel.size() == comparison.fromReferenceMel.size(),
          "both rows are one vocabulary wide");
    check(argmaxOf(comparison.fromOurMel) == argmaxOf(comparison.fromReferenceMel),
          "the same token wins");
    check(comparison.difference <= 2.0f * comparison.resolution,
          "our mel is inside the reference's own resolution");
    check(comparison.difference <= 0.1f, "and small in absolute terms");
}

// A signal that fills the first 11 s and then stops, so the speech-to-silence
// edge is framed the way an 11 s recording padded to the window frames it, and
// the reflection at the head has real content to reflect. Deterministic: two
// sines and the same LCG the rest of Tests/Mel uses.
std::vector<float> syntheticSignal()
{
    return paddedToWindow(melTest::toneAndNoise(11 * sampleRate));
}

bool canRun(Device& device)
{
    return device.isValid() && hasGgmlModel() && hasModelFile(preprocessorFile);
}
} // namespace

auto tOurMelDrivesTheOracleToTheSameLogits =
    test("Oracle/ourMelDrivesTheOracleToTheSameLogits") = []
{
    auto& device = Device::shared();

    if (!canRun(device))
        return;

    const auto samples = paddedToWindow(readPcm16Wav(WHISPER_EACP_JFK_WAV));
    check(samples.size() == (std::size_t) windowSamples, "jfk.wav was read");

    auto oracle = loadOracle();
    check(oracle != nullptr, "the GGML model loaded");

    if (!oracle)
        return;

    const auto comparison = compareThroughOracle(device, *oracle, samples);

    checkMelIsWellFormed(comparison.ourMel);
    checkAgainstTheOracle("jfk.wav", comparison);
};

// Speech is a narrow input: it never reaches the window's edges, never goes
// digitally silent, and never holds a pure tone. The frames a recording
// exercises are not the frames a front-end can get wrong.
auto tOurMelAgreesOnASyntheticSignal =
    test("Oracle/ourMelAgreesOnASyntheticSignal") = []
{
    auto& device = Device::shared();

    if (!canRun(device))
        return;

    auto oracle = loadOracle();
    check(oracle != nullptr, "the GGML model loaded");

    if (!oracle)
        return;

    const auto comparison = compareThroughOracle(device, *oracle, syntheticSignal());

    checkMelIsWellFormed(comparison.ourMel);
    checkAgainstTheOracle("tone and noise", comparison);
};
