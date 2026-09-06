#include <WhisperEACP/Mel/Mel.h>

#include <NanoTest/NanoTest.h>

// The front-end's own shapes, pinned by value the way Tests/Audio pins the
// audio contract: these are preprocessor_config.json's feature_size and the
// non-redundant half of an n_fft-point spectrum, not choices.

using namespace nano;

auto tMelShapesAreWhispers = test("Mel/shapesAreWhispers") = []
{
    check(WSP::melBands == 80);
    check(WSP::spectrumBins == 201);
    check(WSP::spectrumBins == WSP::fftSize / 2 + 1);
};

auto tDefaultShapeIsThirtySeconds = test("Mel/defaultShapeIsThirtySeconds") = []
{
    const auto shape = WSP::MelShape {};

    check(shape.sampleCount == WSP::windowSamples);
    check(shape.fftLength == WSP::fftSize);
    check(shape.hopLength == WSP::hopSize);
    check(shape.frameCount == WSP::windowFrames);
    check(shape.melCount == WSP::melBands);

    check(shape.binCount() == WSP::spectrumBins);
    check(shape.melElementCount() == 80 * 3000);
    check(shape.filterElementCount() == 80 * 201);
    check(shape.spectrumElementCount() == 3000 * 201);
};

// HF computes 1 + sampleCount / hop frames and drops the last, which is the
// only reason 30 s of audio is 3000 frames and not 3001.
auto tFrameCountDropsTheLastFrame = test("Mel/frameCountDropsTheLastFrame") = []
{ check(1 + WSP::windowSamples / WSP::hopSize == WSP::windowFrames + 1); };
