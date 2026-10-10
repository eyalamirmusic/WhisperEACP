#include "CoreMLMel.h"

#include <WhisperEACP/Mel/Window.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace WSP
{
namespace
{
using eacp::GPU::Float;
using eacp::ML::Graph;
using GraphTensor = eacp::ML::Tensor;

constexpr auto log10OfE = 0.4342944819032518f;

// HF's floor, log10 of 1e-10, and Whisper's clamp below the loudest cell.
constexpr auto floorLog = -10.0f;
constexpr auto decadesKept = 8.0f;

// fp16's smallest subnormal: the scaled mel is held at or above it before its
// log, so an empty cell's log is finite.
constexpr auto smallestHalf = 5.9604644775390625e-8f;

// Where the loudest bin's power is put, 2^15: under fp16's 65504 with room
// for the filterbank's rounding, and eight decades above it still normal.
constexpr auto powerPeak = 32768.0f;

// The quietest power the spectrum gain lifts to powerPeak, fp16's smallest
// normal, which caps that gain at about 23000.
constexpr auto quietestPower = 6.103515625e-5f;

// Where each hop row a frame reads lies against the frame's taps. Frame t
// starts lead rows before row t, offset columns into that row, so the rows it
// reads are padded rows t to t + lastRow, with the leading lead rows the
// reflection before the signal.
struct Framing
{
    explicit Framing(const MelShape& shape)
        : hop(shape.hopLength)
        , taps(shape.fftLength)
        , frames(shape.frameCount)
        , signalRows(shape.sampleCount / shape.hopLength)
        , lead((shape.fftLength / 2 + shape.hopLength - 1) / shape.hopLength)
        , offset(lead * shape.hopLength - shape.fftLength / 2)
        , lastRow((offset + shape.fftLength - 1) / shape.hopLength)
    {
    }

    int paddedRows() const { return frames + lastRow; }
    int trailingRows() const { return paddedRows() - lead - signalRows; }

    int firstTapIn(int row) const { return std::max(0, row * hop - offset); }
    int endTapIn(int row) const { return std::min(taps, (row + 1) * hop - offset); }

    int hop;
    int taps;
    int frames;
    int signalRows;
    int lead;
    int offset;
    int lastRow;
};

void requireFrameable(const MelShape& shape, Span<const float> filterBank)
{
    if (shape.hopLength <= 0 || shape.sampleCount % shape.hopLength != 0)
        throw std::logic_error {"the Core ML mel reads its samples a hop to a "
                                "row, so the window is a whole number of hops"};

    if ((int) filterBank.size() != shape.filterElementCount())
        throw std::logic_error {"the filterbank holds "
                                + std::to_string(filterBank.size())
                                + " values, and the mel's shape wants "
                                + std::to_string(shape.filterElementCount())};

    const auto framing = Framing {shape};

    if (framing.lead + 1 > framing.signalRows
        || framing.trailingRows() + 1 > framing.signalRows)
        throw std::logic_error {"the Core ML mel's window is shorter than the "
                                "reflection at its edges"};
}

// A 0/1 matrix, [sourceWidth, targetWidth]: column c takes source element
// sourceOf(c).
template <typename SourceOf>
Vector<float> permutation(int sourceWidth, int targetWidth, SourceOf sourceOf)
{
    auto values = Vector<float> {};
    values.resize(sourceWidth * targetWidth, 0.0f);

    for (auto column = 0; column < targetWidth; ++column)
        values[sourceOf(column) * targetWidth + column] = 1.0f;

    return values;
}

// The signal's first lead rows' worth of reflection: padded element
// c - lead * hop is sample lead * hop - c, for c in [0, lead * hop).
GraphTensor leadingReflection(Graph& graph, GraphTensor rows, const Framing& at)
{
    const auto sourceWidth = (at.lead + 1) * at.hop;
    const auto targetWidth = at.lead * at.hop;

    const auto head = graph.reshape(graph.slice(rows, {0, 0}, {at.lead + 1, at.hop}),
                                    {1, sourceWidth});

    const auto order =
        permutation(sourceWidth,
                    targetWidth,
                    [targetWidth](int column) { return targetWidth - column; });
    const auto reflect = graph.halfConstant(
        "mel_leading_reflection", {sourceWidth, targetWidth}, order);

    return graph.reshape(graph.matmul(head, reflect), {at.lead, at.hop});
}

// And after the last sample: padded element samples + c is sample
// samples - 2 - c, read out of the last trailing + 1 rows.
GraphTensor trailingReflection(Graph& graph, GraphTensor rows, const Framing& at)
{
    const auto trailing = at.trailingRows();
    const auto sourceWidth = (trailing + 1) * at.hop;
    const auto targetWidth = trailing * at.hop;

    const auto tail = graph.reshape(graph.slice(rows,
                                                {at.signalRows - trailing - 1, 0},
                                                {at.signalRows, at.hop}),
                                    {1, sourceWidth});

    const auto order =
        permutation(sourceWidth,
                    targetWidth,
                    [sourceWidth](int column) { return sourceWidth - 2 - column; });
    const auto reflect = graph.halfConstant(
        "mel_trailing_reflection", {sourceWidth, targetWidth}, order);

    return graph.reshape(graph.matmul(tail, reflect), {trailing, at.hop});
}

GraphTensor paddedRows(Graph& graph, GraphTensor rows, const Framing& at)
{
    auto parts = Vector<GraphTensor> {leadingReflection(graph, rows, at)};
    const auto trailing = at.trailingRows();

    if (trailing >= 0)
        parts.add(rows);
    else
        parts.add(graph.slice(rows, {0, 0}, {at.signalRows + trailing, at.hop}));

    if (trailing > 0)
        parts.add(trailingReflection(graph, rows, at));

    return graph.concat(parts, 0);
}

// The Hann-windowed DFT over taps [first, end), [end - first, 2 * bins]: the
// real parts of every bin in the first bins columns, the imaginary in the rest.
Vector<float> windowedBasis(const MelShape& shape, int first, int end)
{
    const auto bins = shape.binCount();
    const auto length = shape.fftLength;
    const auto window = periodicHannWindow(length);

    auto basis = Vector<float> {};
    basis.resize(2 * bins * (end - first), 0.0f);

    for (auto tap = first; tap < end; ++tap)
    {
        const auto row = (tap - first) * 2 * bins;

        for (auto bin = 0; bin < bins; ++bin)
        {
            const auto turns = (double) ((bin * tap) % length) / (double) length;
            const auto angle = 2.0 * std::numbers::pi * turns;
            const auto weight = (double) window[tap];

            basis[row + bin] = (float) (weight * std::cos(angle));
            basis[row + bins + bin] = (float) (-weight * std::sin(angle));
        }
    }

    return basis;
}

// [frames, 2 * bins]: every frame's spectrum, one product per row a frame
// reads, summed.
GraphTensor spectrum(Graph& graph,
                     GraphTensor padded,
                     const MelShape& shape,
                     const Framing& at)
{
    const auto plus = [](const Float& a, const Float& b) { return a + b; };
    auto total = GraphTensor {};

    for (auto row = 0; row <= at.lastRow; ++row)
    {
        const auto first = at.firstTapIn(row);
        const auto end = at.endTapIn(row);
        const auto column = at.offset + first - row * at.hop;

        const auto frameRows = graph.slice(
            padded, {row, column}, {row + at.frames, column + end - first});

        const auto values = windowedBasis(shape, first, end);
        const auto basis = graph.halfConstant("mel_basis_" + std::to_string(row),
                                              {end - first, 2 * shape.binCount()},
                                              values);

        const auto product = graph.matmul(frameRows, basis);
        total = row == 0 ? product : graph.apply(total, product, plus);
    }

    return total;
}

float largestRowSum(Span<const float> filterBank, const MelShape& shape)
{
    auto largest = 0.0f;

    for (auto band = 0; band < shape.melCount; ++band)
    {
        auto sum = 0.0f;

        for (auto bin = 0; bin < shape.binCount(); ++bin)
            sum += std::abs(filterBank[band * shape.binCount() + bin]);

        largest = std::max(largest, sum);
    }

    return largest > 0.0f ? largest : 1.0f;
}

GraphTensor peakOf(Graph& graph, GraphTensor matrix)
{
    return graph.max(graph.max(matrix, 1, true), 0, true);
}
} // namespace

eacp::ML::Shape logMelInputShape(const MelShape& shape)
{
    return {shape.sampleCount / shape.hopLength, shape.hopLength};
}

Graph recordLogMelProgram(const MelShape& shape, Span<const float> filterBank)
{
    auto graph = Graph {};
    const auto samples = graph.input(
        samplesFeature, logMelInputShape(shape), eacp::ML::DType::float16);

    graph.output(recordLogMel(graph, samples, shape, filterBank), melFeature);

    return graph;
}

GraphTensor recordLogMel(Graph& graph,
                         GraphTensor samples,
                         const MelShape& shape,
                         Span<const float> filterBank)
{
    requireFrameable(shape, filterBank);

    const auto at = Framing {shape};
    const auto bins = shape.binCount();

    const auto magnitude = [](const Float& x) { return abs(x); };
    const auto gain = graph.apply(peakOf(graph, graph.apply(samples, magnitude)),
                                  [](const Float& peak)
                                  { return 1.0f / max(quietestPeak, peak); });

    const auto scaled = graph.apply(
        samples, gain, [](const Float& x, const Float& g) { return x * g; });

    const auto frames = spectrum(graph, paddedRows(graph, scaled, at), shape, at);

    const auto real = graph.slice(frames, {0, 0}, {at.frames, bins});
    const auto imaginary = graph.slice(frames, {0, bins}, {at.frames, 2 * bins});

    const auto squared = [](const Float& re, const Float& im)
    { return re * re + im * im; };

    // The spectrum's own gain, applied before the square so that a quiet
    // bin's power is the square of a normal number rather than a subnormal.
    const auto spectrumGain =
        graph.apply(peakOf(graph, graph.apply(real, imaginary, squared)),
                    [](const Float& peak)
                    { return sqrt(powerPeak / max(quietestPower, peak)); });

    const auto power = graph.apply(
        real,
        imaginary,
        spectrumGain,
        [](const Float& re, const Float& im, const Float& g)
        {
            const auto scaledReal = re * g;
            const auto scaledImaginary = im * g;
            return scaledReal * scaledReal + scaledImaginary * scaledImaginary;
        });

    const auto filterScale = 1.0f / largestRowSum(filterBank, shape);
    auto scaledFilters = Vector<float> {};
    scaledFilters.resize(filterBank.size(), 0.0f);

    for (auto band = 0; band < shape.melCount; ++band)
        for (auto bin = 0; bin < bins; ++bin)
            scaledFilters[bin * shape.melCount + band] =
                filterBank[band * bins + bin] * filterScale;

    const auto filters =
        graph.halfConstant("mel_filters", {bins, shape.melCount}, scaledFilters);
    const auto mel = graph.matmul(power, filters);

    const auto logMel = graph.apply(
        mel, [](const Float& m) { return log(max(smallestHalf, m)) * log10OfE; });

    // The scaled domain's log less the true one's: log10 of the filter scale,
    // and twice log10 of each gain, since power is the square of what they
    // scaled. Each gain's log apart, since their product need not fit fp16.
    const auto filterDecades = std::log10(filterScale);
    const auto scaleDecades = graph.apply(
        gain,
        spectrumGain,
        [filterDecades](const Float& g, const Float& h)
        { return (log(g) + log(h)) * (2.0f * log10OfE) + filterDecades; });

    const auto normalise =
        [](const Float& cell, const Float& peak, const Float& decades)
    {
        const auto truePeak = peak - decades;
        const auto flooredPeak = max(floorLog, truePeak);
        const auto belowPeak = cell - peak + (truePeak - flooredPeak);
        const auto kept = max(max(-decadesKept, belowPeak), floorLog - flooredPeak);

        return (kept + flooredPeak + 4.0f) * 0.25f;
    };

    return graph.apply(logMel, peakOf(graph, logMel), scaleDecades, normalise);
}
} // namespace WSP
