#include "Common.h"

#include <algorithm>
#include <limits>

using namespace nano;
using namespace WSP;
using namespace eacp::GPU;

namespace
{
// The shape the two kernels and the three references are all indexed by, so a
// test names its numbers once. modelWidth is derived rather than given: W is
// H * D by definition, and a shape where it is not is not a shape attention has.
struct AttentionShape
{
    int headCount = 1;
    int headWidth = 1;
    int queryCount = 1;
    int keyCount = 1;

    int modelWidth() const { return headCount * headWidth; }
    int scoreCount() const { return headCount * queryCount * keyCount; }
    int queryElementCount() const { return queryCount * modelWidth(); }
    int keyElementCount() const { return keyCount * modelWidth(); }
    int scoreRowCount() const { return headCount * queryCount; }

    double defaultScale() const { return 1.0 / std::sqrt((double) headWidth); }
};

// Whether the causal rule suppresses this key for this query, from the rule
// itself rather than from the kernel's unsigned rearrangement of it: query i
// stands at absolute position keyCount - queryCount + i, and a key past that
// position has not happened yet.
bool isLaterThanQuery(const AttentionShape& shape, int query, int key)
{
    return key > shape.keyCount - shape.queryCount + query;
}

// scores[h, i, j] = scale * dot(queries[i, head h], keys[j, head h]), laid out
// as headCount * queryCount rows of keyCount.
Vector<float> scoresReference(const Vector<float>& queries,
                              const Vector<float>& keys,
                              const AttentionShape& shape,
                              double scale)
{
    auto width = shape.modelWidth();
    auto expected = sized(shape.scoreCount());

    for (auto head = 0; head < shape.headCount; ++head)
    {
        auto headColumn = head * shape.headWidth;

        for (auto query = 0; query < shape.queryCount; ++query)
        {
            for (auto key = 0; key < shape.keyCount; ++key)
            {
                auto total = 0.0;

                for (auto channel = 0; channel < shape.headWidth; ++channel)
                    total += (double) queries[query * width + headColumn + channel]
                             * keys[key * width + headColumn + channel];

                auto at = (head * shape.queryCount + query) * shape.keyCount + key;
                expected[at] = (float) (scale * total);
            }
        }
    }

    return expected;
}

// out[i, h * D + c] = sum over j of probabilities[h, i, j] * values[j, h * D + c].
Vector<float> applyReference(const Vector<float>& probabilities,
                             const Vector<float>& values,
                             const AttentionShape& shape)
{
    auto width = shape.modelWidth();
    auto expected = sized(shape.queryElementCount());

    for (auto query = 0; query < shape.queryCount; ++query)
    {
        for (auto column = 0; column < width; ++column)
        {
            auto head = column / shape.headWidth;
            auto base = (head * shape.queryCount + query) * shape.keyCount;
            auto total = 0.0;

            for (auto key = 0; key < shape.keyCount; ++key)
                total += (double) probabilities[base + key]
                         * values[key * width + column];

            expected[query * width + column] = (float) total;
        }
    }

    return expected;
}

// The whole of eager multi-head attention in doubles, written straight from the
// definition rather than out of the two helpers above — the chain is what this
// checks, so its reference has to be independent of the pieces the chain is
// built from.
//
// Causal masking here is exclusion, not a sentinel: a suppressed key takes no
// part in the maximum, contributes nothing to the total and carries a weight of
// exactly zero. That is the thing the kernel's very negative score is supposed
// to be indistinguishable from, so a reference that also wrote a sentinel would
// be checking the sentinel against itself.
Vector<float> attentionReference(const Vector<float>& queries,
                                 const Vector<float>& keys,
                                 const Vector<float>& values,
                                 const AttentionShape& shape,
                                 double scale,
                                 bool causal)
{
    auto width = shape.modelWidth();
    auto expected = sized(shape.queryElementCount());
    auto row = Vector<double>();
    auto attends = Vector<int>();
    row.resize(shape.keyCount);
    attends.resize(shape.keyCount);

    for (auto head = 0; head < shape.headCount; ++head)
    {
        auto headColumn = head * shape.headWidth;

        for (auto query = 0; query < shape.queryCount; ++query)
        {
            auto largest = -std::numeric_limits<double>::infinity();

            for (auto key = 0; key < shape.keyCount; ++key)
            {
                attends[key] = causal && isLaterThanQuery(shape, query, key) ? 0 : 1;

                if (attends[key] == 0)
                {
                    row[key] = 0.0;
                    continue;
                }

                auto total = 0.0;

                for (auto channel = 0; channel < shape.headWidth; ++channel)
                    total += (double) queries[query * width + headColumn + channel]
                             * keys[key * width + headColumn + channel];

                row[key] = scale * total;
                largest = std::max(largest, row[key]);
            }

            auto total = 0.0;

            for (auto key = 0; key < shape.keyCount; ++key)
            {
                row[key] = attends[key] == 0 ? 0.0 : std::exp(row[key] - largest);
                total += row[key];
            }

            for (auto channel = 0; channel < shape.headWidth; ++channel)
            {
                auto column = headColumn + channel;
                auto blended = 0.0;

                for (auto key = 0; key < shape.keyCount; ++key)
                    blended += row[key] * values[key * width + column];

                expected[query * width + column] = (float) (blended / total);
            }
        }
    }

    return expected;
}

Vector<float> runScores(const Vector<float>& queries,
                        const Vector<float>& keys,
                        const AttentionShape& shape,
                        double scale,
                        bool causal = false)
{
    auto queryBuffer = storageOf(queries);
    auto keyBuffer = storageOf(keys);
    auto scores = outputFor(shape.scoreCount());

    auto kernel = AttentionScores {};
    kernel.queries = queryBuffer;
    kernel.keys = keyBuffer;
    kernel.scores = scores;
    kernel.modelWidth = (unsigned) shape.modelWidth();
    kernel.headWidth = (unsigned) shape.headWidth;
    kernel.queryCount = (unsigned) shape.queryCount;
    kernel.keyCount = (unsigned) shape.keyCount;
    kernel.causal = causal ? 1u : 0u;
    kernel.scale = (float) scale;

    return runOverGrid(kernel, scores, shape.keyCount, shape.scoreRowCount());
}

Vector<float> runApply(const Vector<float>& probabilities,
                       const Vector<float>& values,
                       const AttentionShape& shape)
{
    auto probabilityBuffer = storageOf(probabilities);
    auto valueBuffer = storageOf(values);
    auto output = outputFor(shape.queryElementCount());

    auto kernel = AttentionApply {};
    kernel.probabilities = probabilityBuffer;
    kernel.values = valueBuffer;
    kernel.output = output;
    kernel.modelWidth = (unsigned) shape.modelWidth();
    kernel.headWidth = (unsigned) shape.headWidth;
    kernel.queryCount = (unsigned) shape.queryCount;
    kernel.keyCount = (unsigned) shape.keyCount;

    return runOverGrid(kernel, output, shape.modelWidth(), shape.queryCount);
}

// The three dispatches attention is, each its own pass: threads of a dispatch
// are ordered against the next dispatch's reads by the end of that dispatch and
// nothing else, and the softmax reads every score its predecessor wrote.
//
// The scores buffer needs no reshaping between the first and the second: a
// [headCount, queryCount, keyCount] row-major block already is
// headCount * queryCount rows of keyCount, which is the only thing Softmax
// wants to know about it.
// The softmaxed scores on their own, which is where a causal mask has to be
// visible as an exact zero rather than as a small weight.
Vector<float> runNormalisedScores(const Vector<float>& queries,
                                  const Vector<float>& keys,
                                  const AttentionShape& shape,
                                  double scale,
                                  bool causal)
{
    auto scores = runScores(queries, keys, shape, scale, causal);
    auto scoreBuffer = storageOf(scores);
    auto probabilities = outputFor(shape.scoreCount());

    auto normalising = Softmax {};
    normalising.input = scoreBuffer;
    normalising.output = probabilities;
    normalising.rowLength = (unsigned) shape.keyCount;

    return runOverRows(
        normalising, probabilities, shape.scoreRowCount(), shape.scoreCount());
}

Vector<float> runAttention(const Vector<float>& queries,
                           const Vector<float>& keys,
                           const Vector<float>& values,
                           const AttentionShape& shape,
                           double scale,
                           bool causal = false)
{
    auto queryBuffer = storageOf(queries);
    auto keyBuffer = storageOf(keys);
    auto valueBuffer = storageOf(values);

    auto scores = outputFor(shape.scoreCount());
    auto probabilities = outputFor(shape.scoreCount());
    auto output = outputFor(shape.queryElementCount());

    auto scoring = AttentionScores {};
    scoring.queries = queryBuffer;
    scoring.keys = keyBuffer;
    scoring.scores = scores;
    scoring.modelWidth = (unsigned) shape.modelWidth();
    scoring.headWidth = (unsigned) shape.headWidth;
    scoring.queryCount = (unsigned) shape.queryCount;
    scoring.keyCount = (unsigned) shape.keyCount;
    scoring.causal = causal ? 1u : 0u;
    scoring.scale = (float) scale;
    scoring.prepare();

    auto normalising = Softmax {};
    normalising.input = scores;
    normalising.output = probabilities;
    normalising.rowLength = (unsigned) shape.keyCount;
    normalising.prepare();

    auto applying = AttentionApply {};
    applying.probabilities = probabilities;
    applying.values = valueBuffer;
    applying.output = output;
    applying.modelWidth = (unsigned) shape.modelWidth();
    applying.headWidth = (unsigned) shape.headWidth;
    applying.queryCount = (unsigned) shape.queryCount;
    applying.keyCount = (unsigned) shape.keyCount;
    applying.prepare();

    auto commands = Device::shared().makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        pass.dispatch(scoring, shape.keyCount, shape.scoreRowCount());
    }

    {
        auto pass = commands.beginCompute();
        pass.dispatch(normalising, shape.scoreRowCount());
    }

    {
        auto pass = commands.beginCompute();
        pass.dispatch(applying, shape.modelWidth(), shape.queryCount);
    }

    commands.commit();
    return readBack(output, shape.queryElementCount());
}

// Nothing here is a multiple of the 8 x 8 dispatch group, the head count does
// not divide the query count, and no two of the four extents agree — so a
// kernel that confused a head stride for a row stride, or leaned on the grid
// landing exactly on the matrix, fails here rather than on the first real model.
constexpr auto oddShape = AttentionShape {2, 3, 5, 7};

// One query against many keys: the shape a decoder step has once a KV cache is
// carrying every key of the sequence so far, and the one a kernel that quietly
// assumed a square score matrix cannot survive.
constexpr auto cacheShape = AttentionShape {4, 2, 1, 11};

// Causal masking's two shapes. Square is the decoder's self-attention over a
// whole prompt at once, where the rule is the plain lower triangle. The second
// is what the same prompt looks like resumed against a cache: three new queries
// against seven keys, so query 0 stands at position 4 and the mask is a
// trapezoid rather than a triangle — the case an implementation that took the
// query index for an absolute position gets wrong by exactly the cache's depth.
constexpr auto causalSquareShape = AttentionShape {2, 3, 5, 5};
constexpr auto causalCacheShape = AttentionShape {3, 2, 3, 7};
} // namespace

auto tAttentionScoresMatchCpu = test("Kernels/attentionScoresMatchCpu") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto queries = spreadValues(oddShape.queryElementCount(), 11235u, 2.f);
    auto keys = spreadValues(oddShape.keyElementCount(), 81321u, 2.f);
    auto scale = oddShape.defaultScale();

    auto result = runScores(queries, keys, oddShape, scale);
    auto expected = scoresReference(queries, keys, oddShape, scale);

    check(result.size() == oddShape.scoreCount());

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
};

auto tAttentionScoresOneQueryManyKeys =
    test("Kernels/attentionScoresOneQueryManyKeys") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto queries = spreadValues(cacheShape.queryElementCount(), 60613u, 3.f);
    auto keys = spreadValues(cacheShape.keyElementCount(), 94110u, 3.f);
    auto scale = cacheShape.defaultScale();

    auto result = runScores(queries, keys, cacheShape, scale);
    auto expected = scoresReference(queries, keys, cacheShape, scale);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
};

// A head reads only its own columns, so zeroing every other head's slice of the
// keys must leave head zero's scores untouched. A kernel that walked the whole
// model width instead of the head width still matches its own reference; it
// does not survive this.
auto tAttentionScoresHeadsAreIndependent =
    test("Kernels/attentionScoresHeadsAreIndependent") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto queries = spreadValues(oddShape.queryElementCount(), 17u, 2.f);
    auto keys = spreadValues(oddShape.keyElementCount(), 19u, 2.f);
    auto scale = oddShape.defaultScale();

    auto firstHeadOnly = keys;

    for (auto key = 0; key < oddShape.keyCount; ++key)
        for (auto column = oddShape.headWidth; column < oddShape.modelWidth();
             ++column)
            firstHeadOnly[key * oddShape.modelWidth() + column] = 0.f;

    auto whole = runScores(queries, keys, oddShape, scale);
    auto trimmed = runScores(queries, firstHeadOnly, oddShape, scale);

    for (auto i = 0; i < oddShape.queryCount * oddShape.keyCount; ++i)
        check(whole[i] == trimmed[i]);
};

auto tAttentionApplyMatchesCpu = test("Kernels/attentionApplyMatchesCpu") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto probabilities = spreadValues(oddShape.scoreCount(), 27182u, 1.f);
    auto values = spreadValues(oddShape.keyElementCount(), 31415u, 2.f);

    auto result = runApply(probabilities, values, oddShape);
    auto expected = applyReference(probabilities, values, oddShape);

    check(result.size() == oddShape.queryElementCount());

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
};

auto tAttentionApplyOneQueryManyKeys =
    test("Kernels/attentionApplyOneQueryManyKeys") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto probabilities = spreadValues(cacheShape.scoreCount(), 5772u, 1.f);
    auto values = spreadValues(cacheShape.keyElementCount(), 16180u, 2.f);

    auto result = runApply(probabilities, values, cacheShape);
    auto expected = applyReference(probabilities, values, cacheShape);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
};

// Scores, softmax, apply — against a reference that computes all three from the
// definition in doubles, so a mismatch is the chain rather than any one stage
// agreeing with itself.
auto tAttentionChainMatchesCpu = test("Kernels/attentionChainMatchesCpu") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto queries = spreadValues(oddShape.queryElementCount(), 2718u, 2.f);
    auto keys = spreadValues(oddShape.keyElementCount(), 1618u, 2.f);
    auto values = spreadValues(oddShape.keyElementCount(), 1414u, 3.f);
    auto scale = oddShape.defaultScale();

    auto result = runAttention(queries, keys, values, oddShape, scale);
    auto expected =
        attentionReference(queries, keys, values, oddShape, scale, false);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
};

auto tAttentionChainOneQueryManyKeys =
    test("Kernels/attentionChainOneQueryManyKeys") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto queries = spreadValues(cacheShape.queryElementCount(), 33u, 2.f);
    auto keys = spreadValues(cacheShape.keyElementCount(), 44u, 2.f);
    auto values = spreadValues(cacheShape.keyElementCount(), 55u, 3.f);
    auto scale = cacheShape.defaultScale();

    auto result = runAttention(queries, keys, values, cacheShape, scale);
    auto expected =
        attentionReference(queries, keys, values, cacheShape, scale, false);

    check(result.size() == cacheShape.queryElementCount());

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
};

// Zero queries make every score zero, so the softmax is uniform and attention
// is the mean of the values down the keys — the one case whose answer can be
// written out without computing the chain, and the one a head offset applied to
// V but not to the probabilities still gets wrong.
auto tAttentionWithZeroQueriesAveragesValues =
    test("Kernels/attentionWithZeroQueriesAveragesValues") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto queries = sized(oddShape.queryElementCount());

    for (auto i = 0; i < queries.size(); ++i)
        queries[i] = 0.f;

    auto keys = spreadValues(oddShape.keyElementCount(), 909u, 2.f);
    auto values = spreadValues(oddShape.keyElementCount(), 707u, 4.f);

    auto result =
        runAttention(queries, keys, values, oddShape, oddShape.defaultScale());

    for (auto column = 0; column < oddShape.modelWidth(); ++column)
    {
        auto mean = 0.0;

        for (auto key = 0; key < oddShape.keyCount; ++key)
            mean += values[key * oddShape.modelWidth() + column];

        mean /= oddShape.keyCount;

        for (auto query = 0; query < oddShape.queryCount; ++query)
            check(
                isClose(result[query * oddShape.modelWidth() + column], mean, 1e-5));
    }
};

// The mask where it is written rather than where it is felt: every suppressed
// score is the sentinel exactly, every other score is what the unmasked kernel
// computes, and the two sets are the ones the rule names. A sentinel that leaked
// into an allowed entry, or an index that walked the triangle the wrong way, is
// a wrong score here rather than a small difference three dispatches later.
auto tAttentionCausalScoresSuppressLaterKeys =
    test("Kernels/attentionCausalScoresSuppressLaterKeys") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    for (const auto& shape: {causalSquareShape, causalCacheShape})
    {
        auto queries = spreadValues(shape.queryElementCount(), 8080u, 2.f);
        auto keys = spreadValues(shape.keyElementCount(), 8081u, 2.f);
        auto scale = shape.defaultScale();

        auto result = runScores(queries, keys, shape, scale, true);
        auto open = scoresReference(queries, keys, shape, scale);

        for (auto head = 0; head < shape.headCount; ++head)
        {
            for (auto query = 0; query < shape.queryCount; ++query)
            {
                for (auto key = 0; key < shape.keyCount; ++key)
                {
                    const auto at =
                        (head * shape.queryCount + query) * shape.keyCount + key;

                    if (isLaterThanQuery(shape, query, key))
                        check(result[at] == causalMaskScore);
                    else
                        check(isClose(result[at], open[at], 1e-5));
                }
            }
        }
    }
};

// The sentinel is only worth what the softmax makes of it, so this is the
// assertion that matters: a suppressed key comes out of Softmax at exactly
// zero, not at a weight too small to notice, and the row still sums to one over
// the keys that are left.
auto tAttentionCausalWeightsAreExactlyZero =
    test("Kernels/attentionCausalWeightsAreExactlyZero") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    for (const auto& shape: {causalSquareShape, causalCacheShape})
    {
        auto queries = spreadValues(shape.queryElementCount(), 9090u, 6.f);
        auto keys = spreadValues(shape.keyElementCount(), 9091u, 6.f);

        auto result =
            runNormalisedScores(queries, keys, shape, shape.defaultScale(), true);

        for (auto head = 0; head < shape.headCount; ++head)
        {
            for (auto query = 0; query < shape.queryCount; ++query)
            {
                auto total = 0.0;

                for (auto key = 0; key < shape.keyCount; ++key)
                {
                    const auto at =
                        (head * shape.queryCount + query) * shape.keyCount + key;

                    if (isLaterThanQuery(shape, query, key))
                        check(result[at] == 0.f);
                    else
                        check(result[at] > 0.f);

                    total += result[at];
                }

                check(isClose((float) total, 1.0, 1e-5));
            }
        }
    }
};

// Scores, softmax, apply, with the mask on — against a reference that excludes
// the suppressed keys outright, so what is being checked is that the sentinel
// and the exclusion are the same computation and not merely close.
auto tAttentionCausalChainMatchesCpu =
    test("Kernels/attentionCausalChainMatchesCpu") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto queries = spreadValues(causalSquareShape.queryElementCount(), 111u, 2.f);
    auto keys = spreadValues(causalSquareShape.keyElementCount(), 222u, 2.f);
    auto values = spreadValues(causalSquareShape.keyElementCount(), 333u, 3.f);
    auto scale = causalSquareShape.defaultScale();

    auto result =
        runAttention(queries, keys, values, causalSquareShape, scale, true);

    auto expected =
        attentionReference(queries, keys, values, causalSquareShape, scale, true);

    check(result.size() == causalSquareShape.queryElementCount());

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
};

// The same chain at the shape a resumed decoder has, where the queries are the
// last three of seven positions rather than all of them.
auto tAttentionCausalChainOverACache =
    test("Kernels/attentionCausalChainOverACache") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto queries = spreadValues(causalCacheShape.queryElementCount(), 444u, 2.f);
    auto keys = spreadValues(causalCacheShape.keyElementCount(), 555u, 2.f);
    auto values = spreadValues(causalCacheShape.keyElementCount(), 666u, 3.f);
    auto scale = causalCacheShape.defaultScale();

    auto result = runAttention(queries, keys, values, causalCacheShape, scale, true);

    auto expected =
        attentionReference(queries, keys, values, causalCacheShape, scale, true);

    check(result.size() == causalCacheShape.queryElementCount());

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
};

// The last query of a causal block sees every key, so its output row has to be
// the one the unmasked chain produces — bit for bit, since it is the same
// arithmetic in the same order. That is what says the mask suppresses nothing
// it was not asked to, from the side the tolerance-based checks cannot see.
auto tAttentionCausalLastQueryIsUnmasked =
    test("Kernels/attentionCausalLastQueryIsUnmasked") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    for (const auto& shape: {causalSquareShape, causalCacheShape})
    {
        auto queries = spreadValues(shape.queryElementCount(), 777u, 2.f);
        auto keys = spreadValues(shape.keyElementCount(), 888u, 2.f);
        auto values = spreadValues(shape.keyElementCount(), 999u, 3.f);
        auto scale = shape.defaultScale();

        auto masked = runAttention(queries, keys, values, shape, scale, true);
        auto open = runAttention(queries, keys, values, shape, scale, false);

        const auto lastRow = (shape.queryCount - 1) * shape.modelWidth();

        for (auto column = 0; column < shape.modelWidth(); ++column)
            check(masked[lastRow + column] == open[lastRow + column]);
    }
};

// What decides how negative the sentinel has to be, and the one case a merely
// smallish one gets wrong: a row whose real scores all sit far below the mask
// value. The softmax subtracts the row maximum, so a sentinel the row can rise
// above becomes that maximum, and the suppressed key walks off with the weight
// instead of none of it — the exact inversion of what the mask is for.
//
// One channel per head, strongly negative in every query and strongly positive
// in every key, drags every score to about -1800 while the remaining channels
// keep the row from being uniform. The assertions are the exact ones —
// suppressed weights zero, allowed weights positive, the row summing to one —
// so none of this rides on a tolerance.
auto tAttentionCausalMaskOutrunsVeryNegativeScores =
    test("Kernels/attentionCausalMaskOutrunsVeryNegativeScores") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto shape = AttentionShape {2, 4, 5, 5};

    auto queries = spreadValues(shape.queryElementCount(), 1234u, 2.f);
    auto keys = spreadValues(shape.keyElementCount(), 4321u, 2.f);

    for (auto head = 0; head < shape.headCount; ++head)
    {
        const auto channel = head * shape.headWidth;

        for (auto query = 0; query < shape.queryCount; ++query)
            queries[query * shape.modelWidth() + channel] = -60.f;

        for (auto key = 0; key < shape.keyCount; ++key)
            keys[key * shape.modelWidth() + channel] = 60.f;
    }

    auto plain = runScores(queries, keys, shape, shape.defaultScale(), false);

    for (auto i = 0; i < plain.size(); ++i)
        check(plain[i] < -1000.f);

    auto result =
        runNormalisedScores(queries, keys, shape, shape.defaultScale(), true);

    for (auto head = 0; head < shape.headCount; ++head)
    {
        for (auto query = 0; query < shape.queryCount; ++query)
        {
            auto total = 0.0;

            for (auto key = 0; key < shape.keyCount; ++key)
            {
                const auto at =
                    (head * shape.queryCount + query) * shape.keyCount + key;

                if (isLaterThanQuery(shape, query, key))
                    check(result[at] == 0.f);
                else
                    check(result[at] > 0.f);

                total += result[at];
            }

            check(isClose((float) total, 1.0, 1e-5));
        }
    }
};
