#include "TiledProduct.h"

// Nothing to check where eacp has no SIMD-group matrix to write one against:
// SimdTiledMatMul.h declares no program there. See CMake/Findeacp.cmake.
#if defined(WHISPER_EACP_HAS_SIMD_MATRIX)

using namespace nano;
using namespace WSP;
using namespace eacp::GPU;

// The SIMD-group product, through the same scalar reference the register-tiled
// one is checked against in TiledMatMulTests.cpp. Its tiling is coarser — a
// 64 x 64 tile of C over a 32-deep slab, against 32 x 32 over 16 — so the
// shapes here are chosen against those numbers rather than reused: what has to
// hold is that a fragment is never loaded or stored across an edge, and that
// every element of a tile hanging off one still comes out right.

// Smaller than a single tile in every extent, so the whole dispatch is one
// group whose tile is almost entirely outside the shape.
auto tSimdTiledLinearOddShape = test("Kernels/simdTiledLinearOddShape") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkLinear<SimdTiledLinear>(5, 7, 3, 100u);
};

// Every extent crosses a 64 boundary and none lands on one, and the inner
// count leaves 13 of the second 32-deep slab past the data.
auto tSimdTiledLinearRaggedTiles = test("Kernels/simdTiledLinearRaggedTiles") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkLinear<SimdTiledLinear>(100, 45, 76, 200u);
};

// One extent ragged at a time, so a failure says which of the three guards
// went: the rows, the columns, or the slab.
auto tSimdTiledLinearRaggedRows = test("Kernels/simdTiledLinearRaggedRows") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkLinear<SimdTiledLinear>(70, 64, 128, 210u);
};

auto tSimdTiledLinearRaggedColumns =
    test("Kernels/simdTiledLinearRaggedColumns") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkLinear<SimdTiledLinear>(128, 64, 70, 220u);
};

auto tSimdTiledLinearRaggedInner = test("Kernels/simdTiledLinearRaggedInner") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkLinear<SimdTiledLinear>(128, 33, 128, 230u);
};

// Every extent a multiple of the tiling: two row tiles, two column tiles and
// two whole slabs, with no clamped load and no guarded element anywhere.
auto tSimdTiledLinearWholeTiles = test("Kernels/simdTiledLinearWholeTiles") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkLinear<SimdTiledLinear>(128, 64, 128, 300u);
};

// The two widths the encoder's projections have, at a fraction of its rows:
// 384 into 384 and 384 into 1536, which is what fc1 and the four attention
// projections dispatch.
auto tSimdTiledLinearEncoderWidths =
    test("Kernels/simdTiledLinearEncoderWidths") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkLinear<SimdTiledLinear>(
        64, 384, 384, 310u, TiledProduct::dotProductTolerance(384));

    TiledProduct::checkLinear<SimdTiledLinear>(
        64, 1536, 384, 320u, TiledProduct::dotProductTolerance(1536));
};

auto tSimdTiledLinearPackedWeights =
    test("Kernels/simdTiledLinearPackedWeights") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkPackedLinear<HalfWeightSimdTiledLinear>(37, 21, 35, 400u);
    TiledProduct::checkPackedLinear<HalfWeightSimdTiledLinear>(100, 45, 76, 410u);
};

// The batch fold and the causal mask together: one batch per head over the
// head's slice of rows the batch strides address, and a trapezoid whose
// masked count is asserted rather than only its values.
auto tSimdTiledAttentionScores = test("Kernels/simdTiledAttentionScores") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkAttentionScores<SimdTiledLinear>(35, 39, 3, 5, 500u);
};

// The same, at the head width the model has and across tile boundaries in
// both extents.
auto tSimdTiledAttentionScoresModelHeads =
    test("Kernels/simdTiledAttentionScoresModelHeads") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkAttentionScores<SimdTiledLinear>(70, 80, 2, 64, 510u);
};

auto tSimdTiledAttentionApply = test("Kernels/simdTiledAttentionApply") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkAttentionApply<SimdTiledMatMul>(41, 37, 2, 9, 600u);
};

// The head width the encoder applies over is exactly one column tile, and the
// keys it sums across are the ragged inner extent.
auto tSimdTiledAttentionApplyModelHeads =
    test("Kernels/simdTiledAttentionApplyModelHeads") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkAttentionApply<SimdTiledMatMul>(70, 45, 2, 64, 610u);
};

// The softmax the pair does between them, at a shape whose keys are ragged in
// both the scores' column tiles and the apply's inner extent, and at the model
// head width, where the apply's columns are exactly one tile.
auto tSimdSoftmaxAttention = test("Kernels/simdSoftmaxAttention") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkSoftmaxAttention<MaximaSimdTiledLinear,
                                        SoftmaxSimdTiledMatMul>(41, 37, 2, 9, 700u);
};

auto tSimdSoftmaxAttentionModelHeads =
    test("Kernels/simdSoftmaxAttentionModelHeads") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkSoftmaxAttention<MaximaSimdTiledLinear,
                                        SoftmaxSimdTiledMatMul>(
        70, 150, 2, 64, 710u);
};

auto tSimdTiledLinearOneProgramTwoShapes =
    test("Kernels/simdTiledLinearOneProgramTwoShapes") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkTwoShapesThroughOneProgram<SimdTiledLinear>();
};

auto tSimdTiledLinearFoldsGeluAndResidual =
    test("Kernels/simdTiledLinearFoldsGeluAndResidual") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkGeluAndResidual<SimdTiledLinear>(40, 24, 36, 800u);
    TiledProduct::checkGeluAndResidual<SimdTiledLinear>(100, 45, 76, 810u);
};

// The two programs against each other rather than against the reference: the
// same shape through both has to come out the same, which is what says the
// switch between them is a change of kernel and not of answer.
auto tSimdTiledMatchesRegisterTiled =
    test("Kernels/simdTiledMatchesRegisterTiled") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto rows = 100;
    constexpr auto inner = 45;
    constexpr auto columns = 76;

    auto shape = TiledMatMulShape::forLinear(rows, inner, columns);
    auto a = spreadValues(rows * inner, 900u, 2.f);
    auto b = spreadValues(columns * inner, 901u, 3.f);
    auto bias = spreadValues(columns, 902u, 1.f);

    auto registerTiled = TiledLinear {};
    auto simdTiled = SimdTiledLinear {};

    auto expected = TiledProduct::run(
        registerTiled, a, storageOf(b), bias, shape, rows * columns);
    auto result =
        TiledProduct::run(simdTiled, a, storageOf(b), bias, shape, rows * columns);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
};

#endif
