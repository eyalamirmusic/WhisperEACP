#include "TiledProduct.h"

using namespace nano;
using namespace WSP;
using namespace eacp::GPU;

// The register-tiled product, against the scalar reference in TiledProduct.h.
// The SIMD-group product runs through the same checks in
// SimdTiledMatMulTests.cpp, at shapes chosen for its own tiling.

// Nothing here is a multiple of a tile, a slab or a group: a kernel that ran
// its tile past the edge, or counted a partial slab as a whole one, fails.
auto tTiledLinearOddShape = test("Kernels/tiledLinearOddShape") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkLinear<TiledLinear>(5, 7, 3, 100u);
};

// Every extent crosses a tile boundary and none lands on one, and the inner
// count is not a multiple of the slab.
auto tTiledLinearAcrossTiles = test("Kernels/tiledLinearAcrossTiles") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkLinear<TiledLinear>(70, 50, 45, 200u);
};

// The model's own projection shape, every extent a multiple of everything.
auto tTiledLinearModelShape = test("Kernels/tiledLinearModelShape") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkLinear<TiledLinear>(96, 64, 128, 300u);
};

auto tTiledLinearPackedWeights = test("Kernels/tiledLinearPackedWeights") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkPackedLinear<HalfWeightTiledLinear>(37, 21, 35, 400u);
};

// Attention scores as the encoder and decoder compute them — 35 queries over
// three heads against a cache of 39 keys, scaled and causally masked over the
// trapezoid that leaves, where query 0 stands at position 4.
auto tTiledAttentionScores = test("Kernels/tiledAttentionScores") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkAttentionScores<TiledLinear>(35, 39, 3, 5, 500u);
};

// The register-tiled pair's softmax, which is the same assertion at that
// kernel's own tile and part counts.
auto tTiledSoftmaxAttention = test("Kernels/tiledSoftmaxAttention") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkSoftmaxAttention<MaximaTiledLinear, SoftmaxTiledMatMul>(
        41, 37, 2, 9, 700u);
};

auto tTiledSoftmaxAttentionModelHeads =
    test("Kernels/tiledSoftmaxAttentionModelHeads") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkSoftmaxAttention<MaximaTiledLinear, SoftmaxTiledMatMul>(
        70, 150, 2, 64, 710u);
};

auto tTiledAttentionApply = test("Kernels/tiledAttentionApply") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkAttentionApply<TiledMatMul>(41, 37, 2, 9, 600u);
};

auto tTiledLinearOneProgramTwoShapes =
    test("Kernels/tiledLinearOneProgramTwoShapes") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkTwoShapesThroughOneProgram<TiledLinear>();
};

auto tTiledLinearFoldsGeluAndResidual =
    test("Kernels/tiledLinearFoldsGeluAndResidual") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkGeluAndResidual<TiledLinear>(40, 24, 36, 800u);
};
