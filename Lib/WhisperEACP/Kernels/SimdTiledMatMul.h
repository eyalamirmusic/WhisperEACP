#pragma once

#include "TiledMatMul.h"

// WHISPER_EACP_HAS_SIMD_MATRIX is set by CMake/Findeacp.cmake from the eacp
// this tree was configured against: the primitive is on a branch until it
// lands on develop, and an eacp without it has no simdMatrix() to compile.
// Without it the role aliases at the bottom all name the register-tiled
// product, which is what every non-Metal backend runs regardless — so the tree
// still builds and still transcribes, at the throughput it had before this.
#if defined(WHISPER_EACP_HAS_SIMD_MATRIX)

namespace WSP
{
using eacp::GPU::SimdMatrix;

// The same product TiledMatMulProgram computes — the same TiledMatMulShape, the
// same batch and stride semantics, the same fold on the store — out of
// SIMD-group matrices rather than out of a register block per thread.
//
// A threadgroup is 256 threads, which is eight SIMD groups, and it owns a
// 64 x 64 tile of C. The eight groups stand two deep and four across, so each
// holds 32 rows by 16 columns of the tile as eight 8 x 8 accumulator
// fragments; the inner dimension goes by in slabs of 32, staged into one
// threadgroup array as a 64 x 32 block of A and a 32 x 64 block of B, and each
// slab is four multiply-accumulates deep.
//
// The tile of C goes back out through the same array the slabs came in
// through, which is what lets a partial tile be copied out element by element:
// a fragment is loaded and stored whole and has no per-element guard to put on
// one. The store's fold — the scale, the bias, the GELU, the residual and the
// causal mask — happens in that copy-out, which every element passes through
// anyway.
//
// Three numbers this shape is chosen by, all measured on an M5 Max: sixteen
// accumulator fragments a SIMD group instead of eight collapses to under a
// TFLOPS, staging C back through threadgroup memory costs nothing, and
// branching around the fragment stores rather than storing them
// unconditionally halves the throughput.
template <OperandLayout bLayout, WeightStorage bStorage>
struct SimdTiledMatMulProgram final : ComputeProgram
{
    static constexpr auto tile = 64;
    static constexpr auto innerTile = 32;
    static constexpr auto threads = 256;
    static constexpr auto fragment = ComputeProgram::simdMatrixWidth;

    static constexpr auto rowFragments = 4;
    static constexpr auto columnFragments = 2;
    static constexpr auto tileElements = tile * innerTile + innerTile * tile;
    static constexpr auto columnSlabBase = tile * innerTile;

    // simdgroup_load wants the natural stride, so the A slab carries no
    // bank-conflict pitch the way the register-tiled kernel's does.
    SimdTiledMatMulProgram()
        : ComputeProgram({threads, 1, 1})
    {
        compile();
    }

    void dispatch(ComputePass& pass, const TiledMatMulShape& shape)
    {
        rowCount = (std::uint32_t) shape.rows;
        columnCount = (std::uint32_t) shape.columns;
        innerCount = (std::uint32_t) shape.inner;
        aRowStride = (std::uint32_t) shape.aRowStride;
        aBatchStride = (std::uint32_t) shape.aBatchStride;
        bStride = (std::uint32_t) shape.bStride;
        bBatchStride = (std::uint32_t) shape.bBatchStride;
        cRowStride = (std::uint32_t) shape.cRowStride;
        cBatchStride = (std::uint32_t) shape.cBatchStride;
        scale = shape.scale;
        causal = shape.causal ? 1u : 0u;
        gelu = shape.gelu ? 1u : 0u;
        residual = shape.residual ? 1u : 0u;

        const auto columnTiles = (shape.columns + tile - 1) / tile;
        const auto rowTiles = (shape.rows + tile - 1) / tile;

        pass.dispatch(*this, columnTiles * threads, shape.batches * rowTiles);
    }

    void define() override
    {
        constexpr auto tileWidth = (unsigned) tile;
        constexpr auto slab = (unsigned) innerTile;
        constexpr auto side = (unsigned) fragment;

        auto lane = localPosition().x;
        auto group = groupPosition();
        auto simd = simdGroupIndex();

        auto rowTiles = (rowCount + (tileWidth - 1u)) / tileWidth;
        auto batch = group.y / rowTiles;
        auto m0 = (group.y % rowTiles) * tileWidth;
        auto n0 = group.x * tileWidth;

        auto aBase = batch * aBatchStride;
        auto bBase = batch * bBatchStride;
        auto cBase = batch * cBatchStride;

        auto rowOffset = (simd % 2u) * 32u;
        auto columnOffset = (simd / 2u) * 16u;

        auto staging = shared<Float>(tileElements);
        auto slabStride = unsignedInteger(slab);
        auto tileStride = unsignedInteger(tileWidth);

        // Eight consecutive inner elements per thread, which is 256 threads
        // covering the 64 x 32 slab of A exactly, and the same count covering
        // the transposing store of B when B is contiguous along k.
        auto loadRow = lane / 4u;
        auto loadDepth = (lane % 4u) * 8u;

        auto aRow = aBase + min(m0 + loadRow, rowCount - 1u) * aRowStride;
        auto bRow = bBase + min(n0 + loadRow, columnCount - 1u) * bStride;

        SimdMatrix accumulators[rowFragments * columnFragments];

        for (auto& accumulator: accumulators)
            accumulator = simdMatrix();

        auto k0 = var(0u);

        loop(
            k0.get() < innerCount,
            [&]
            {
                if constexpr (bLayout == OperandLayout::ContiguousK)
                {
                    for (auto i = 0u; i < 8u; ++i)
                    {
                        auto k = k0.get() + loadDepth + i;
                        auto inside = k < innerCount;
                        auto at = min(k, innerCount - 1u);

                        write(staging,
                              loadRow * slab + loadDepth + i,
                              select(inside, a[aRow + at], 0.f));

                        write(staging,
                              columnSlabBase + (loadDepth + i) * tileWidth + loadRow,
                              select(inside, weight(bRow + at), 0.f));
                    }
                }
                else
                {
                    // B read along n instead: one inner row per eight threads,
                    // eight of its columns each, so adjacent threads still
                    // fetch adjacent words.
                    auto slabRow = lane / 8u;
                    auto run = (lane % 8u) * 8u;
                    auto slabK = k0.get() + slabRow;
                    auto slabInside = slabK < innerCount;
                    auto slabBRow = bBase + min(slabK, innerCount - 1u) * bStride;

                    for (auto i = 0u; i < 8u; ++i)
                    {
                        auto k = k0.get() + loadDepth + i;
                        auto inside = k < innerCount;
                        auto at = min(k, innerCount - 1u);

                        write(staging,
                              loadRow * slab + loadDepth + i,
                              select(inside, a[aRow + at], 0.f));

                        auto n = min(n0 + run + i, columnCount - 1u);

                        write(staging,
                              columnSlabBase + slabRow * tileWidth + run + i,
                              select(slabInside, weight(slabBRow + n), 0.f));
                    }
                }

                barrier();

                for (auto kk = 0u; kk < slab; kk += side)
                {
                    SimdMatrix left[rowFragments];
                    SimdMatrix right[columnFragments];

                    for (auto i = 0u; i < (unsigned) rowFragments; ++i)
                        left[i] = simdMatrix(
                            staging, (rowOffset + i * side) * slab + kk, slabStride);

                    for (auto j = 0u; j < (unsigned) columnFragments; ++j)
                        right[j] = simdMatrix(staging,
                                              columnSlabBase + kk * tileWidth
                                                  + columnOffset + j * side,
                                              tileStride);

                    for (auto i = 0u; i < (unsigned) rowFragments; ++i)
                        for (auto j = 0u; j < (unsigned) columnFragments; ++j)
                            multiplyAccumulate(accumulators[i * columnFragments + j],
                                               left[i],
                                               right[j]);
                }

                barrier();
                k0 += slab;
            });

        barrier();

        for (auto i = 0u; i < (unsigned) rowFragments; ++i)
            for (auto j = 0u; j < (unsigned) columnFragments; ++j)
                write(staging,
                      (rowOffset + i * side) * tileWidth + columnOffset + j * side,
                      tileStride,
                      accumulators[i * columnFragments + j]);

        barrier();

        for (auto i = 0u; i < (unsigned) (tile * tile / threads); ++i)
        {
            auto index = lane + i * (unsigned) threads;
            auto m = m0 + index / tileWidth;
            auto n = n0 + index % tileWidth;

            ifThen(m < rowCount && n < columnCount,
                   [&]
                   {
                       auto at = cBase + m * cRowStride + n;
                       auto value = scale * staging[index] + bias[n];
                       auto activated = select(gelu != 0u, exactGelu(value), value);
                       auto carried = select(residual != 0u, output[at], 0.f);
                       auto masked = causal != 0u && n + rowCount > m + columnCount;

                       write(output,
                             at,
                             select(masked, causalMaskScore, activated + carried));
                   });
        }
    }

    Float weight(const UInt& index)
    {
        if constexpr (bStorage == WeightStorage::PackedHalf)
            return b.readHalf(index);
        else
            return b[index];
    }

    Uniform<InputBuffer> a;
    Uniform<InputBuffer> b;
    Uniform<InputBuffer> bias;
    Uniform<OutputBuffer> output;
    Uniform<UInt> rowCount;
    Uniform<UInt> columnCount;
    Uniform<UInt> innerCount;
    Uniform<UInt> aRowStride;
    Uniform<UInt> aBatchStride;
    Uniform<UInt> bStride;
    Uniform<UInt> bBatchStride;
    Uniform<UInt> cRowStride;
    Uniform<UInt> cBatchStride;
    Uniform<Float> scale;
    Uniform<UInt> causal;
    Uniform<UInt> gelu;
    Uniform<UInt> residual;

    EACP_SHADER(a,
                b,
                bias,
                output,
                rowCount,
                columnCount,
                innerCount,
                aRowStride,
                aBatchStride,
                bStride,
                bBatchStride,
                cRowStride,
                cBatchStride,
                scale,
                causal,
                gelu,
                residual)
};

using SimdTiledLinear =
    SimdTiledMatMulProgram<OperandLayout::ContiguousK, WeightStorage::Float>;
using HalfWeightSimdTiledLinear =
    SimdTiledMatMulProgram<OperandLayout::ContiguousK, WeightStorage::PackedHalf>;
using SimdTiledMatMul =
    SimdTiledMatMulProgram<OperandLayout::ContiguousN, WeightStorage::Float>;
} // namespace WSP

#endif

namespace WSP
{
// Which program each of the tree's product roles dispatches through. Both
// compute the same TiledMatMulShape, so a role is switched by its alias and
// nothing at the call site moves.
//
// The SIMD-group form is a Metal specialisation. eacp lowers a fragment to 64
// floats of each thread's own wherever there is no wave matrix instruction —
// correct, and doing the arithmetic 32 times over — so the register-tiled form
// is what every other backend gets, and what a role goes back to if it ever
// measures no better on it. Every role here measured better; what each was
// worth is in plan.md's fourth performance round.
#if defined(__APPLE__) && defined(WHISPER_EACP_HAS_SIMD_MATRIX)
using LinearProduct = SimdTiledLinear;
using HalfWeightLinearProduct = HalfWeightSimdTiledLinear;
using AttentionScoresProduct = SimdTiledLinear;
using AttentionApplyProduct = SimdTiledMatMul;
using SpectrumProduct = SimdTiledLinear;
using CrossProjectionProduct = SimdTiledLinear;
using HalfWeightCrossProjectionProduct = HalfWeightSimdTiledLinear;
#else
using LinearProduct = TiledLinear;
using HalfWeightLinearProduct = HalfWeightTiledLinear;
using AttentionScoresProduct = TiledLinear;
using AttentionApplyProduct = TiledMatMul;
using SpectrumProduct = TiledLinear;
using CrossProjectionProduct = TiledLinear;
using HalfWeightCrossProjectionProduct = HalfWeightTiledLinear;
#endif
} // namespace WSP
