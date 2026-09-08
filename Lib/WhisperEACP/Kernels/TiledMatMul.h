#pragma once

#include "Attention.h"
#include "Gelu.h"
#include "KernelTypes.h"
#include "MatMul.h"

namespace WSP
{
// How the second operand lies in memory: which of its two indices is the
// contiguous one. nn.Linear's [out, in] weight and an attention's keys are
// contiguous along k; a row-major matrix and an attention's values along n.
enum class OperandLayout
{
    ContiguousK,
    ContiguousN
};

// One product the tiled kernel computes, batches of them at once:
//
//   C[b, m, n] = select(masked, causalMaskScore,
//                       scale * sum over k of A[b, m, k] * B[b, k, n] + bias[n])
//
// with the GELU applied to the result when gelu is set, and the result added
// to what C already holds when residual is set — the two stages either side
// of a projection, folded into its store rather than dispatched on their own —
// and with every operand addressed through a row stride and a batch stride,
// so a head's slice of a [positions, width] activation is an operand without
// being copied out of it:
//
//   A[b, m, k]   a[b * aBatchStride + m * aRowStride + k]
//   B[b, k, n]   b[b * bBatchStride + n * bStride + k]   ContiguousK
//                b[b * bBatchStride + k * bStride + n]   ContiguousN
//   C[b, m, n]   c[b * cBatchStride + m * cRowStride + n]
//
// The three named shapes are the three products a transformer layer is made
// of. A linear is one batch with the strides a row-major activation and an
// nn.Linear weight already have. Attention scores are one batch per head over
// the head's columns of the query and key rows, scaled, and causally masked on
// the store exactly as AttentionScores masks — key n later than query m, with
// query m standing at position keyCount - queryCount + m. Attention apply is
// one batch per head over the probabilities and the head's columns of the
// value rows, written into the head's columns of the output.
struct TiledMatMulShape
{
    int rows = 0;
    int columns = 0;
    int inner = 0;
    int batches = 1;
    int aRowStride = 0;
    int aBatchStride = 0;
    int bStride = 0;
    int bBatchStride = 0;
    int cRowStride = 0;
    int cBatchStride = 0;
    float scale = 1.f;
    bool causal = false;
    bool gelu = false;
    bool residual = false;

    static TiledMatMulShape forLinear(int rows, int inner, int columns)
    {
        auto shape = TiledMatMulShape {};
        shape.rows = rows;
        shape.columns = columns;
        shape.inner = inner;
        shape.aRowStride = inner;
        shape.bStride = inner;
        shape.cRowStride = columns;
        return shape;
    }

    static TiledMatMulShape forAttentionScores(int queryCount,
                                               int keyCount,
                                               int headCount,
                                               int headWidth,
                                               int modelWidth,
                                               float scale,
                                               bool causal)
    {
        auto shape = TiledMatMulShape {};
        shape.rows = queryCount;
        shape.columns = keyCount;
        shape.inner = headWidth;
        shape.batches = headCount;
        shape.aRowStride = modelWidth;
        shape.aBatchStride = headWidth;
        shape.bStride = modelWidth;
        shape.bBatchStride = headWidth;
        shape.cRowStride = keyCount;
        shape.cBatchStride = queryCount * keyCount;
        shape.scale = scale;
        shape.causal = causal;
        return shape;
    }

    static TiledMatMulShape forAttentionApply(
        int queryCount, int keyCount, int headCount, int headWidth, int modelWidth)
    {
        auto shape = TiledMatMulShape {};
        shape.rows = queryCount;
        shape.columns = headWidth;
        shape.inner = keyCount;
        shape.batches = headCount;
        shape.aRowStride = keyCount;
        shape.aBatchStride = queryCount * keyCount;
        shape.bStride = modelWidth;
        shape.bBatchStride = headWidth;
        shape.cRowStride = modelWidth;
        shape.cBatchStride = headWidth;
        return shape;
    }
};

// The product above, computed in 32 x 32 tiles of C by groups of 64 threads,
// each thread holding a 4 x 4 block of the tile in registers. The inner
// dimension goes by in slabs of sixteen: the group loads a 32 x 16 slab of A
// and a 16 x 32 slab of B into shared memory, every element of both read from
// global memory once per tile rather than once per output, and each thread
// multiplies its four rows of the one against its four columns of the other.
// That is the arithmetic intensity a thread per output cannot have, and it is
// where the encoder's time was.
//
// dispatch(pass, shape): the grid is one group per tile per batch, the batch
// and the row tile folded into the height, and the group is the fixed 8 x 8 so
// the loads and the barriers between them are whole-group. A row or column
// past the shape's edge is computed from a clamped index and never stored; an
// inner index past the edge is loaded as zero, so a partial last slab
// contributes nothing.
//
// The bias buffer is always bound, for MatMul's reason, and is columns long:
// a product with none binds zeroes. The weight is the operand that can be
// fp16, for MatMul's reason too, and only in the ContiguousK layout a shipped
// weight has.
template <OperandLayout bLayout, WeightStorage bStorage>
struct TiledMatMulProgram final : ComputeProgram
{
    static constexpr auto tile = 32;
    static constexpr auto innerTile = 16;
    static constexpr auto block = 4;

    // The A slab's row pitch, one over the slab width so that the eight rows
    // a group reads at once land on eight different banks.
    static constexpr auto aPitch = innerTile + 1;

    TiledMatMulProgram() { compile(); }

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

        pass.dispatch(*this,
                      columnTiles * groupSize2D,
                      shape.batches * rowTiles * groupSize2D);
    }

    void define() override
    {
        constexpr auto tileWidth = (unsigned) tile;
        constexpr auto slab = (unsigned) innerTile;
        constexpr auto pitch = (unsigned) aPitch;

        auto local = localPosition();
        auto group = groupPosition();
        auto tx = local.x;
        auto ty = local.y;
        auto thread = ty * 8u + tx;

        auto rowTiles = (rowCount + (tileWidth - 1u)) / tileWidth;
        auto batch = group.y / rowTiles;
        auto m0 = (group.y % rowTiles) * tileWidth;
        auto n0 = group.x * tileWidth;

        auto aBase = batch * aBatchStride;
        auto bBase = batch * bBatchStride;
        auto cBase = batch * cBatchStride;

        auto aTile = shared<Float>(tile * aPitch);
        auto bTile = shared<Float>(innerTile * tile);

        auto zero = constant(0.f);
        auto zero4 = float4(zero, zero, zero, zero);
        auto acc0 = var(zero4);
        auto acc1 = var(zero4);
        auto acc2 = var(zero4);
        auto acc3 = var(zero4);
        Var<Float4>* accumulators[block] = {&acc0, &acc1, &acc2, &acc3};

        // A slab: the thread's row of the tile and which half of the slab it
        // fetches, eight consecutive inner elements. The B slab the same way
        // over columns when B is contiguous along k, and over one inner row's
        // thirty-two columns in four runs of eight when it is contiguous along
        // n. Either way adjacent threads fetch adjacent words.
        auto lane = thread / 2u;
        auto half = (thread % 2u) * 8u;
        auto aRow = aBase + min(m0 + lane, rowCount - 1u) * aRowStride;

        auto k0 = var(0u);

        loop(k0.get() < innerCount,
             [&]
             {
                 for (auto i = 0u; i < 8u; ++i)
                 {
                     auto k = k0.get() + half + i;
                     auto value = select(
                         k < innerCount, a[aRow + min(k, innerCount - 1u)], 0.f);

                     write(aTile, lane * pitch + half + i, value);
                 }

                 if constexpr (bLayout == OperandLayout::ContiguousK)
                 {
                     auto bRow = bBase + min(n0 + lane, columnCount - 1u) * bStride;

                     for (auto i = 0u; i < 8u; ++i)
                     {
                         auto k = k0.get() + half + i;
                         auto value = select(k < innerCount,
                                             weight(bRow + min(k, innerCount - 1u)),
                                             0.f);

                         write(bTile, (half + i) * tileWidth + lane, value);
                     }
                 }
                 else
                 {
                     auto slabRow = thread / 4u;
                     auto run = (thread % 4u) * 8u;
                     auto k = k0.get() + slabRow;
                     auto bRow = bBase + min(k, innerCount - 1u) * bStride;

                     for (auto i = 0u; i < 8u; ++i)
                     {
                         auto n = min(n0 + run + i, columnCount - 1u);
                         auto value = select(k < innerCount, weight(bRow + n), 0.f);

                         write(bTile, slabRow * tileWidth + run + i, value);
                     }
                 }

                 barrier();

                 for (auto kk = 0u; kk < slab; ++kk)
                 {
                     auto bColumns = kk * tileWidth + tx * 4u;
                     auto right = float4(bTile[bColumns],
                                         bTile[bColumns + 1u],
                                         bTile[bColumns + 2u],
                                         bTile[bColumns + 3u]);

                     for (auto i = 0u; i < (unsigned) block; ++i)
                         *accumulators[i] +=
                             right * aTile[(ty * 4u + i) * pitch + kk];
                 }

                 barrier();
                 k0 += slab;
             });

        for (auto i = 0u; i < (unsigned) block; ++i)
        {
            auto m = m0 + ty * 4u + i;

            ifThen(m < rowCount,
                   [&]
                   {
                       auto row = accumulators[i]->get();
                       Float parts[block] = {row.x(), row.y(), row.z(), row.w()};

                       for (auto j = 0u; j < (unsigned) block; ++j)
                       {
                           auto n = n0 + tx * 4u + j;

                           ifThen(n < columnCount,
                                  [&]
                                  {
                                      auto at = cBase + m * cRowStride + n;
                                      auto value = scale * parts[j] + bias[n];
                                      auto activated = select(
                                          gelu != 0u, exactGelu(value), value);
                                      auto carried =
                                          select(residual != 0u, output[at], 0.f);
                                      auto masked =
                                          causal != 0u
                                          && n + rowCount > m + columnCount;

                                      write(output,
                                            at,
                                            select(masked,
                                                   causalMaskScore,
                                                   activated + carried));
                                  });
                       }
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

// A linear and an attention's scores read B along k; an attention's apply
// reads it along n.
using TiledLinear =
    TiledMatMulProgram<OperandLayout::ContiguousK, WeightStorage::Float>;
using HalfWeightTiledLinear =
    TiledMatMulProgram<OperandLayout::ContiguousK, WeightStorage::PackedHalf>;
using TiledMatMul =
    TiledMatMulProgram<OperandLayout::ContiguousN, WeightStorage::Float>;
} // namespace WSP
