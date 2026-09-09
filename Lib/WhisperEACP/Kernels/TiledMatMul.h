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

// What the first operand goes through on its way into the product. Softmax
// reads A as a row of raw scores and exponentiates each element as it is
// staged, against the row maximum the product that wrote those scores left
// behind — and, because one group's inner loop walks the whole row, sums what
// it staged and divides the row by that sum on the store. So the softmax of an
// attention costs no pass of its own: the score matrix is written once and read
// once, where the normalising pass over it read it twice more and wrote it
// twice.
//
// A row of A is the softmax row: batch b, row m, is row b * rows + m, which for
// an attention's apply is head h, query i, exactly as the [heads, queries,
// keys] score buffer is indexed.
enum class AFold
{
    None,
    Softmax
};

// Whether the product also reports, for each row of each of its column tiles,
// the largest value it stored there — rowMaxima[(b * rows + m) * tiles + t],
// tiles being the product's own column tile count. Which is the maximum an
// AFold::Softmax reader of that same matrix needs, without the pass over it
// that finding one otherwise costs: the reader folds the tiles of its row.
//
// The maximum is of the value stored, fold and mask included, so a masked
// score is in it — which is what makes exp(masked - maximum) the zero
// AttentionScores intends. A part of a tile that lies entirely past the shape
// reports causalMaskScore, which is what an empty maximum is here for the same
// reason it is what a masked score is: below every real one, and a literal
// rather than a limit, so what the shader holds is what the host wrote.
enum class RowMaxima
{
    None,
    PerColumnTile
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
template <OperandLayout bLayout,
          WeightStorage bStorage,
          AFold aFold = AFold::None,
          RowMaxima cMaxima = RowMaxima::None>
struct TiledMatMulProgram final : ComputeProgram
{
    static constexpr auto tile = 32;
    static constexpr auto innerTile = 16;
    static constexpr auto block = 4;

    // SimdTiledMatMulProgram's, at this kernel's own group: two threads walk a
    // row of the tile, sixteen of its columns each, and each writes what it
    // saw.
    static constexpr auto maximaPerRow = 2;

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

        maximaStride = (std::uint32_t) (columnTiles * maximaPerRow);
        foldTiles = (std::uint32_t) ((shape.inner + tile - 1) / tile * maximaPerRow);

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
        auto stagedRow = min(m0 + lane, rowCount - 1u);
        auto aRow = aBase + stagedRow * aRowStride;

        // SimdTiledMatMulProgram's, and its note on why each staging thread
        // folds the row's maxima for itself.
        auto sums = shared<Float>(aFold == AFold::Softmax ? (unsigned) tile : 1u);

        auto foldMaximum = var(0.f);
        auto foldSum = var(0.f);

        if constexpr (aFold == AFold::Softmax)
        {
            auto maximaRow = (batch * rowCount + stagedRow) * foldTiles;
            auto largest = var(causalMaskScore);
            auto foldTile = var(0u);

            loop(foldTile.get() < foldTiles,
                 [&]
                 {
                     largest =
                         max(largest.get(), rowMaxima[maximaRow + foldTile.get()]);
                     foldTile += 1u;
                 });

            foldMaximum = largest.get();
        }

        auto k0 = var(0u);

        loop(k0.get() < innerCount,
             [&]
             {
                 for (auto i = 0u; i < 8u; ++i)
                 {
                     auto k = k0.get() + half + i;

                     stageA(aTile,
                            lane * pitch + half + i,
                            aRow + min(k, innerCount - 1u),
                            k < innerCount,
                            foldMaximum,
                            foldSum);
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

        // The two partial sums a row's two staging threads hold go through the
        // B slab, which is between the last product and the store.
        if constexpr (aFold == AFold::Softmax)
        {
            write(bTile, thread, foldSum.get());
            barrier();

            ifThen(thread < tileWidth,
                   [&]
                   {
                       write(sums,
                             thread,
                             1.f / (bTile[2u * thread] + bTile[2u * thread + 1u]));
                   });

            barrier();
        }

        // The tile of C goes back out through threadgroup memory when its row
        // maxima are wanted, since a thread's own 4 x 4 block is four rows by
        // four columns and a row's maximum is over the whole tile's width.
        auto cTile =
            shared<Float>(cMaxima == RowMaxima::PerColumnTile ? tile * tile : 1);

        if constexpr (cMaxima == RowMaxima::PerColumnTile)
        {
            for (auto i = 0u; i < (unsigned) block; ++i)
            {
                auto row = accumulators[i]->get();
                Float parts[block] = {row.x(), row.y(), row.z(), row.w()};

                for (auto j = 0u; j < (unsigned) block; ++j)
                    write(cTile, (ty * 4u + i) * tileWidth + tx * 4u + j, parts[j]);
            }

            barrier();

            constexpr auto groupThreads = (unsigned) (groupSize2D * groupSize2D);

            for (auto t = 0u; t < (unsigned) (tile * tile) / groupThreads; ++t)
            {
                auto index = thread + t * groupThreads;
                auto m = m0 + index / tileWidth;
                auto n = n0 + index % tileWidth;

                ifThen(m < rowCount && n < columnCount,
                       [&]
                       {
                           auto at = cBase + m * cRowStride + n;
                           auto stored = var(storedValue(
                               cTile[index], sums, index / tileWidth, m, n, at));

                           // Through the var, not the expression: a read
                           // handle re-materialises after a store to its slot,
                           // so a residual product's second evaluation would
                           // read what the first one just wrote.
                           write(output, at, stored.get());
                           write(cTile, index, stored.get());
                       });
            }

            barrier();

            constexpr auto columnsPerLane = (unsigned) (tile / maximaPerRow);

            auto maximaRow = thread / (unsigned) maximaPerRow;
            auto maximaFirst = (thread % (unsigned) maximaPerRow) * columnsPerLane;
            auto largest = var(causalMaskScore);

            for (auto i = 0u; i < columnsPerLane; ++i)
                largest = max(largest.get(),
                              select(n0 + maximaFirst + i < columnCount,
                                     cTile[maximaRow * tileWidth + maximaFirst + i],
                                     causalMaskScore));

            ifThen(m0 + maximaRow < rowCount,
                   [&]
                   {
                       auto at = (batch * rowCount + m0 + maximaRow) * maximaStride
                                 + group.x * (unsigned) maximaPerRow
                                 + thread % (unsigned) maximaPerRow;

                       write(tileMaxima, at, largest.get());
                   });

            return;
        }

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

                           ifThen(
                               n < columnCount,
                               [&]
                               {
                                   auto at = cBase + m * cRowStride + n;

                                   write(output,
                                         at,
                                         storedValue(
                                             parts[j], sums, ty * 4u + i, m, n, at));
                               });
                       }
                   });
        }
    }

    // The fold every store passes through: the softmax's row divisor where one
    // is being applied, then the scale, the bias, the GELU, the residual and
    // the causal mask.
    template <typename TileRow, typename Row, typename Column, typename At>
    Float storedValue(const Float& accumulated,
                      const Shared<Float>& sums,
                      const TileRow& tileRow,
                      const Row& m,
                      const Column& n,
                      const At& at)
    {
        auto value = scale * normalised(accumulated, sums, tileRow) + bias[n];
        auto activated = select(gelu != 0u, exactGelu(value), value);
        auto carried = select(residual != 0u, output[at], 0.f);
        auto masked = causal != 0u && n + rowCount > m + columnCount;

        return select(masked, causalMaskScore, activated + carried);
    }

    template <typename TileRow>
    Float normalised(const Float& accumulated,
                     const Shared<Float>& sums,
                     const TileRow& tileRow)
    {
        if constexpr (aFold == AFold::Softmax)
            return accumulated * sums[tileRow];
        else
            return accumulated;
    }

    // The A element as it is staged: exponentiated against the row maximum and
    // added into this thread's share of the row's sum when the softmax is
    // being folded in, and the element itself otherwise.
    template <typename Slot, typename Index, typename Inside>
    void stageA(const Shared<Float>& tile,
                const Slot& slot,
                const Index& index,
                const Inside& inside,
                const Var<Float>& maximum,
                Var<Float>& sum)
    {
        if constexpr (aFold == AFold::Softmax)
        {
            auto staged = var(select(inside, exp(a[index] - maximum.get()), 0.f));

            write(tile, slot, staged.get());
            sum += staged.get();
        }
        else
        {
            write(tile, slot, select(inside, a[index], 0.f));
        }
    }

    Float weight(const UInt& index)
    {
        if constexpr (bStorage == WeightStorage::PackedHalf)
            return b.readHalf(index);
        else
            return b[index];
    }

    // Written out rather than declared by EACP_SHADER because rowStats is only
    // a binding of the folding instantiation: a program that does not read it
    // does not name it, and so nothing that dispatches one has to bind it.
    void reflectMembers(eacp::GPU::ShaderVisitor& visitor) override
    {
        visitor("a", a);
        visitor("b", b);
        visitor("bias", bias);
        visitor("output", output);
        visitor("rowCount", rowCount);
        visitor("columnCount", columnCount);
        visitor("innerCount", innerCount);
        visitor("aRowStride", aRowStride);
        visitor("aBatchStride", aBatchStride);
        visitor("bStride", bStride);
        visitor("bBatchStride", bBatchStride);
        visitor("cRowStride", cRowStride);
        visitor("cBatchStride", cBatchStride);
        visitor("scale", scale);
        visitor("causal", causal);
        visitor("gelu", gelu);
        visitor("residual", residual);

        if constexpr (aFold == AFold::Softmax)
        {
            visitor("rowMaxima", rowMaxima);
            visitor("foldTiles", foldTiles);
        }

        if constexpr (cMaxima == RowMaxima::PerColumnTile)
        {
            visitor("tileMaxima", tileMaxima);
            visitor("maximaStride", maximaStride);
        }
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

    // The tile maxima a RowMaxima::PerColumnTile product writes, and the ones
    // an AFold::Softmax product folds. No instantiation is both.
    Uniform<OutputBuffer> tileMaxima;
    Uniform<InputBuffer> rowMaxima;
    Uniform<UInt> maximaStride;
    Uniform<UInt> foldTiles;
};

// A linear and an attention's scores read B along k; an attention's apply
// reads it along n, and reads it through the softmax when the row it is
// applying was left un-normalised.
using TiledLinear =
    TiledMatMulProgram<OperandLayout::ContiguousK, WeightStorage::Float>;
using HalfWeightTiledLinear =
    TiledMatMulProgram<OperandLayout::ContiguousK, WeightStorage::PackedHalf>;
using TiledMatMul =
    TiledMatMulProgram<OperandLayout::ContiguousN, WeightStorage::Float>;
using SoftmaxTiledMatMul = TiledMatMulProgram<OperandLayout::ContiguousN,
                                              WeightStorage::Float,
                                              AFold::Softmax>;
using MaximaTiledLinear = TiledMatMulProgram<OperandLayout::ContiguousK,
                                             WeightStorage::Float,
                                             AFold::None,
                                             RowMaxima::PerColumnTile>;
} // namespace WSP
