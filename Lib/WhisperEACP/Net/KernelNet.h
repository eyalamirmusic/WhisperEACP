#pragma once

#include "Net.h"

#include <WhisperEACP/Kernels/Kernels.h>

#include <optional>

namespace WSP
{
// Which of the tree's two tunings a kernel net runs: the encoder's many-row
// layer norm, tiled projections at every row count and the attention that folds
// its softmax into the apply, or the decoder's one-row layer norm, split
// projections for a step's few rows and the attention over a cache.
enum class KernelProfile
{
    encoder,
    decoder
};

struct ScratchReservation
{
    int elements = 0;
    int count = 0;
};

struct AttentionReservation
{
    int queryCapacity = 0;
    int keyCapacity = 0;
};

// Everything a recording binds that is not the caller's, sized once so a
// recording allocates nothing: the intermediates, by how many of each capacity
// are alive at once; the unfolded convolution windows; a score buffer per key
// capacity attended over; and the zeroes a projection with no bias binds.
struct KernelNetLayout
{
    int heads = 0;
    int headWidth = 0;
    int widestLayerOutput = 0;
    int columnElements = 0;
    EA::StaticVector<ScratchReservation, 4> scratch;
    EA::StaticVector<AttentionReservation, 2> attention;
    EA::StaticVector<int, 2> zeroBiasLengths;
};

// The kernel backend: each op is the dispatches the tree ran before the seam
// existed, recorded into the pass given to begin().
//
// A value lives in a scratch buffer of the capacity it could reach, the
// lowest-numbered free one, and goes back when its last Tensor does. The
// barriers are derived rather than written: a dispatch that reads a buffer
// written since the last barrier, or writes one read or written since it, is
// preceded by one. That puts back exactly the ordering the recordings spelled
// out by hand, the three projections a self-attention opens with running
// barrier-free included, since they read one buffer and write three others.
//
// An op is dispatched when the next one is called, or at end(), so output()
// can still point the op that made a value at the caller's buffer; that is how
// the encoder's last layer norm lands in the rows the decoder reads, with no
// copy after it.
class KernelNet final : public Net
{
public:
    explicit KernelNet(KernelProfile profileToUse);

    void prepare(eacp::GPU::Device& device, const KernelNetLayout& layoutToUse);

    void begin(eacp::GPU::ComputePass& passToUse);
    void end();

    Tensor input(const Binding& source, const Shape& shape, DType type) override;
    void output(const Tensor& value, const Binding& target) override;

    Tensor rows(const Weight& table, int first, int count) override;
    Tensor transpose(const Tensor& matrix) override;
    Tensor cached(const Cache& cache) override;

    Tensor conv1d(const Tensor& frames,
                  const Weight& weight,
                  const Weight& bias,
                  int stride,
                  int padding,
                  Activation activation) override;

    Tensor linear(const Tensor& input,
                  const Weight& weight,
                  const Weight* bias,
                  Activation activation) override;

    Tensor linearAdd(const Tensor& input,
                     const Weight& weight,
                     const Weight* bias,
                     const Tensor& stream) override;

    Tensor add(const Tensor& stream, const Tensor& addend) override;

    Tensor layerNorm(const Tensor& input,
                     const Weight& weight,
                     const Weight& bias) override;

    Tensor attention(const Tensor& queries,
                     const Tensor& keys,
                     const Tensor& values,
                     int heads,
                     bool causal) override;

    Tensor embed(const Tensor& tokens,
                 const Weight& tokenTable,
                 const Weight& positionTable,
                 int firstPosition) override;

    Tensor appendLinear(Cache& cache,
                        const Tensor& input,
                        const Weight& weight,
                        const Weight* bias) override;

    Cache makeCache(int capacityRows, int width) override;

    // How many lanes share one output's inner sum in the projections a step
    // takes. A 384-wide input row is 96 float4s, so at 96 every lane reads
    // exactly one and none idles; fc2's 1536 are four each. At the stock 8 a
    // 384-wide projection dispatched 48 groups, which is a few thousand
    // threads on a device that wants tens of thousands: measured against 8,
    // 16, 32, 64, 128, 192, 256 and 384, and 96 is the floor of that curve on
    // every one of the three shapes.
    static constexpr auto stepSplitCount = 96;

    // The logits projection wants far fewer. Its 51864 outputs fill the device
    // at any split count, so the only thing more lanes buy is a longer fold
    // and a thread that reads eight bytes — 64 measured 4% slower than 32 over
    // the whole decode, 128 12% slower.
    //
    // The count is the **shape's** rather than the weight's, which is what
    // keeps the float and packed logits bit identical: the two read the same
    // matrix at two widths, and they sum it in the same order only if they are
    // dispatched at the same split. Decoder/TinyEn/packedLogitsWeightIsIdentical
    // is what says so.
    static constexpr auto logitsSplitCount = 32;

    // Up to how many rows the decoder profile splits a projection's inner sum
    // across a group rather than tiling the product. A step is one row and a
    // prompt a few; the cross-attention projections over the encoder's 1500
    // are the other kind.
    static constexpr auto splitRowLimit = 16;

protected:
    void retain(int index) override;
    void release(int index) override;

private:
    struct Record
    {
        eacp::GPU::BufferRange storage;
        Shape shape;
        Shape capacity;
        int rowStride = 0;
        int columnStride = 1;
        int slot = -1;
        int base = -1;
        int references = 0;
        bool live = false;
    };

    enum class OpKind
    {
        none,
        convolution,
        linear,
        linearAdd,
        add,
        layerNorm,
        attention,
        embed,
        append
    };

    struct PendingOp
    {
        OpKind kind = OpKind::none;
        int result = -1;
        int first = -1;
        int second = -1;
        int third = -1;
        const Weight* weight = nullptr;
        const Weight* bias = nullptr;
        const Weight* positionTable = nullptr;
        int stride = 0;
        int padding = 0;
        int heads = 0;
        int position = 0;
        Activation activation = Activation::none;
        bool causal = false;
        eacp::GPU::BufferRange target;
    };

    struct ScratchSlot
    {
        eacp::GPU::Buffer buffer;
        int elements = 0;
        int holder = -1;
    };

    struct CacheStorage
    {
        eacp::GPU::Buffer buffer;
        int capacity = 0;
        int width = 0;
    };

    struct ZeroBias
    {
        eacp::GPU::Buffer buffer;
        int length = 0;
    };

    struct ScoreScratch
    {
        int keyCapacity = 0;
        eacp::GPU::Buffer scores;
        std::optional<eacp::GPU::Buffer> tileMaxima {};
    };

    using BufferKey = const eacp::GPU::Buffer*;

    static constexpr auto maxRecords = 64;
    static constexpr auto maxTracked = 16;

    int makeRecord();
    Tensor view(const Record& source);
    Tensor fresh(const Shape& shape, const Shape& capacity);
    int takeSlot(int elements);
    Record& recordOf(const Tensor& value);

    void flush();
    void run(const PendingOp& op);
    static bool makesNewValue(OpKind kind);

    void runConvolution(const PendingOp& op);
    void runProjection(const PendingOp& op);
    void runAdd(const PendingOp& op);
    void runLayerNorm(const PendingOp& op);
    void runAttention(const PendingOp& op);
    void runWindowAttention(const PendingOp& op);
    void runCachedAttention(const PendingOp& op);
    void runEmbed(const PendingOp& op);

    void dispatchProjection(const eacp::GPU::BufferRange& input,
                            int rowCount,
                            int innerCount,
                            const Weight& weight,
                            const eacp::GPU::Buffer& bias,
                            const eacp::GPU::BufferRange& target,
                            bool gelu,
                            bool residual);

    void order(std::initializer_list<BufferKey> reads,
               std::initializer_list<BufferKey> writes);

    bool splitsProjection(int rowCount) const;
    const eacp::GPU::Buffer& zeroBiasFor(int length) const;
    const ScoreScratch& scoresFor(int keyCapacity) const;
    eacp::GPU::ComputePass& activePass();

    KernelProfile profile;
    KernelNetLayout layout;
    eacp::GPU::Device* gpu = nullptr;
    eacp::GPU::ComputePass* recordingPass = nullptr;

    Vector<Record> records;
    PendingOp pending;
    EA::StaticVector<BufferKey, maxTracked> readSinceBarrier;
    EA::StaticVector<BufferKey, maxTracked> writtenSinceBarrier;

    Vector<ScratchSlot> slots;
    Vector<CacheStorage> caches;
    Vector<ZeroBias> zeroBiases;
    Vector<ScoreScratch> scoreScratch;
    std::optional<eacp::GPU::Buffer> columns;

    Unfold unfold;
    Add sum;
    LayerNorm normalisation;
    Embed embedding;

    LinearProduct projection;
    HalfWeightLinearProduct packedProjection;
    SplitLinear splitProjection {stepSplitCount};
    HalfWeightSplitLinear packedSplitProjection {stepSplitCount};
    SplitLinear splitLogits {logitsSplitCount};
    HalfWeightSplitLinear packedSplitLogits {logitsSplitCount};

    // A window's scores are written once and read once. The scores product
    // reports the largest score in each row of each of its column tiles as it
    // stores them, and the apply folds a row's tiles into the maximum it
    // exponentiates against, sums what it stages and divides the row by that
    // sum on the store, so the softmax between the two costs no pass.
    AttentionScoresProduct windowScores;
    SoftmaxAttentionApplyProduct windowApply;

    AttentionScores cachedScores;
    Softmax softmax;
    AttentionApply cachedApply;
    SingleQueryAttention singleQueryAttention;
};
} // namespace WSP
