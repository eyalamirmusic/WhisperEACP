#pragma once

#include "Net.h"

#include <eacp/ML/Graph/Graph.h>
#include <eacp/ML/MIL/Package.h>

#include <functional>
#include <optional>
#include <string>

namespace WSP
{
// Greedy selection inside a decoder step: the last row of the output recorded
// under logits, plus the additive fp16 row fed as suppression (0 where a token
// may be sampled, a large negative number where it may not), reduced to the
// index of its largest entry and put out as token, int32 [1]. The mask is an
// input rather than a constant because which tokens are suppressed at which
// step is the generation config's, which is Whisper's to hold.
//
// onTheEngine spells the argmax out of reductions, comparisons and one product
// rather than as MIL's reduce_argmax, which Core ML places on the CPU: the
// scores laid out a radix to a row, the lowest row holding the maximum picked
// out, then the lowest column of it. Every value in it is an fp16 integer
// below 2048 or a score compared for equality, so it is exact, and a tie goes
// to the lowest index as reduce_argmax's does. The token then comes out as
// fp16 [1, 2], its two digits in CoreMLNet::tokenRadix, most significant
// first.
struct CoreMLSelection
{
    std::string logits = "logits";
    std::string suppression = "suppression";
    std::string token = "next_token";
    bool onTheEngine = false;
};

struct CoreMLNetOptions
{
    // scaled_dot_product_attention, which wants Core ML specification 9
    // (macOS 15 / iOS 18). Off, attention is two matmuls around a softmax,
    // which specification 8 loads.
    bool fusedAttention = true;

    // How many input frames make one position: the encoder's second
    // convolution has a stride of two, so a context of n positions reads 2n
    // frames of each band.
    int framesPerPosition = 2;

    // A decoder program's: outputs the body records under these names are
    // left out of the program, so a step that only has to say which token
    // comes next does not write its hidden rows and its logits out.
    Vector<std::string> omittedOutputs;

    std::optional<CoreMLSelection> selection;

    // A decoder step's: the token, and apart from it the position, looked up
    // as a one-hot row multiplied into its table rather than gathered, since
    // Core ML keeps a gather on the CPU. The token is then tokenDigitsFeature,
    // fp16 [1, 2], its two digits in tokenRadix, and the position
    // positionFeature as fp16 [1, 1]: integers fp16 holds exactly, which a
    // whole vocabulary index is not. The product reads the whole table, which
    // for the token is the vocabulary's 40 MB a step.
    bool embedsTokenByProduct = false;
    bool embedsPositionByProduct = false;

    // A decoder program that opens a sequence: recordSequenceStart and the
    // prompt's recordDecoderStep recorded into one program. A cache the start
    // fills is read back inside the program rather than as an input, and every
    // other cache begins empty, so the prompt's rows are all a self cache
    // holds: attended causally among themselves, with no self mask, and put out
    // as appendedRowsFeature. The prompt's positions are the ones the body
    // records with, rows of the table rather than the position input.
    bool opensSequence = false;

    // An encoder program's: the mel input as [1, frames, bands] rather than
    // [1, bands, frames], and transposed inside, which is the layout a Core ML
    // mel (CoreMLMel.h) hands a context's frames over in.
    bool framesMajorInput = false;
};

// The graph backend: each op is recorded into an eacp::ML::Graph that is
// compiled once, ahead of time, into a Core ML model. Nothing here touches a
// device; the net is built on every platform, and only the runner that loads
// what build() returns is Apple's.
//
// It records one of two kinds of program.
//
// The encoder's serves every context in the position set given at
// construction: the input is declared as enumerated shapes, one per member,
// and the body records with the window's extents, which the backend reads
// only for the axes that do not follow from the context. Its layout is its
// own business: the band-major mel is already the [1, bands, frames] a
// convolution reads, so the one transpose the body asks for is a relabelling,
// and the convolutions' channels-first output is turned into [positions,
// width] rows before anything else reads it. What comes out is fp16 rows,
// positions by width, which the seam copy widens.
//
// A decoder's is fixed in shape, and a Core ML program keeps nothing between
// predictions, so a cache is a feature of its own, cacheFeature(), and what
// happens to one depends on how much is appended to it. A projection filling
// the whole capacity writes the cache out: that is recordSequenceStart's cross
// keys and values. A shorter one reads the cache in as an input, appends the
// new rows after all capacity rows of it, and puts the new rows out as
// appendedRowsFeature(), for the caller to write into the cache at the
// position the step took; which of the capacity rows are valid is the run
// time's to say, through selfMaskFeature, [1, capacity] fp16, 1 for a row
// that holds an earlier token and 0 for the rest. cached() reads a cache
// whole under crossMaskFeature, the same for the encoder rows a sequence
// attends to. A step embeds one token, and the position it takes is the
// positionFeature input rather than the firstPosition the body records with,
// since one program serves every position. A program that opens a sequence
// (CoreMLNetOptions::opensSequence) is the start and the prompt in one: the
// cross caches it fills feed its own attention, and the prompt's tokens all
// go in at once. The masks reach the fused attention as its bool mask (eacp's
// G9), or an additive one in front of the softmax without it.
//
// A Tensor is a MIL value, and MIL values are immutable, so there is nothing
// to reclaim when the last copy goes.
//
// An F32 weight goes into the blob through halfConstant, which narrows a value
// fp16 cannot hold exactly without saying so; tiny.en's are asserted exact.
class CoreMLNet final : public Net
{
public:
    // The encoder's program, over the contexts in positionsToUse.
    CoreMLNet(Span<const int> positionsToUse, const CoreMLNetOptions& optionsToUse);

    // A decoder's, every shape fixed.
    explicit CoreMLNet(const CoreMLNetOptions& optionsToUse);

    static constexpr auto positionFeature = "position";
    static constexpr auto selfMaskFeature = "self_mask";
    static constexpr auto crossMaskFeature = "cross_mask";
    static constexpr auto tokenDigitsFeature = "token_digits";

    // A token as two fp16 digits: 51864 is past the 2048 fp16 counts in
    // exactly, and 256 squared is not.
    static constexpr auto tokenRadix = 256;

    static std::string cacheFeature(const Cache& cache);
    static std::string appendedRowsFeature(const Cache& cache);

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

    const eacp::ML::Graph& graph() const { return program; }
    eacp::ML::Package build() const;

    bool isValid() const { return program.isValid(); }
    const Vector<std::string>& errors() const { return program.errors(); }

    bool isFixedShape() const { return positionSet.empty(); }
    Span<const int> positions() const { return positionSet; }
    const CoreMLNetOptions& options() const { return netOptions; }

    // The features the compiled model is addressed by, in the order the body
    // declared them.
    const Vector<std::string>& inputNames() const { return inputFeatures; }
    const Vector<std::string>& outputNames() const { return outputFeatures; }

protected:
    void retain(int index) override;
    void release(int index) override;

private:
    // What a net value is in the graph. The mel before the transpose and the
    // convolutions' outputs are channels first, [1, channels, frames], which
    // is the net's [frames, channels] seen the other way round; rows are the
    // net's own layout; a positional prefix is the table, cut to the rows of
    // whatever it is added to once it is. Tokens are an int32 input only
    // embed reads. An appended cache is a cache's capacity rows with a step's
    // rows after them, and a whole cache is one read as it is; both are rows
    // whose first maskedRows are valid only as far as a mask says.
    enum class Layout
    {
        bandMajorInput,
        channelsFirst,
        rows,
        positionalPrefix,
        tokens,
        appendedCache,
        wholeCache
    };

    struct Value
    {
        eacp::ML::Tensor tensor;
        Layout layout = Layout::rows;
        int maskedRows = 0;
    };

    struct CacheInfo
    {
        int capacity = 0;
        int width = 0;
        std::optional<eacp::ML::Tensor> filled;
    };

    Tensor make(eacp::ML::Tensor tensor, Layout layout, int maskedRows = 0);
    const Value& valueOf(const Tensor& handle) const;

    eacp::ML::Tensor rowsOf(const Tensor& handle);
    int extentOf(eacp::ML::Tensor value, int rank, int axis) const;
    eacp::ML::Tensor weightOf(const Weight& weight);
    eacp::ML::Tensor sum(eacp::ML::Tensor stream, eacp::ML::Tensor addend);
    eacp::ML::Tensor activated(eacp::ML::Tensor value, Activation activation);

    eacp::ML::Tensor namedInput(const std::string& name,
                                const eacp::ML::Shape& shape,
                                eacp::ML::DType type);

    // An fp16 input with one shape per context, shapeFor(positions).
    eacp::ML::Tensor
        contextInput(const std::string& name,
                     const std::function<eacp::ML::Shape(int)>& shapeFor);
    int largestContext() const;
    void declareOutput(eacp::ML::Tensor value, const std::string& name);
    void recordSelection(eacp::ML::Tensor logits);
    eacp::ML::Tensor argmaxDigits(eacp::ML::Tensor scores);

    eacp::ML::Tensor counting(int count);
    eacp::ML::Tensor oneHot(eacp::ML::Tensor index, int count);
    eacp::ML::Tensor oneHotTokens(eacp::ML::Tensor digits, int vocabulary);

    void requireFixedShape(std::string_view op) const;
    CacheInfo& cacheInfo(const Cache& cache);
    eacp::ML::Tensor allowedKeys(const Value& keys, eacp::ML::Tensor queries);
    eacp::ML::Tensor causalMask(int rows);
    eacp::ML::Tensor
        positionRows(const Weight& positionTable, int firstPosition, int count);

    eacp::ML::Tensor splitHeads(eacp::ML::Tensor rowsToSplit, int heads);
    eacp::ML::Tensor mergeHeads(eacp::ML::Tensor perHead);
    eacp::ML::Tensor unfusedAttention(eacp::ML::Tensor queries,
                                      eacp::ML::Tensor keys,
                                      eacp::ML::Tensor values);
    eacp::ML::Tensor maskedAttention(eacp::ML::Tensor queries,
                                     eacp::ML::Tensor keys,
                                     eacp::ML::Tensor values,
                                     eacp::ML::Tensor allowed);

    static std::string
        featureName(const Binding& binding, std::string_view fallback, int index);
    static void requireUnbound(const Binding& binding);

    Vector<int> positionSet;
    CoreMLNetOptions netOptions;
    eacp::ML::Graph program;

    Vector<Value> recorded;
    EA::MapVector<std::string, eacp::ML::Tensor> constants;
    EA::MapVector<std::string, eacp::ML::Tensor> declaredInputs;
    EA::MapVector<std::string, eacp::ML::Tensor> masks;
    Vector<CacheInfo> caches;
    int unnamedConstants = 0;
    Vector<std::string> inputFeatures;
    Vector<std::string> outputFeatures;
};
} // namespace WSP
