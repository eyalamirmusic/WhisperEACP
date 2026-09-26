#pragma once

#include "Net.h"

#include <eacp/ML/Graph/Graph.h>
#include <eacp/ML/MIL/Package.h>

#include <string>

namespace WSP
{
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
};

// The graph backend: each op is recorded into an eacp::ML::Graph that is
// compiled once, ahead of time, into a Core ML model. Nothing here touches a
// device; the net is built on every platform, and only the runner that loads
// what build() returns is Apple's.
//
// The program serves every context in the position set given at construction:
// the input is declared as enumerated shapes, one per member, and the body
// records with the window's extents, which the backend reads only for the
// axes that do not follow from the context. Its layout is its own business:
// the band-major mel is already the [1, bands, frames] a convolution reads, so
// the one transpose the body asks for is a relabelling, and the convolutions'
// channels-first output is turned into [positions, width] rows before
// anything else reads it. What comes out is fp16 rows, positions by width,
// which the seam copy widens.
//
// A Tensor is a MIL value, and MIL values are immutable, so there is nothing
// to reclaim when the last copy goes. The decoder's ops are not here yet.
//
// An F32 weight goes into the blob through halfConstant, which narrows a value
// fp16 cannot hold exactly without saying so; tiny.en's are asserted exact.
class CoreMLNet final : public Net
{
public:
    CoreMLNet(Span<const int> positionsToUse, const CoreMLNetOptions& optionsToUse);

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
    // whatever it is added to once it is.
    enum class Layout
    {
        bandMajorInput,
        channelsFirst,
        rows,
        positionalPrefix
    };

    struct Value
    {
        eacp::ML::Tensor tensor;
        Layout layout = Layout::rows;
    };

    Tensor make(eacp::ML::Tensor tensor, Layout layout);
    const Value& valueOf(const Tensor& handle) const;

    eacp::ML::Tensor rowsOf(const Tensor& handle);
    int extentOf(eacp::ML::Tensor value, int rank, int axis) const;
    eacp::ML::Tensor weightOf(const Weight& weight);
    eacp::ML::Tensor sum(eacp::ML::Tensor stream, eacp::ML::Tensor addend);
    eacp::ML::Tensor activated(eacp::ML::Tensor value, Activation activation);

    eacp::ML::Tensor splitHeads(eacp::ML::Tensor rowsToSplit, int heads);
    eacp::ML::Tensor mergeHeads(eacp::ML::Tensor perHead);
    eacp::ML::Tensor unfusedAttention(eacp::ML::Tensor queries,
                                      eacp::ML::Tensor keys,
                                      eacp::ML::Tensor values);

    static std::string
        featureName(const Binding& binding, std::string_view fallback, int index);
    static void requireUnbound(const Binding& binding);

    Vector<int> positionSet;
    CoreMLNetOptions netOptions;
    eacp::ML::Graph program;

    Vector<Value> recorded;
    EA::MapVector<std::string, eacp::ML::Tensor> constants;
    int unnamedConstants = 0;
    Vector<std::string> inputFeatures;
    Vector<std::string> outputFeatures;
};
} // namespace WSP
