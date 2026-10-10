#pragma once

#include <WhisperEACP/Decoder/Decoder.h>

#if EACP_HAS_COREML

#include <WhisperEACP/Net/CoreMLNet.h>

#include <eacp/Core/Threads/Async.h>
#include <eacp/ML/Model/Model.h>

namespace WSP
{
struct CoreMLDecoderOptions
{
    // The CPU, because that is where a decode step is fastest: a step is a
    // single row through every projection, which leaves the engine and the
    // GPU waiting on the hand-over more than computing (plan.md's decoder
    // measurement).
    eacp::ML::ComputeUnits units = eacp::ML::ComputeUnits::cpu;

    // The two masked attentions as scaled_dot_product_attention under a bool
    // mask from the inputs (eacp's G9, specification 9), rather than matmul,
    // an additive mask, softmax and matmul. Off, because it measured slower
    // where the decoder is fastest: a tiny.en step took 2.24 ms fused against
    // 0.68 ms unfused on the CPU and 2.06 against 1.04 ms under CPU and
    // engine, and the same either way on the GPU settings (Debug build, a
    // 200-step median). The unfused program also loads on specification 8.
    bool fusedAttention = false;

    // Empty is eacp::ML::defaultCacheDirectory().
    eacp::FilePath cacheDirectory;

    // Keeps a step's logits as an output beside its token, for a caller that
    // compares them. Off, the token is the only thing a step writes out but
    // the new cache rows, and the logits row never leaves Core ML.
    bool keepsLogits = false;

    // The step's token and position embedded by one-hot products rather than
    // gathered, and its argmax spelled out of reductions rather than MIL's
    // reduce_argmax (CoreMLNetOptions::embedsTokenByProduct and
    // embedsPositionByProduct, CoreMLSelection::onTheEngine): the three ops
    // Core ML otherwise places on the CPU, so with all three the step is the
    // engine's alone under cpuAndNeuralEngine. All off by default, the
    // decoder's default units being the CPU's.
    bool embedsTokenByProduct = false;
    bool embedsPositionByProduct = false;
    bool selectsOnTheEngine = false;

    // How many tokens a sequence opens with, all of them in the sequence
    // start's one prediction: tiny.en's prompt, decoder_start and
    // notimestamps, is two.
    int promptLength = 1;
};

// The decoder as two Core ML models, recorded from the same two bodies the
// kernel Decoder runs (recordSequenceStart and recordDecoderStep) into
// fixed-shape CoreMLNets.
//
// The sequence start projects the cross-attention keys and values of every
// layer out of the whole window of encoder rows once, straight into the
// arrays every step reads them from, and in the same prediction decodes the
// prompt (CoreMLNetOptions::opensSequence): its tokens at once, attending
// causally among themselves and over the cross keys it has just projected,
// and gives back the token that follows it with the prompt's key and value
// rows. A step embeds one token at a position given at run time, attends over
// the self caches, fixed at maxPositions rows behind a mask, and over the
// cross caches behind a second one, and puts out the token its logits choose
// once the caller's suppression row is added (eacp's argmax, G8), with the
// key and value rows it made. Those rows are written into the caches here
// before the next step; nothing but them and the token crosses back.
//
// Main thread only, like everything that touches the runtime; the blocking
// forms predict on the caller's thread.
class CoreMLDecoder final
{
public:
    explicit CoreMLDecoder(const DecoderShape& shapeToUse);

    // Records, builds and loads both models. A graph the bodies could not
    // record is a ModelError; a model Core ML would not compile or load comes
    // back as the Result. The weights only have to live for the call, and are
    // host weights (WeightPlacement::Host). A prepare that throws or fails
    // leaves the decoder unprepared.
    eacp::ML::Result prepare(const DecoderWeights& hostWeights,
                             const CoreMLDecoderOptions& options);

    // The same with both loads on the models' queues, resolved on the main
    // thread. Until then isPreparing() is true and everything else is a
    // logic_error; destroying the decoder abandons the Async.
    eacp::Threads::Async<eacp::ML::Result>
        prepareAsync(const DecoderWeights& hostWeights,
                     const CoreMLDecoderOptions& options);

    bool isPreparing() const { return loading; }
    bool isPrepared() const;

    const DecoderShape& shape() const { return decoderShape; }

    // What a suppressed token's logit is moved by: far enough that it never
    // wins, near enough that fp16 holds the sum.
    static constexpr float suppressedScore = -30000.0f;

    // The additive row a step's argmax reads, from a mask of Whisper's form:
    // nonzero at a token that may not be sampled.
    eacp::ML::MultiArray makeSuppression(Span<const float> mask) const;

    // Opens a sequence over the first crossPositionCount rows of the encoder's
    // output, the [crossPositions, width] fp32 buffer the kernels write or the
    // fp16 rows the Core ML encoder left in its array, with the prompt, and
    // returns the token that follows it under suppression. The rows past the
    // count are never attended to. A prompt of other than promptLength tokens
    // is a logic_error.
    int beginSequence(const eacp::GPU::Buffer& encoderRows,
                      int crossPositionCount,
                      Span<const int> prompt,
                      const eacp::ML::MultiArray& suppression);
    int beginSequence(const eacp::ML::MultiArray& encoderRows,
                      int crossPositionCount,
                      Span<const int> prompt,
                      const eacp::ML::MultiArray& suppression);

    // Appends token at position() and returns the token that follows it.
    // A step past maxPositions is a ModelError, one before beginSequence a
    // logic_error.
    int step(int token, const eacp::ML::MultiArray& suppression);

    int position() const { return decodedPositions; }
    int promptLength() const { return openingTokens; }

    // The last prediction's logits, the sequence start's or a step's,
    // [vocabularySize]; only under keepsLogits.
    Vector<float> lastLogits() const;

    bool wasCacheHit() const;
    eacp::ML::ComputeUnits units() const { return startModel.units(); }
    bool usesFusedAttention() const { return fusedAttention; }
    bool embedsTokenOnTheEngine() const { return embedsTokenByProduct; }
    bool embedsPositionOnTheEngine() const { return embedsPositionByProduct; }
    bool selectsTokenOnTheEngine() const { return selectsOnTheEngine; }

    double lastLoadSeconds() const { return loadSeconds; }

    // The sequence start's prediction, the prompt's included, and every
    // step's since it, summed.
    double lastSequenceStartSeconds() const { return sequenceStartSeconds; }
    double sequencePredictSeconds() const { return stepSeconds; }

    // Where Core ML placed each model's ops. Blocks while it builds the plan,
    // which under CPU and Neural Engine is the engine's compile again.
    eacp::ML::ComputePlan startComputePlan() const
    {
        return startModel.computePlan();
    }

    eacp::ML::ComputePlan stepComputePlan() const { return stepModel.computePlan(); }

private:
    struct Programs
    {
        eacp::ML::Package start;
        eacp::ML::Package step;
        eacp::ML::Options modelOptions;
        bool fusedAttention = false;
        bool keepsLogits = false;
        bool embedsTokenByProduct = false;
        bool embedsPositionByProduct = false;
        bool selectsOnTheEngine = false;
        int promptLength = 1;
    };

    Programs record(const DecoderWeights& hostWeights,
                    const CoreMLDecoderOptions& options);
    void adopt(const Programs& programs);
    void unprepare();
    void requireReady() const;
    int openSequence(int crossPositionCount,
                     Span<const int> prompt,
                     const eacp::ML::MultiArray& suppression);
    void writeTokens(eacp::ML::MultiArray& array, Span<const int> tokens) const;
    int chosenToken() const;

    DecoderShape decoderShape;
    DecoderCaches caches;

    eacp::ML::Model startModel;
    eacp::ML::Model stepModel;
    bool prepared = false;
    bool loading = false;
    bool fusedAttention = false;
    bool keepsLogits = false;
    bool embedsTokenByProduct = false;
    bool embedsPositionByProduct = false;
    bool selectsOnTheEngine = false;
    int openingTokens = 1;

    eacp::ML::MultiArray encoderRows;
    eacp::ML::MultiArray promptIn;
    eacp::ML::MultiArray tokenIn;
    eacp::ML::MultiArray positionIn;
    eacp::ML::MultiArray selfMask;
    eacp::ML::MultiArray crossMask;
    eacp::ML::MultiArray nextToken;
    eacp::ML::MultiArray logits;
    eacp::ML::MultiArray promptLogits;
    const eacp::ML::MultiArray* lastLogitRows = nullptr;
    Vector<eacp::ML::MultiArray> selfArrays;
    Vector<eacp::ML::MultiArray> promptRows;
    Vector<eacp::ML::MultiArray> appendedRows;

    eacp::ML::Inputs startInputs;
    eacp::ML::Outputs startOutputs;
    eacp::ML::Inputs stepInputs;
    eacp::ML::Outputs stepOutputs;

    Vector<float> allowedRows;
    int decodedPositions = 0;
    bool sequenceOpen = false;

    double loadSeconds = 0.0;
    double sequenceStartSeconds = 0.0;
    double stepSeconds = 0.0;
};
} // namespace WSP

#endif
