#pragma once

#include <WhisperEACP/Encoder/Encoder.h>

#if EACP_HAS_COREML

#include <WhisperEACP/Mel/MelShape.h>
#include <WhisperEACP/Net/CoreMLNet.h>

#include <eacp/Core/Threads/Async.h>
#include <eacp/ML/Model/Model.h>

#include <optional>

namespace WSP
{
// The mel front end on Core ML (Net/CoreMLMel.h), as a model of its own in
// front of the encoder's: the window's samples in, the frames-major mel out,
// and a context's first 2n rows of it copied into the encoder's input.
//
// Two models rather than one because one will not run on the engine. The
// encoder's context is the shape of its enumerated input, and Core ML allows
// one enumerated input to a model; the samples beside it are then a fixed
// input of a flexible model, which Core ML's E5 runtime refuses ("tensor_buffer
// has known strides while the model has FlexibleShapeInfo") and runs on the
// CPU instead, every op of it. Carrying the context in the samples' own shape
// would need a fixed slice of an enumerated axis, which eacp's Graph::slice
// does not take. plan.md has both.
struct CoreMLFrontEnd
{
    MelShape shape;

    // melCount x binCount, row-major: PreprocessorConfig::melFilters.
    Vector<float> filterBank;
};

struct CoreMLEncoderOptions
{
    // Not Core ML's default of all: under all, Core ML puts the whole encoder
    // on the GPU, and only CPU and Neural Engine reaches the engine, which is
    // what leaves the GPU free for the decoder.
    eacp::ML::ComputeUnits units = eacp::ML::ComputeUnits::cpuAndNeuralEngine;

    // scaled_dot_product_attention where the OS loads specification 9, and
    // matmul, softmax and matmul where it stops at 8.
    bool fusedAttention = eacp::ML::supportsSpecification(9);

    // Empty is eacp::ML::defaultCacheDirectory().
    eacp::FilePath cacheDirectory;

    // The mel front end on Core ML too, under the same units: the window's
    // samples in, rather than the mel a Metal command buffer wrote, so an
    // encode puts nothing on the GPU. The encoder is then run through the
    // overloads that take samples, and those alone.
    std::optional<CoreMLFrontEnd> frontEnd;
};

// The encoder as one Core ML model, built at prepare by recording the shared
// body (recordEncoder) into a CoreMLNet and compiling it for every context in
// the position set, then run once per window.
//
// The seam is two copies. The mel goes in as the first 2n frames of each band
// of the kernels' [bands, window frames] fp32 buffer, narrowed to the fp16
// array of that context, after the GPU work that wrote it; the rows come out
// fp16 and are widened into the front of the caller's [positions, width] fp32
// buffer, which the decoder reads as it reads the kernel encoder's. The arrays
// are made the first time a context is used and kept, so a context costs its
// allocation once.
//
// The compile is cached (eacp::ML::Model), keyed on the program's bytes and
// on a hash of the weights' blob, so a machine builds one model per set of
// weights and every later prepare loads it where it lies. Under CPU and
// Neural Engine the first load of the eighteen-context program is the engine
// compiling every member, about 14 s; a warm one is milliseconds.
class CoreMLEncoder final
{
public:
    CoreMLEncoder(const EncoderShape& shapeToUse, Span<const int> positionsToUse);

    // Records, builds and loads the model. A graph the body could not record
    // is a ModelError naming the first thing that went wrong; a model Core ML
    // would not compile or load comes back as the Result, with its message.
    // The weights only have to live for the call: every byte they point at is
    // copied into the program. A prepare that throws or fails leaves the
    // encoder unprepared, whatever it held before. Refused while an
    // encodeAsync is pending.
    eacp::ML::Result prepare(const EncoderWeights& hostWeights,
                             const CoreMLEncoderOptions& options);

    // The same with the compile and the load on the model's queue, which is
    // what keeps a window alive through the engine's first compile. The graph
    // is recorded and built on the calling thread, the main one, so the
    // weights only have to live for the call here too, and a graph that could
    // not be recorded throws before anything is queued. Resolves on the main
    // thread with the load's Result, the encoder prepared when it is ok.
    // Until then isPreparing() is true and every other prepare, encode and
    // encodeAsync is a logic_error; destroying the encoder abandons the Async.
    eacp::Threads::Async<eacp::ML::Result>
        prepareAsync(const EncoderWeights& hostWeights,
                     const CoreMLEncoderOptions& options);

    bool isPreparing() const { return loading; }

    bool isPrepared() const
    {
        return prepared && model.isLoaded() && (!computesMel || melModel.isLoaded());
    }

    const EncoderShape& shape() const { return encoderShape; }
    Span<const int> positions() const { return positionSet; }

    // mel is the [melBins, inputFrames] fp32 band-major buffer the front end
    // writes, and rows receives positionCount rows of width fp32 at its
    // start. Blocking, on the caller's thread. A count outside the position
    // set, or a prediction Core ML refuses, is a ModelError; a rows buffer too
    // small for the count, or a context whose encodeAsync is still pending,
    // is a logic_error.
    void encode(const eacp::GPU::Buffer& mel,
                eacp::GPU::Buffer& rows,
                int positionCount);

    // The same with the prediction on the model's queue. The mel is copied in
    // before this returns; the rows are copied out on the main thread before
    // the Async resolves, so rows must outlive it. Called on the main thread.
    // One at a time per context, since each context has one pair of arrays.
    // Destroying the encoder abandons a pending Async: it never settles.
    eacp::Threads::Async<eacp::ML::Result> encodeAsync(const eacp::GPU::Buffer& mel,
                                                       eacp::GPU::Buffer& rows,
                                                       int positionCount);

    // The two above without the second copy: the rows stay fp16 in the
    // context's own array, rowsAt(positionCount), which is what a Core ML
    // decoder reads, so they never make the trip through the GPU's buffer.
    void encode(const eacp::GPU::Buffer& mel, int positionCount);
    eacp::Threads::Async<eacp::ML::Result> encodeAsync(const eacp::GPU::Buffer& mel,
                                                       int positionCount);

    // The four above for an encoder prepared with a front end: samples is the
    // whole window, MelShape's sampleCount floats, narrowed to fp16 into the
    // mel model's samples array before these return; the mel is predicted,
    // the context's frames copied row for row into the encoder's input, and
    // the encoder predicted. The async forms hop back to the main thread for
    // that copy. One encode at a time across every context, since the
    // contexts share the mel model's arrays. The mel buffer forms are a
    // logic_error on such an encoder, and these on one without.
    void encode(Span<const float> samples,
                eacp::GPU::Buffer& rows,
                int positionCount);
    void encode(Span<const float> samples, int positionCount);

    eacp::Threads::Async<eacp::ML::Result> encodeAsync(Span<const float> samples,
                                                       eacp::GPU::Buffer& rows,
                                                       int positionCount);
    eacp::Threads::Async<eacp::ML::Result> encodeAsync(Span<const float> samples,
                                                       int positionCount);

    bool includesMel() const { return computesMel; }

    // The last encode from samples: the mel model's prediction alone, and the
    // samples narrowed into its array.
    double lastMelPredictSeconds() const { return melSeconds; }
    double lastSampleCopySeconds() const { return sampleCopySeconds; }

    // Where Core ML placed the mel model's ops; empty without a front end.
    // Blocks while it builds the plan.
    eacp::ML::ComputePlan melComputePlan() const { return melModel.computePlan(); }

    // [positionCount, width] fp16, what the last encode at that context
    // wrote, valid until the next one there. A logic_error for a context not
    // yet encoded or with an encodeAsync pending.
    const eacp::ML::MultiArray& rowsAt(int positionCount) const;

    bool wasCacheHit() const { return model.wasCacheHit(); }
    eacp::ML::ComputeUnits units() const { return model.units(); }
    bool usesFusedAttention() const { return fusedAttention; }

    // Read once and kept. Under CPU and Neural Engine a read costs the engine
    // compile again, about 14 s however warm the cache is, so only a caller
    // that is going to print or assert it should ask.
    const eacp::ML::ComputePlan& computePlan();

    double lastLoadSeconds() const { return loadSeconds; }

    // The prediction alone, without either copy across the seam. After an
    // encodeAsync it is the Prediction's own predictSeconds, timed on the
    // model's queue, so neither the wait for the queue nor the hop back is in
    // it.
    double lastPredictSeconds() const { return predictSeconds; }

private:
    // mel is the context's input: the mel, or with a front end the frames
    // array whose shape alone says which context runs.
    struct Member
    {
        int positions = 0;
        eacp::ML::MultiArray mel;
        eacp::ML::MultiArray rows;
        bool inFlight = false;
    };

    struct ProgramFeatures
    {
        std::string inputName;
        std::string outputName;
        bool fusedAttention = false;
        std::optional<MelShape> melShape;
    };

    struct RecordedProgram
    {
        eacp::ML::Package package;
        eacp::ML::Options modelOptions;
        ProgramFeatures features;
        std::optional<eacp::ML::Package> melPackage;
    };

    RecordedProgram record(const EncoderWeights& hostWeights,
                           const CoreMLEncoderOptions& options);
    void adopt(const ProgramFeatures& features);

    Member& memberFor(int positionCount);
    Member& filledFromMel(const eacp::GPU::Buffer& mel, int positionCount);
    Member& filledFromSamples(Span<const float> samples, int positionCount);
    Member& melFromSamples(Span<const float> samples, int positionCount);
    void copyMelInto(Member& member);
    Member* findMember(int positionCount);
    eacp::ML::Inputs inputsFor(const Member& member) const;
    void predictInto(Member& member, eacp::GPU::Buffer* rows);
    eacp::Threads::Async<eacp::ML::Result> predictAsyncInto(Member& member,
                                                            eacp::GPU::Buffer* rows);
    eacp::Threads::Async<eacp::ML::Result>
        predictAsyncFromSamples(Member& member, eacp::GPU::Buffer* rows);
    eacp::ML::Result loadMel(const RecordedProgram& program);
    void requireRowsFit(const eacp::GPU::Buffer& rows, int positionCount) const;
    void finishAsync(int positionCount);
    bool isAnyInFlight() const;
    void unprepare();

    EncoderShape encoderShape;
    Vector<int> positionSet;
    Vector<Member> members;

    eacp::ML::Model model;
    bool prepared = false;
    bool loading = false;
    std::string inputName;
    std::string outputName;
    bool fusedAttention = false;
    std::optional<eacp::ML::ComputePlan> plan;

    eacp::ML::Model melModel;
    bool computesMel = false;
    MelShape melShape;
    eacp::ML::MultiArray samplesIn;
    eacp::ML::MultiArray melOut;

    double loadSeconds = 0.0;
    double predictSeconds = 0.0;
    double melSeconds = 0.0;
    double sampleCopySeconds = 0.0;
};

// A plan as the benchmark and DeviceInfo print it: one line of op counts per
// device, then one line per op type and device for the ops that are not on
// the Neural Engine, each line opening with indent and ending in a newline.
std::string describePlacement(const eacp::ML::ComputePlan& plan,
                              const std::string& indent);
} // namespace WSP

#endif
