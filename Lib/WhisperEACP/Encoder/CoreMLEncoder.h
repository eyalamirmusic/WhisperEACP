#pragma once

#include <WhisperEACP/Encoder/Encoder.h>

#if EACP_HAS_COREML

#include <WhisperEACP/Net/CoreMLNet.h>

#include <eacp/Core/Threads/Async.h>
#include <eacp/ML/Model/Model.h>

#include <optional>

namespace WSP
{
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

    bool isPrepared() const { return prepared && model.isLoaded(); }

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
    struct Member
    {
        int positions = 0;
        eacp::ML::MultiArray mel;
        eacp::ML::MultiArray rows;
        bool inFlight = false;
    };

    Member& memberFor(int positionCount);
    void fillInput(Member& member, const eacp::GPU::Buffer& mel);
    void requireRowsFit(const eacp::GPU::Buffer& rows, int positionCount) const;
    void finishAsync(int positionCount);
    bool isAnyInFlight() const;
    void unprepare();

    EncoderShape encoderShape;
    Vector<int> positionSet;
    Vector<Member> members;

    eacp::ML::Model model;
    bool prepared = false;
    std::string inputName;
    std::string outputName;
    bool fusedAttention = false;
    std::optional<eacp::ML::ComputePlan> plan;

    double loadSeconds = 0.0;
    double predictSeconds = 0.0;
};

// A plan as the benchmark and DeviceInfo print it: one line of op counts per
// device, then one line per op type and device for the ops that are not on
// the Neural Engine, each line opening with indent and ending in a newline.
std::string describePlacement(const eacp::ML::ComputePlan& plan,
                              const std::string& indent);
} // namespace WSP

#endif
