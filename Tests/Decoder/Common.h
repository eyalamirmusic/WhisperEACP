#pragma once

#include <WhisperEACP/Decoder/Decoder.h>
#include <WhisperEACP/Decoder/DecoderWeights.h>

// The encoder's suite rather than a second copy of what it holds: packHalf, the
// safetensors builder, the spread of inputs, the layer norm and linear
// references, and the error measure are all the same pieces on this side of the
// model. It brings Tests/Kernels/Common.h and Tests/Model/Common.h with it,
// which is why neither is named here.
//
// Nothing encoder-shaped is reused: referenceAttention over there takes an
// EncoderShape and a square score matrix, and neither holds for a decoder that
// runs one query against a causal prefix and then against the encoder's rows.
// That one is written out below.
#include "../Encoder/Common.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace WSP::Testing
{
// The tensor names, written out here rather than taken from the loader: this is
// what says the loader looks for the names HuggingFace actually ships, and a
// list shared with the code under test could not say it.
inline std::string decoderTensor(std::string_view suffix)
{
    return "model.decoder." + std::string {suffix};
}

inline std::string decoderLayerTensor(int index, std::string_view suffix)
{
    return decoderTensor("layers.") + std::to_string(index) + "."
           + std::string {suffix};
}

// Small enough that a scalar reference in doubles runs the whole forward pass
// instantly, and lopsided enough that a confused stride shows up: no two
// extents equal, none a multiple of the 8 x 8 dispatch group, and a head width
// (4) that is neither the model width nor the number of heads.
inline DecoderShape smallDecoderShape()
{
    return {.width = 8,
            .heads = 2,
            .layers = 2,
            .feedForwardWidth = 16,
            .vocabularySize = 24,
            .maxPositions = 16,
            .crossPositions = 6};
}

// ---------------------------------------------------------------------------
// A safetensors file assembled in memory, so the loader's name and shape
// mapping is under test alongside the arithmetic.
// ---------------------------------------------------------------------------

// embed_tokens stays F32 in both variants, because the loader rejects a packed
// one outright: it is the gather's table as well as the logits projection's
// weight, and the gather subscripts floats. Only the projections change storage
// here, which is exactly the set Linear can read either way.
inline SafeTensors syntheticDecoderFile(const DecoderShape& shape,
                                        ProjectionStorage projectionStorage)
{
    auto builder = TensorFileBuilder {};
    auto seed = 2000u;

    auto next = [&](int count, float range)
    { return spreadValues(count, seed++, range); };

    auto addProjection =
        [&](const std::string& name, int outputWidth, int innerCount)
    {
        const auto values = next(outputWidth * innerCount, 0.5f);
        const auto extents = std::vector<int> {outputWidth, innerCount};

        if (projectionStorage == ProjectionStorage::PackedHalf)
            builder.addHalves(name, extents, values);
        else
            builder.addFloats(name, extents, values);
    };

    const auto width = shape.width;

    builder.addFloats(decoderTensor("embed_tokens.weight"),
                      {shape.vocabularySize, width},
                      next(shape.vocabularySize * width, 0.3f));

    builder.addFloats(decoderTensor("embed_positions.weight"),
                      {shape.maxPositions, width},
                      next(shape.maxPositions * width, 0.3f));

    for (auto layer = 0; layer < shape.layers; ++layer)
    {
        auto addAttention = [&](std::string_view attention, std::string_view norm)
        {
            const auto prefix = std::string {attention} + ".";

            builder.addFloats(
                decoderLayerTensor(layer, std::string {norm} + ".weight"),
                {width},
                shiftedValues(width, seed++, 0.2f));
            builder.addFloats(
                decoderLayerTensor(layer, std::string {norm} + ".bias"),
                {width},
                next(width, 0.2f));

            addProjection(
                decoderLayerTensor(layer, prefix + "q_proj.weight"), width, width);
            builder.addFloats(decoderLayerTensor(layer, prefix + "q_proj.bias"),
                              {width},
                              next(width, 0.2f));

            addProjection(
                decoderLayerTensor(layer, prefix + "k_proj.weight"), width, width);

            addProjection(
                decoderLayerTensor(layer, prefix + "v_proj.weight"), width, width);
            builder.addFloats(decoderLayerTensor(layer, prefix + "v_proj.bias"),
                              {width},
                              next(width, 0.2f));

            addProjection(
                decoderLayerTensor(layer, prefix + "out_proj.weight"), width, width);
            builder.addFloats(decoderLayerTensor(layer, prefix + "out_proj.bias"),
                              {width},
                              next(width, 0.2f));
        };

        addAttention("self_attn", "self_attn_layer_norm");
        addAttention("encoder_attn", "encoder_attn_layer_norm");

        builder.addFloats(decoderLayerTensor(layer, "final_layer_norm.weight"),
                          {width},
                          shiftedValues(width, seed++, 0.2f));
        builder.addFloats(decoderLayerTensor(layer, "final_layer_norm.bias"),
                          {width},
                          next(width, 0.2f));

        addProjection(
            decoderLayerTensor(layer, "fc1.weight"), shape.feedForwardWidth, width);
        builder.addFloats(decoderLayerTensor(layer, "fc1.bias"),
                          {shape.feedForwardWidth},
                          next(shape.feedForwardWidth, 0.2f));

        addProjection(
            decoderLayerTensor(layer, "fc2.weight"), width, shape.feedForwardWidth);
        builder.addFloats(
            decoderLayerTensor(layer, "fc2.bias"), {width}, next(width, 0.2f));
    }

    builder.addFloats(decoderTensor("layer_norm.weight"),
                      {width},
                      shiftedValues(width, seed++, 0.2f));
    builder.addFloats(decoderTensor("layer_norm.bias"), {width}, next(width, 0.2f));

    return builder.parse();
}

// ---------------------------------------------------------------------------
// The scalar reference: HuggingFace's WhisperDecoder.forward in doubles,
// written from the definition rather than out of the kernels' own references,
// since the chain is what this checks.
// ---------------------------------------------------------------------------

struct ReferenceAttentionWeights
{
    Vector<float> normWeight;
    Vector<float> normBias;
    Vector<float> queryWeight;
    Vector<float> queryBias;
    Vector<float> keyWeight;
    Vector<float> valueWeight;
    Vector<float> valueBias;
    Vector<float> outputWeight;
    Vector<float> outputBias;
};

struct ReferenceDecoderLayer
{
    ReferenceAttentionWeights selfAttention;
    ReferenceAttentionWeights crossAttention;
    Vector<float> finalNormWeight;
    Vector<float> finalNormBias;
    Vector<float> feedForwardWeight;
    Vector<float> feedForwardBias;
    Vector<float> feedForwardOutputWeight;
    Vector<float> feedForwardOutputBias;
};

struct ReferenceDecoderModel
{
    Vector<float> tokenEmbedding;
    Vector<float> positionalEmbedding;
    Vector<float> finalNormWeight;
    Vector<float> finalNormBias;
    std::vector<ReferenceDecoderLayer> layers;
};

// Read back through readFloats, which widens whatever the file holds — so the
// reference runs on exactly the numbers the kernels read, and a packed weight
// costs the comparison nothing.
inline ReferenceDecoderModel readReferenceDecoderModel(const SafeTensors& file,
                                                       const DecoderShape& shape)
{
    auto model = ReferenceDecoderModel {};
    model.tokenEmbedding = file.readFloats(decoderTensor("embed_tokens.weight"));
    model.positionalEmbedding =
        file.readFloats(decoderTensor("embed_positions.weight"));
    model.finalNormWeight = file.readFloats(decoderTensor("layer_norm.weight"));
    model.finalNormBias = file.readFloats(decoderTensor("layer_norm.bias"));

    for (auto index = 0; index < shape.layers; ++index)
    {
        auto read = [&](std::string_view suffix)
        { return file.readFloats(decoderLayerTensor(index, suffix)); };

        auto attention = [&](std::string_view name, std::string_view norm)
        {
            const auto prefix = std::string {name} + ".";
            const auto normName = std::string {norm};

            return ReferenceAttentionWeights {read(normName + ".weight"),
                                              read(normName + ".bias"),
                                              read(prefix + "q_proj.weight"),
                                              read(prefix + "q_proj.bias"),
                                              read(prefix + "k_proj.weight"),
                                              read(prefix + "v_proj.weight"),
                                              read(prefix + "v_proj.bias"),
                                              read(prefix + "out_proj.weight"),
                                              read(prefix + "out_proj.bias")};
        };

        model.layers.push_back({attention("self_attn", "self_attn_layer_norm"),
                                attention("encoder_attn", "encoder_attn_layer_norm"),
                                read("final_layer_norm.weight"),
                                read("final_layer_norm.bias"),
                                read("fc1.weight"),
                                read("fc1.bias"),
                                read("fc2.weight"),
                                read("fc2.bias")});
    }

    return model;
}

// Scaled dot-product attention over [queryCount, width] queries and
// [keyCount, width] keys and values, the heads strided slices of each row. When
// causal, query i sees keys 0 through i and no further, which is the mask
// WhisperDecoder builds for self-attention; cross-attention passes false and
// sees every encoder row.
inline std::vector<double>
    referenceDecoderAttention(const std::vector<double>& queries,
                              int queryCount,
                              const std::vector<double>& keys,
                              const std::vector<double>& values,
                              int keyCount,
                              const DecoderShape& shape,
                              bool causal)
{
    const auto width = shape.width;
    const auto headWidth = shape.headWidth();
    const auto scale = 1.0 / std::sqrt((double) headWidth);

    auto output = std::vector<double>((std::size_t) (queryCount * width));
    auto row = std::vector<double>((std::size_t) keyCount);

    for (auto head = 0; head < shape.heads; ++head)
    {
        const auto headColumn = head * headWidth;

        for (auto query = 0; query < queryCount; ++query)
        {
            const auto visible = causal ? query + 1 : keyCount;
            auto largest = -std::numeric_limits<double>::infinity();

            for (auto key = 0; key < visible; ++key)
            {
                auto total = 0.0;

                for (auto channel = 0; channel < headWidth; ++channel)
                    total +=
                        queries[(std::size_t) (query * width + headColumn + channel)]
                        * keys[(std::size_t) (key * width + headColumn + channel)];

                row[(std::size_t) key] = scale * total;
                largest = std::max(largest, row[(std::size_t) key]);
            }

            auto total = 0.0;

            for (auto key = 0; key < visible; ++key)
            {
                row[(std::size_t) key] = std::exp(row[(std::size_t) key] - largest);
                total += row[(std::size_t) key];
            }

            for (auto channel = 0; channel < headWidth; ++channel)
            {
                auto weighted = 0.0;

                for (auto key = 0; key < visible; ++key)
                    weighted +=
                        row[(std::size_t) key]
                        * values[(std::size_t) (key * width + headColumn + channel)];

                output[(std::size_t) (query * width + headColumn + channel)] =
                    weighted / total;
            }
        }
    }

    return output;
}

// Both halves of the forward pass, so a GPU decoder can be bisected at either:
// the rows layer_norm produces and the vocabulary-wide row the tied embedding
// turns each of them into.
struct ReferenceDecoding
{
    std::vector<double> hidden;
    std::vector<double> logits;
};

// HuggingFace's WhisperDecoder.forward over a whole token sequence starting at
// position 0 — the prompt a run begins with, before any cached step:
//
//   h = embed_tokens[token] + embed_positions[t]
//   per layer: h += self_attn.out_proj(attention(self_attn_layer_norm(h)))
//                                                          causal over t
//              h += encoder_attn.out_proj(attention(encoder_attn_layer_norm(h),
//                                                   encoderOutput))
//              h += fc2(gelu(final_layer_norm(h) through fc1))
//   h = layer_norm(h)
//   logits = h · embed_tokensᵀ
//
// Checked against transformers v4.27's modeling_whisper.py rather than
// remembered: WhisperDecoderLayer keeps the residual, normalises, attends and
// adds back, three times over; WhisperAttention takes its keys and values from
// hidden_states — which the layer has already replaced with the normalised one
// — unless key_value_states is passed, which is what makes cross-attention read
// the encoder's rows unnormalised; and WhisperDecoder adds embed_positions
// straight onto embed_tokens with no sqrt(width), since embed_scale applies
// only when scale_embedding is true and Whisper's config.json says false.
//
// The logits projection has no bias and no weight of its own: proj_out is in
// _keys_to_ignore_on_save and tied to the input embedding, so the file carries
// no tensor for it and the matrix is embed_tokens.
inline ReferenceDecoding referenceDecode(const ReferenceDecoderModel& model,
                                         const DecoderShape& shape,
                                         const Vector<float>& encoderOutput,
                                         const std::vector<int>& tokens)
{
    const auto width = shape.width;
    const auto tokenCount = (int) tokens.size();

    auto encoderRows = std::vector<double>((std::size_t) encoderOutput.size());

    for (auto index = 0; index < encoderOutput.size(); ++index)
        encoderRows[(std::size_t) index] = encoderOutput[index];

    auto hidden = std::vector<double>((std::size_t) (tokenCount * width));

    for (auto position = 0; position < tokenCount; ++position)
        for (auto column = 0; column < width; ++column)
            hidden[(std::size_t) (position * width + column)] =
                (double) model
                    .tokenEmbedding[tokens[(std::size_t) position] * width + column]
                + (double) model.positionalEmbedding[position * width + column];

    // crossSource null is self-attention, whose keys and values are projected
    // from the same normalised rows the queries are — the layer norm is applied
    // once, before the projections, and the mask is the only thing keeping
    // token t out of its own future.
    auto attend = [&](const ReferenceAttentionWeights& weights,
                      const std::vector<double>* crossSource,
                      int keyCount,
                      bool causal)
    {
        const auto normalised = referenceLayerNorm(
            hidden, tokenCount, width, weights.normWeight, weights.normBias);

        const auto queries = referenceLinear(normalised,
                                             tokenCount,
                                             width,
                                             weights.queryWeight,
                                             &weights.queryBias,
                                             width);

        const auto& keySource = crossSource == nullptr ? normalised : *crossSource;

        const auto keys = referenceLinear(
            keySource, keyCount, width, weights.keyWeight, nullptr, width);

        const auto values = referenceLinear(keySource,
                                            keyCount,
                                            width,
                                            weights.valueWeight,
                                            &weights.valueBias,
                                            width);

        const auto attended = referenceDecoderAttention(
            queries, tokenCount, keys, values, keyCount, shape, causal);

        const auto projected = referenceLinear(attended,
                                               tokenCount,
                                               width,
                                               weights.outputWeight,
                                               &weights.outputBias,
                                               width);

        for (auto index = 0u; index < hidden.size(); ++index)
            hidden[index] += projected[index];
    };

    for (const auto& layer: model.layers)
    {
        attend(layer.selfAttention, nullptr, tokenCount, true);
        attend(layer.crossAttention, &encoderRows, shape.crossPositions, false);

        const auto renormalised = referenceLayerNorm(
            hidden, tokenCount, width, layer.finalNormWeight, layer.finalNormBias);

        auto feedForward = referenceLinear(renormalised,
                                           tokenCount,
                                           width,
                                           layer.feedForwardWeight,
                                           &layer.feedForwardBias,
                                           shape.feedForwardWidth);

        for (auto& value: feedForward)
            value = referenceGelu(value);

        const auto feedForwardOutput = referenceLinear(feedForward,
                                                       tokenCount,
                                                       shape.feedForwardWidth,
                                                       layer.feedForwardOutputWeight,
                                                       &layer.feedForwardOutputBias,
                                                       width);

        for (auto index = 0u; index < hidden.size(); ++index)
            hidden[index] += feedForwardOutput[index];
    }

    auto decoding = ReferenceDecoding {};
    decoding.hidden = referenceLayerNorm(
        hidden, tokenCount, width, model.finalNormWeight, model.finalNormBias);

    decoding.logits = referenceLinear(decoding.hidden,
                                      tokenCount,
                                      width,
                                      model.tokenEmbedding,
                                      nullptr,
                                      shape.vocabularySize);

    return decoding;
}

// An encoder output of about the size a real one is: layer_norm leaves rows
// centred with a unit-ish spread, so a decoder fed numbers of the wrong
// magnitude is one whose cross-attention softmax was never in its usual range.
inline Vector<float> syntheticEncoderOutput(const DecoderShape& shape)
{
    auto values = sized(shape.crossElementCount());

    for (auto position = 0; position < shape.crossPositions; ++position)
    {
        for (auto column = 0; column < shape.width; ++column)
        {
            const auto phase = 0.13 * position + 0.041 * column;
            values[position * shape.width + column] =
                (float) (std::sin(phase) + 0.5 * std::cos(2.7 * phase));
        }
    }

    return values;
}

inline int argmaxRow(const std::vector<double>& rows, int row, int rowLength)
{
    auto best = 0;

    for (auto column = 1; column < rowLength; ++column)
        if (rows[(std::size_t) (row * rowLength + column)]
            > rows[(std::size_t) (row * rowLength + best)])
            best = column;

    return best;
}

// The rows of a reference decoding one step is answerable for, since a step
// after the first produces the tail of a sequence the reference computed whole.
inline std::vector<double>
    rowsOf(const std::vector<double>& all, int firstRow, int rowCount, int rowLength)
{
    const auto begin = all.begin() + (std::ptrdiff_t) (firstRow * rowLength);
    return {begin, begin + (std::ptrdiff_t) (rowCount * rowLength)};
}

// ---------------------------------------------------------------------------
// Running the GPU decoder, one step at a time the way a sampler would.
// ---------------------------------------------------------------------------

// One step's outputs, read back together: the rows after the final layer norm,
// the vocabulary rows the tied projection turned them into, and the greedy
// token the Argmax kernel picked out of each. Comparing all three is what lets
// a disagreement bisect to before the logits, after them, or in the sampling.
struct StepResult
{
    Vector<float> hidden;
    Vector<float> logits;
    Vector<std::uint32_t> argmax;
};

// A decoder, its weights and the sequence state. Each call is its own command
// buffer and its own commit, because a test reads what it wrote before deciding
// what to feed next — a real run records several steps before it needs a number
// back.
class DecoderRun
{
public:
    DecoderRun(const DecoderShape& shapeToUse, const SafeTensors& file)
        : decoderShape(shapeToUse)
        , weights(file, shapeToUse)
        , decoder(shapeToUse)
    {
        decoder.prepare(eacp::GPU::Device::shared());
        selection.prepare(eacp::GPU::Device::shared(),
                          decoderShape.maxPositions,
                          decoderShape.logitElementCount());
    }

    // Seconds around the commit, which is the only clock available: eacp's
    // FrameTimer is driven by Frame and a CommandBuffer has no timestamp hook,
    // so an off-screen compute pass is timed by wall clock from a Debug host.
    double begin(const Vector<float>& encoderOutput)
    {
        const auto rows = storageOf(encoderOutput);
        auto commands = eacp::GPU::Device::shared().makeCommandBuffer();

        {
            auto pass = commands.beginCompute();
            decoder.beginSequence(pass, rows, weights);
        }

        return commitSeconds(commands);
    }

    StepResult step(const std::vector<int>& tokens)
    {
        const auto count = (int) tokens.size();
        const auto logits = outputFor(count * decoderShape.logitElementCount());

        lastStepSeconds = recordStep(tokens, logits);

        return {readBack(decoder.hiddenStates(), count * decoderShape.width),
                readBack(logits, count * decoderShape.logitElementCount()),
                greedyTokens(logits, count)};
    }

    // The step that must not record: the logits buffer is filled with a value
    // no logit could be, and a step that threw has to leave it there.
    bool stepThrowsLeavingLogitsUntouched(const std::vector<int>& tokens)
    {
        constexpr auto poison = -12345.f;

        const auto count = (int) tokens.size();
        const auto elements = count * decoderShape.logitElementCount();

        auto filled = sized(elements);

        for (auto index = 0; index < elements; ++index)
            filled[index] = poison;

        const auto logits = storageOf(filled);
        const auto before = decoder.position();
        const auto threw = throwsModelError([&] { recordStep(tokens, logits); });

        if (!threw || decoder.position() != before)
            return false;

        const auto after = readBack(logits, elements);

        for (auto index = 0; index < elements; ++index)
            if (after[index] != poison)
                return false;

        return true;
    }

    int position() const { return decoder.position(); }
    double stepSeconds() const { return lastStepSeconds; }

private:
    static double commitSeconds(eacp::GPU::CommandBuffer& commands)
    {
        const auto start = std::chrono::steady_clock::now();
        commands.commit();
        const auto elapsed = std::chrono::steady_clock::now() - start;

        return std::chrono::duration<double>(elapsed).count();
    }

    double recordStep(const std::vector<int>& tokens,
                      const eacp::GPU::Buffer& logits)
    {
        auto ids = unsignedSized((int) tokens.size());

        for (auto index = 0; index < ids.size(); ++index)
            ids[index] = (std::uint32_t) tokens[(std::size_t) index];

        const auto tokenBuffer = storageOf(ids);
        auto commands = eacp::GPU::Device::shared().makeCommandBuffer();

        {
            auto pass = commands.beginCompute();
            decoder.step(pass, tokenBuffer, ids.size(), weights, logits);
        }

        return commitSeconds(commands);
    }

    // Greedy sampling on the GPU, which is the layer above the decoder rather
    // than part of it: an all-zero mask, so what this asserts is the argmax
    // itself at the vocabulary's real width.
    Vector<std::uint32_t> greedyTokens(const eacp::GPU::Buffer& logits, int rowCount)
    {
        const auto rowLength = decoderShape.logitElementCount();
        const auto mask = storageOf(sized(rowLength));
        const auto chosen = outputFor(rowCount);

        auto commands = eacp::GPU::Device::shared().makeCommandBuffer();

        {
            auto pass = commands.beginCompute();
            selection.encode(pass,
                             eacp::GPU::BufferRange::of(logits),
                             mask,
                             chosen,
                             rowCount,
                             rowLength);
        }

        commands.commit();
        return readBackUnsigned(chosen, rowCount);
    }

    DecoderShape decoderShape;
    DecoderWeights weights;
    Decoder decoder;
    Argmax selection;
    double lastStepSeconds = 0.0;
};
} // namespace WSP::Testing
