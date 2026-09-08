#pragma once

#include <WhisperEACP/Decoder/DecoderShape.h>
#include <WhisperEACP/Model/SafeTensors.h>

#include <optional>

namespace WSP
{
// One decoder block's tensors, under the names HuggingFace's WhisperDecoder
// gives them: model.decoder.layers.<index>.encoder_attn.q_proj.weight and the
// rest. The names are the file's, not ours, so nothing here is a convention
// that could drift from the repo it loads.
//
// A block is the encoder's with a second attention wedged in the middle:
// self-attention over the tokens decoded so far, then cross-attention over the
// encoder's output, then the feed forward. Each has its own pre-norm.
//
// k_proj is the one projection with no bias in either attention, which is not
// an omission here but PyTorch's own `bias=False` on those two Linears. The
// zero buffer the kernel binds in its place belongs to the Decoder rather than
// to the weights: it is a property of the dispatch, and one buffer serves every
// layer.
struct DecoderLayerWeights
{
    DecoderLayerWeights(const SafeTensors& file,
                        const DecoderShape& shape,
                        int index);

    TensorBuffer selfAttentionNormWeight;
    TensorBuffer selfAttentionNormBias;
    TensorBuffer selfQueryWeight;
    TensorBuffer selfQueryBias;
    TensorBuffer selfKeyWeight;
    TensorBuffer selfValueWeight;
    TensorBuffer selfValueBias;
    TensorBuffer selfAttentionOutputWeight;
    TensorBuffer selfAttentionOutputBias;

    TensorBuffer crossAttentionNormWeight;
    TensorBuffer crossAttentionNormBias;
    TensorBuffer crossQueryWeight;
    TensorBuffer crossQueryBias;
    TensorBuffer crossKeyWeight;
    TensorBuffer crossValueWeight;
    TensorBuffer crossValueBias;
    TensorBuffer crossAttentionOutputWeight;
    TensorBuffer crossAttentionOutputBias;

    TensorBuffer finalNormWeight;
    TensorBuffer finalNormBias;
    TensorBuffer feedForwardWeight;
    TensorBuffer feedForwardBias;
    TensorBuffer feedForwardOutputWeight;
    TensorBuffer feedForwardOutputBias;
};

// Every decoder tensor of a safetensors file, uploaded and checked against the
// shape it is supposed to have. A name the file does not carry, a rank that is
// not the tensor's, and a dimension that disagrees with the config are each a
// ModelError naming the tensor — the alternative is a kernel walking a buffer
// at a stride it does not have, which is silent on both backends.
//
// **There is no proj_out.weight.** Verified two ways: openai/whisper-tiny.en's
// own model.safetensors carries 167 tensors, every one of them under
// model.encoder. or model.decoder., and transformers puts `proj_out.weight` in
// WhisperForConditionalGeneration's _keys_to_ignore_on_save and ties the module
// to the input embedding. So the logits projection is embed_tokens itself —
// `logits = h · embed_tokensᵀ`, reading the same matrix the input embedding was
// gathered from — and a repo that did ship an untied one would be a different
// architecture rather than a variant of this.
//
// **fp16 storage.** A projection weight may stay packed: Linear has a
// half-reading variant and the Decoder picks it by TensorBuffer::storage. Every
// other tensor here is bound to a program that has no half read — LayerNorm and
// the embedding gather subscript a float buffer, and every bias is a float
// subscript inside Linear itself — so a packed one there would be wrong by a
// factor of two in every index while staying silent, and is rejected with a
// ModelError naming the tensor and saying which kernel reads it.
//
// embed_tokens is the tensor that decides its own case, because it is the one
// bound twice. The gather that builds `h = embed_tokens[token] +
// embed_positions[t]` subscripts it as floats; the logits projection at the end
// could read it packed through Linear. One buffer cannot be both, and the
// gather is the reader with no packed form, so **embed_tokens must be F32** and
// a packed one is a ModelError naming it. That costs a fp16 repo 40 MB on the
// widening path in SafeTensors rather than a second copy of the matrix, and it
// keeps the tie between the embedding and the logits exact instead of exact
// only up to a conversion.
//
// The one buffer that cannot be both is why LogitsWeight below is a *second*
// copy rather than a choice of storage for the one tensor: the gather keeps
// its floats and the projection gets the halves.
struct DecoderWeights
{
    // What the logits projection reads. Tied is embed_tokens itself, the
    // matrix the gather reads. PackedHalfCopy is a second, fp16 copy of it
    // kept beside the float one, so the projection reads 40 MB a step instead
    // of 80 — 133 us of a 590 us step down to about 70.
    //
    // **It is not a numerical trade.** The copy is made only where narrowing
    // is bit-exact, which for a Whisper repo is everywhere: OpenAI's
    // checkpoints are fp16 and HuggingFace's conversion only widens them, so
    // every one of tiny.en's 167 tensors round-trips through fp16 unchanged
    // and the logits are identical to the last bit. A repo genuinely saved in
    // fp32 gets no copy and the tied weight, which is why this asks for a copy
    // rather than promising one — packedTokenEmbedding is empty when the file
    // would have lost something.
    enum class LogitsWeight
    {
        Tied,
        PackedHalfCopy
    };

    DecoderWeights(const SafeTensors& file,
                   const DecoderShape& shapeToUse,
                   LogitsWeight logitsWeightToUse = LogitsWeight::Tied);

    DecoderShape shape;

    // [vocabularySize, width]. Both the gather's table and, unless a packed
    // copy was asked for, the logits projection's weight — which is what ties
    // them.
    TensorBuffer tokenEmbedding;

    // The fp16 copy of the above, present only under PackedHalfCopy.
    std::optional<TensorBuffer> packedTokenEmbedding;

    // What Decoder::step binds for the logits, which is the one place the
    // choice above is read.
    const TensorBuffer& logitsWeight() const
    {
        return packedTokenEmbedding ? *packedTokenEmbedding : tokenEmbedding;
    }

    // [maxTargetPositions, width] in the file, of which row t is added to token
    // t. A model is allowed to carry more rows than a given run needs; fewer is
    // an error.
    TensorBuffer positionalEmbedding;

    TensorBuffer finalNormWeight;
    TensorBuffer finalNormBias;

    Vector<DecoderLayerWeights> layers;
};
} // namespace WSP
