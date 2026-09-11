#pragma once

#include <WhisperEACP/Encoder/EncoderShape.h>
#include <WhisperEACP/Model/SafeTensors.h>
#include <WhisperEACP/Model/TensorLoader.h>

namespace WSP
{
// One transformer block's tensors, under the names HuggingFace's WhisperEncoder
// gives them: model.encoder.layers.<index>.self_attn_layer_norm.weight and the
// rest. The names are the file's, not ours, so nothing here is a convention
// that could drift from the repo it loads.
//
// self_attn.k_proj is the one projection with no bias, which is not an omission
// here but PyTorch's own `bias=False` on that one Linear. The zero buffer the
// kernel binds in its place belongs to the Encoder rather than to the weights:
// it is a property of the dispatch, and one buffer serves every layer.
struct EncoderLayerWeights
{
    EncoderLayerWeights(const SafeTensors& file,
                        const EncoderShape& shape,
                        int index,
                        WeightPacking packing);

    TensorBuffer attentionNormWeight;
    TensorBuffer attentionNormBias;
    TensorBuffer queryWeight;
    TensorBuffer queryBias;
    TensorBuffer keyWeight;
    TensorBuffer valueWeight;
    TensorBuffer valueBias;
    TensorBuffer attentionOutputWeight;
    TensorBuffer attentionOutputBias;
    TensorBuffer finalNormWeight;
    TensorBuffer finalNormBias;
    TensorBuffer feedForwardWeight;
    TensorBuffer feedForwardBias;
    TensorBuffer feedForwardOutputWeight;
    TensorBuffer feedForwardOutputBias;
};

// Every encoder tensor of a safetensors file, uploaded and checked against the
// shape it is supposed to have. A name the file does not carry, a rank that is
// not the tensor's, and a dimension that disagrees with the config are each a
// ModelError naming the tensor — the alternative is a kernel walking a buffer
// at a stride it does not have, which is silent on both backends.
//
// **fp16 storage.** A projection weight may stay packed: Linear has a
// half-reading variant and the Encoder picks it by TensorBuffer::storage. Every
// other tensor here is bound to a program that has no half read — Conv1d,
// LayerNorm and Add subscript a float buffer — so an fp16 one is widened on the
// way to the device, which is what lets a repo shipped in fp16 load at all.
// That is the whole rule: half where a program can read half, floats everywhere
// else whatever the file holds, and never a packed buffer under a float
// subscript, which would be wrong in every index while staying silent.
//
// Which of the two a *projection* is uploaded as is `packing`, and it is a
// policy rather than the file's answer: under WeightPacking::ExactHalf the six
// projections of every layer are narrowed to fp16 wherever narrowing is
// bit-exact — 28 MB of tiny.en's weights read at half the bytes for the same
// values — and uploaded as the file holds them wherever it is not. Nothing
// else here moves: the convolutions, the norms, the biases and the positional
// embedding are float whatever is asked for.
struct EncoderWeights
{
    EncoderWeights(const SafeTensors& file,
                   const EncoderShape& shapeToUse,
                   WeightPacking packingToUse = WeightPacking::Float);

    EncoderShape shape;

    TensorBuffer firstConvolutionWeight;
    TensorBuffer firstConvolutionBias;
    TensorBuffer secondConvolutionWeight;
    TensorBuffer secondConvolutionBias;

    // [maxSourcePositions, width] in the file, of which the first positions()
    // rows are added to the convolved input. A model is allowed to carry more
    // than a given run needs; fewer is an error.
    TensorBuffer positionalEmbedding;

    TensorBuffer finalNormWeight;
    TensorBuffer finalNormBias;

    Vector<EncoderLayerWeights> layers;
};
} // namespace WSP
