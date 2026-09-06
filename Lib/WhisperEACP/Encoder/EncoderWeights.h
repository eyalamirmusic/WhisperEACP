#pragma once

#include <WhisperEACP/Encoder/EncoderShape.h>
#include <WhisperEACP/Model/SafeTensors.h>

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
                        int index);

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
// LayerNorm and Add subscript a float buffer — so a packed one there would be
// wrong by a factor of two in every index while staying silent, and is rejected
// with a ModelError naming the tensor and saying which kernel reads it. That is
// the whole rule: half where a program can read half, an error everywhere else,
// and never a silent bind.
struct EncoderWeights
{
    EncoderWeights(const SafeTensors& file, const EncoderShape& shapeToUse);

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
