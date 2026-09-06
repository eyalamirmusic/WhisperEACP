#pragma once

#include "KernelTypes.h"

namespace WSP
{
// The decoder's input row, which is a pair of gathers and the sum of them:
//
//   output[t, c] = tokenTable[token[t], c]
//                + positionTable[firstPosition + t, c]
//
// exactly what WhisperDecoder.forward computes as embed_tokens(input_ids)
// added to embed_positions(past_length + arange(tokenCount)). One store rather
// than two dispatches and an Add, because neither table element is read twice:
// fusing them costs nothing and saves a temporary the size of the whole input.
//
// The layouts, over a model width W:
//
//   tokens         [tokenCount]        one token id per step
//   tokenTable     [vocabulary, W]     row-major
//   positionTable  [maxPositions, W]   row-major
//   output         [tokenCount, W]     row-major
//
// One thread per (c, t) over a 2D grid: dispatch(kernel, W, tokenCount). The
// token count is the dispatch height and never reaches the body; W is a
// uniform, since it is the stride all three of the tables and the output are
// walked at.
//
// firstPosition is how many tokens were decoded before this call, so a step
// that appends one token to a cache of n binds firstPosition = n and
// dispatches a height of one, while the prompt that opens a sequence binds
// zero and dispatches its whole length. It is a uniform rather than an offset
// folded into positionTable's base because a ComputePass binds whole buffers,
// so there is no base to fold it into.
//
// Token ids are unsigned integers, and they are read as bits rather than as
// numbers. eacp's storage buffers are runs of floats on both backends —
// `device const float*` on Metal, StructuredBuffer<float> on HLSL — so the
// only element type an InputBuffer yields is Float, and the way an integer
// payload comes back is the bitcast asUInt (ShaderValue.h), which is exact by
// construction on both. The caller therefore uploads uint32 ids and this reads
// their pattern back. The alternative eacp offers, an AtomicBuffer of uints,
// is a read-write binding for what is an input and buys nothing here.
//
// What is deliberately not done is storing ids as floats: a vocabulary is
// indexed, not measured, and a float id is a rounding hazard the moment a
// vocabulary passes 2^24.
struct Embed final : ComputeProgram
{
    Embed() { compile(); }

    void define() override
    {
        auto position = threadPosition();
        auto channel = position.x;
        auto step = position.y;

        auto token = asUInt(tokens[step]);

        write(output,
              step * width + channel,
              tokenTable[token * width + channel]
                  + positionTable[(firstPosition + step) * width + channel]);
    }

    Uniform<InputBuffer> tokens;
    Uniform<InputBuffer> tokenTable;
    Uniform<InputBuffer> positionTable;
    Uniform<OutputBuffer> output;
    Uniform<UInt> width;
    Uniform<UInt> firstPosition;

    EACP_SHADER(tokens, tokenTable, positionTable, output, width, firstPosition)
};
} // namespace WSP
