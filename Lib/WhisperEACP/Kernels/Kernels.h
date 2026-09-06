#pragma once

// The op set a transformer is assembled from, each a GPU::ComputeProgram in
// its own header. Shapes are uniforms rather than Whisper's own numbers: these
// are what the encoder and decoder get written out of, and a matmul that only
// knows tiny.en's widths is one that gets rewritten for base.
//
// The outer extent — how many rows, how many elements, how many output rows —
// comes from the dispatch and never appears as a uniform, since the generated
// bounds guard already has it. The inner extents, which are what a kernel walks
// its buffers at, are uniforms.
//
// Every one of these is the simplest kernel that is correct: a thread per row
// or per output element, reducing serially, with no threadgroup tiling. That is
// the version a scalar CPU reference can be checked against without argument,
// and the one an optimised kernel later has to keep agreeing with.

#include "Gelu.h"
#include "KernelTypes.h"
#include "LayerNorm.h"
#include "MatMul.h"
#include "Softmax.h"
