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
// Each began as the simplest kernel that is correct — a thread per row or per
// output element, reducing serially — which is the version a scalar CPU
// reference is checked against without argument. The ones a decode step
// spends its time in now give a group of threads to a row (Reduce.h) or split
// an inner sum across one (SplitLinear, AttentionApply), and keep agreeing
// with the same references.

#include "Add.h"
#include "Argmax.h"
#include "Attention.h"
#include "Conv1d.h"
#include "Embed.h"
#include "Gelu.h"
#include "KernelTypes.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MatMul.h"
#include "Reduce.h"
#include "SingleQueryAttention.h"
#include "Softmax.h"
#include "TiledMatMul.h"
#include "Unfold.h"
