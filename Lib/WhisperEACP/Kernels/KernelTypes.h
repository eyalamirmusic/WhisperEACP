#pragma once

#include <WhisperEACP/Core/Core.h>

#include <eacp/GPU/GPU.h>

namespace WSP
{
// The EDSL vocabulary every kernel in this directory is declared in, brought in
// by name the way Core/Common.h brings in ea_data_structures'. The intrinsics
// and operators need no import: their arguments are eacp::GPU types, so ADL
// finds them.
using eacp::GPU::AtomicBuffer;
using eacp::GPU::ComputePass;
using eacp::GPU::ComputeProgram;
using eacp::GPU::Float;
using eacp::GPU::Float4;
using eacp::GPU::InputBuffer;
using eacp::GPU::OutputBuffer;
using eacp::GPU::Shared;
using eacp::GPU::UInt;
using eacp::GPU::UIntInputBuffer;
using eacp::GPU::UIntOutputBuffer;
using eacp::GPU::Uniform;
using eacp::GPU::Var;
} // namespace WSP
