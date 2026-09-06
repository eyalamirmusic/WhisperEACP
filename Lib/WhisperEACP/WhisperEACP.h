#pragma once

// WhisperEACP: Whisper speech-to-text on eacp's GPU compute stack.
//
// Every layer of the model is authored as an eacp::GPU::ComputeProgram — a C++
// struct whose body is written in eacp's shader EDSL rather than in a shading
// language — so one source emits MSL on Apple and HLSL on Windows, and Metal
// and D3D12 both come free. Audio reaches it through MakeASound.

#include "Core/Core.h"
#include "Audio/Audio.h"
#include "Whisper/Whisper.h"
