#pragma once

// Whisper's front-end: a 400-point STFT at a 160-sample hop, the power
// spectrum, the 80 x 201 filterbank, log10, and the clamp-and-scale the model
// was trained against. 30 s of audio in, an 80 x 3000 mel spectrogram out.
//
// The filterbank is a buffer the caller binds rather than a table compiled in
// — it is published in the model's preprocessor_config.json, so nothing here
// reconstructs one.

#include <WhisperEACP/Audio/Audio.h>
#include <WhisperEACP/Core/Core.h>

#include "Basis.h"
#include "MelKernels.h"
#include "MelShape.h"
#include "MelSpectrogram.h"
#include "Window.h"
