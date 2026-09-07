#pragma once

// The weights, read from HuggingFace safetensors — a JSON header naming each
// tensor's dtype, shape and byte range, followed by one raw blob — and the
// config.json and preprocessor_config.json beside it that carry the layer
// counts, the widths and the mel filterbank.
//
// A published format rather than a convention read off another project's
// source, and the reason nothing here needs a converter.

#include <WhisperEACP/Core/Core.h>
#include <WhisperEACP/Model/ModelConfig.h>
#include <WhisperEACP/Model/ModelError.h>
#include <WhisperEACP/Model/ModelFiles.h>
#include <WhisperEACP/Model/PreprocessorConfig.h>
#include <WhisperEACP/Model/SafeTensors.h>
#include <WhisperEACP/Model/TensorType.h>
