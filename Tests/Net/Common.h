#pragma once

// The encoder's synthetic file, its mel and its error measure from
// Tests/Encoder, and the model and sample paths from Tests/Whisper: this suite
// runs the encoder both ways, so it is assembled out of those two.
#include "../Encoder/Common.h"
#include "../Whisper/Common.h"

#include <WhisperEACP/Net/CoreMLNet.h>

namespace WSP::Testing
{
inline Span<const int> audioContextSpan()
{
    static constexpr auto contexts = Whisper::enumeratedAudioContexts();
    return {contexts.data(), contexts.size()};
}

inline Binding namedBinding(const Shape& capacity, const std::string& name)
{
    return Binding {{}, capacity, name};
}

// The encoder recorded into a Core ML net exactly as CoreMLEncoder records it,
// with nothing loaded.
inline void recordCoreMLEncoder(CoreMLNet& net,
                                const EncoderShape& shape,
                                const EncoderWeights& weights)
{
    recordEncoder(net,
                  shape,
                  weights,
                  namedBinding({shape.melBins, shape.inputFrames}, "mel"),
                  namedBinding({}, "rows"),
                  shape.inputFrames);
}
} // namespace WSP::Testing
