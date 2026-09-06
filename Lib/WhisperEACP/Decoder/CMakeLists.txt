add_library(whisper-decoder STATIC)

# The encoder's dependencies exactly: assembled out of Kernels/, fed by Model/.
# It does not link whisper-encoder — the encoder's output reaches it as a
# buffer, which is the seam that lets the two be written and tested apart.
target_link_libraries(whisper-decoder PUBLIC whisper-kernels whisper-model)

target_sources(whisper-decoder PRIVATE
        Decoder.cpp
        DecoderShape.cpp
        DecoderWeights.cpp)

whisper_set_default_target_setting(whisper-decoder)
whisper_enable_unity_build(whisper-decoder)
whisper_add_ide_sources(whisper-decoder)
