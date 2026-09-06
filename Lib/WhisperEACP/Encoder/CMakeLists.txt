add_library(whisper-encoder STATIC)

# The first module that includes headers from two others, and by design: the
# encoder is assembled out of Kernels/ and fed by Model/. The seam that kept
# those four independent of each other was never meant to hold above them.
target_link_libraries(whisper-encoder PUBLIC whisper-kernels whisper-model)

target_sources(whisper-encoder PRIVATE
        Encoder.cpp
        EncoderShape.cpp
        EncoderWeights.cpp)

whisper_set_default_target_setting(whisper-encoder)
whisper_enable_unity_build(whisper-encoder)
whisper_add_ide_sources(whisper-encoder)
