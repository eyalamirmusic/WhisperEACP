add_library(whisper-encoder STATIC)

target_link_libraries(whisper-encoder PUBLIC whisper-net)

target_sources(whisper-encoder PRIVATE
        Encoder.cpp
        EncoderShape.cpp
        EncoderWeights.cpp)

if (TARGET eacp-ml)
    target_sources(whisper-encoder PRIVATE CoreMLEncoder.cpp)
    target_link_libraries(whisper-encoder PUBLIC eacp-ml)
endif ()

whisper_set_default_target_setting(whisper-encoder)
whisper_enable_unity_build(whisper-encoder)
whisper_add_ide_sources(whisper-encoder)
