add_library(whisper-decoder STATIC)

target_link_libraries(whisper-decoder PUBLIC whisper-net)

target_sources(whisper-decoder PRIVATE
        Decoder.cpp
        DecoderShape.cpp
        DecoderWeights.cpp)

if (TARGET eacp-ml)
    target_sources(whisper-decoder PRIVATE CoreMLDecoder.cpp)
    target_link_libraries(whisper-decoder PUBLIC eacp-ml)
endif ()

whisper_set_default_target_setting(whisper-decoder)
whisper_enable_unity_build(whisper-decoder)
whisper_add_ide_sources(whisper-decoder)
