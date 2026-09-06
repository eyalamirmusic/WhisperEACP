add_library(whisper-audio STATIC)

# MakeASound is PUBLIC: a caller hands this module the very Buffer its device
# callback was given, so its types are in our signatures.
target_link_libraries(whisper-audio PUBLIC whisper-core MakeASound)

target_sources(whisper-audio PRIVATE
        Format.cpp
        WavFile.cpp)

whisper_set_default_target_setting(whisper-audio)
whisper_enable_unity_build(whisper-audio)
whisper_add_ide_sources(whisper-audio)
