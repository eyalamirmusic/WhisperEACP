add_library(whisper-net STATIC)

target_link_libraries(whisper-net PUBLIC whisper-kernels whisper-model eacp-ml-graph)

target_sources(whisper-net PRIVATE
        CoreMLNet.cpp
        KernelNet.cpp
        Net.cpp)

whisper_set_default_target_setting(whisper-net)
whisper_enable_unity_build(whisper-net)
whisper_add_ide_sources(whisper-net)
