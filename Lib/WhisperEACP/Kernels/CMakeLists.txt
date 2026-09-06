add_library(whisper-kernels STATIC)

target_link_libraries(whisper-kernels PUBLIC whisper-core)

target_sources(whisper-kernels PRIVATE
        Kernels.cpp)

whisper_set_default_target_setting(whisper-kernels)
whisper_enable_unity_build(whisper-kernels)
whisper_add_ide_sources(whisper-kernels)
