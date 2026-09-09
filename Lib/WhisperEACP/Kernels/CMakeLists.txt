add_library(whisper-kernels STATIC)

target_link_libraries(whisper-kernels PUBLIC whisper-core)

# PUBLIC: every module holding one of SimdTiledMatMul.h's role aliases compiles
# against the same answer this one did.
if (WHISPER_EACP_HAS_SIMD_MATRIX)
    target_compile_definitions(whisper-kernels PUBLIC WHISPER_EACP_HAS_SIMD_MATRIX)
endif ()

target_sources(whisper-kernels PRIVATE
        Argmax.cpp
        Kernels.cpp
        SingleQueryAttention.cpp)

whisper_set_default_target_setting(whisper-kernels)
whisper_enable_unity_build(whisper-kernels)
whisper_add_ide_sources(whisper-kernels)
