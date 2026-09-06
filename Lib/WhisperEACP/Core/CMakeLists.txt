add_library(whisper-core STATIC)

# Lib/ is the include root, so a header of ours is spelled
# <WhisperEACP/Core/Core.h>, beside eacp's own <eacp/...>.
target_include_directories(whisper-core PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../)

# eacp-gpu rather than eacp-core, and PUBLIC: every layer above this one is a
# ComputeProgram holding GPU buffers, so eacp's GPU headers and its Metal /
# D3D12 link line are in our own signatures. The /Zc:preprocessor that
# EACP_SHADER needs on MSVC rides along the same way, being PUBLIC on eacp-gpu.
target_link_libraries(whisper-core PUBLIC eacp-gpu)

target_sources(whisper-core PRIVATE
        Utils/Version.cpp)

# The one place the library's own version lives is project(), so read it from
# there rather than keeping a second copy in the source that can drift.
target_compile_definitions(whisper-core PRIVATE
        WHISPER_EACP_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        WHISPER_EACP_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        WHISPER_EACP_VERSION_PATCH=${PROJECT_VERSION_PATCH})

whisper_set_default_target_setting(whisper-core)
whisper_enable_unity_build(whisper-core)
whisper_add_ide_sources(whisper-core)
