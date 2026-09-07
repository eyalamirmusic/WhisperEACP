# whisper-runtime rather than whisper-whisper: every other target here is
# whisper- plus the directory's name, and the one directory that is the whole
# runtime cannot follow that without stuttering. The class is WSP::Whisper and
# the header is <WhisperEACP/Whisper/Whisper.h>, which is what a caller writes.
add_library(whisper-runtime STATIC)

# The only target that links every layer, because it is the one that assembles
# them: audio in through whisper-audio's file reader, the front-end, the two
# halves of the model, and the vocabulary the tokens come back out through.
target_link_libraries(whisper-runtime PUBLIC
        whisper-audio
        whisper-decoder
        whisper-encoder
        whisper-mel
        whisper-model
        whisper-tokenizer)

# ResEmbed comes from eacp, which fetches it and whose CMake/ carries the find
# module — reachable here because CMake/Findeacp.cmake re-appends that directory
# to CMAKE_MODULE_PATH. PRIVATE, because loadEmbedded() looks the four files up
# inside the .cpp and no header of ours mentions ResEmbed.
find_package(ResEmbed REQUIRED)
target_link_libraries(whisper-runtime PRIVATE ResEmbed)

target_sources(whisper-runtime PRIVATE
        Whisper.cpp)

whisper_set_default_target_setting(whisper-runtime)
whisper_enable_unity_build(whisper-runtime)
whisper_add_ide_sources(whisper-runtime)
