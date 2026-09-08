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

target_sources(whisper-runtime PRIVATE
        LiveTranscriber.cpp
        Whisper.cpp)

# Where the build's post-build copy put what this binary ships — see
# whisper_copy_resources in CMake/WhisperResources.cmake. One answer per
# platform: CoreFoundation's on Apple, which is the bundle's Resources for a
# bundle and the executable's directory for anything else, and the executable's
# directory on Windows.
if (APPLE)
    target_sources(whisper-runtime PRIVATE
            ResourcesDirectory-Apple.cpp)

    target_link_libraries(whisper-runtime PRIVATE "-framework CoreFoundation")
elseif (WIN32)
    target_sources(whisper-runtime PRIVATE
            ResourcesDirectory-Windows.cpp)
endif ()

whisper_set_default_target_setting(whisper-runtime)
whisper_enable_unity_build(whisper-runtime)
whisper_add_ide_sources(whisper-runtime)
