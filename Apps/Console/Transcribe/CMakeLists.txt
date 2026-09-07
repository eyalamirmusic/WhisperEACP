add_executable(Transcribe Main.cpp)

# whisper-embedded-model unconditionally: it is an empty INTERFACE target when
# WHISPER_EACP_EMBED_MODEL is off, and Whisper::hasEmbeddedModel() is what tells
# the run which of the three argument forms it can serve.
target_link_libraries(Transcribe PRIVATE WhisperEACP whisper-embedded-model)

# The recording the no-argument run transcribes, embedded so that form needs
# nothing on disk at all. The ResEmbed runtime key is the file's basename, which
# is the "jfk.wav" Main.cpp asks for.
find_package(ResEmbed REQUIRED)
res_embed_add(Transcribe
        FILES     "${WHISPER_EACP_SAMPLE_DIR}/jfk.wav"
        NAMESPACE TranscribeSamples
        CATEGORY  TranscribeSamples)

set(BUNDLE_ID "com.eacp.whisper.transcribe")

if (APPLE)
    set_target_properties(Transcribe PROPERTIES
            MACOSX_BUNDLE_BUNDLE_NAME "Transcribe"
            MACOSX_BUNDLE_GUI_IDENTIFIER ${BUNDLE_ID}
            XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER ${BUNDLE_ID})
endif ()

whisper_set_default_target_setting(Transcribe)
