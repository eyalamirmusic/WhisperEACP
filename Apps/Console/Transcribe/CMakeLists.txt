add_executable(Transcribe Main.cpp)
target_link_libraries(Transcribe PRIVATE WhisperEACP)

# No model is copied beside this binary: the first two argument forms fetch
# openai/whisper-tiny.en at startup instead (Model/ModelFetch.h), into a
# directory every binary in this tree shares. The build downloads nothing.

# The recording the no-argument run transcribes, embedded so that form needs
# nothing on disk at all but the fetched model. The ResEmbed runtime key is the
# file's basename, which is the "jfk.wav" Main.cpp asks for.
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
