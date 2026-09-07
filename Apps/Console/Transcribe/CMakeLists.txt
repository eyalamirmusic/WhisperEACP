add_executable(Transcribe Main.cpp)
target_link_libraries(Transcribe PRIVATE WhisperEACP)

# The model the first two argument forms run, copied beside the binary after
# every link. Called unconditionally: it is a no-op when the build fetched no
# model, and Whisper::hasBundledModel() is what tells the run which of the
# three forms it can serve.
whisper_bundle_model(Transcribe)

# The recording the no-argument run transcribes, embedded so that form needs
# nothing on disk but the model beside it. The ResEmbed runtime key is the
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
