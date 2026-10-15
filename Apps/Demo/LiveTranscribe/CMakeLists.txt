add_executable(LiveTranscribe
        Main.cpp
        LevelMeter.cpp
        MainPanel.cpp
        Session.cpp
        TranscriptView.cpp)

# MakeASound arrives through whisper-audio, which links it PUBLIC because a
# caller hands that module the very Buffer its device callback was given.
target_link_libraries(LiveTranscribe PRIVATE WhisperEACP eacp-ui)

set(BUNDLE_ID "com.eacp.whisper.livetranscribe")

set_target_properties(LiveTranscribe PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_BUNDLE_NAME "Live Transcribe"
        MACOSX_BUNDLE_GUI_IDENTIFIER ${BUNDLE_ID}
        XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER ${BUNDLE_ID})

eacp_set_gui_subsystem(LiveTranscribe)
whisper_set_default_target_setting(LiveTranscribe)
whisper_sign_ios_app(LiveTranscribe)

# Opening the capture side needs NSMicrophoneUsageDescription, which eacp's
# template does not carry, and macOS kills an app that asks for the microphone
# without one.
eacp_add_plist_entries(LiveTranscribe NSMicrophoneUsageDescription
        "Live Transcribe reads the microphone to transcribe what it hears.")

# Nothing is copied into Contents/Resources: the four model files are fetched at
# startup (Model/ModelFetch.h) into a directory every binary in this tree shares,
# and the window draws the download rather than the build paying for it. The
# bundle is 2 MB again.
