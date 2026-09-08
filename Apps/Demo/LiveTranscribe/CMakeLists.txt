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

# After the defaults, which point every bundle here at eacp's stock plist.
# Opening the capture side needs NSMicrophoneUsageDescription, which neither
# eacp's template nor CMake's default carries, and macOS kills an app that asks
# for the microphone without one. MakeASound's own probe app has the same file
# for the same reason.
if (APPLE)
    set_target_properties(LiveTranscribe PROPERTIES MACOSX_BUNDLE_INFO_PLIST
            "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist.in")
endif ()

# Into Contents/Resources/WhisperModel, which is where Whisper::loadBundled()
# looks. A no-op when the build fetched no model, and then the window says so
# rather than the app failing to start.
whisper_bundle_model(LiveTranscribe)
