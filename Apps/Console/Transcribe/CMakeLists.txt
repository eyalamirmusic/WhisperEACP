add_executable(Transcribe Main.cpp)
target_link_libraries(Transcribe PRIVATE WhisperEACP)

set(BUNDLE_ID "com.eacp.whisper.transcribe")

if (APPLE)
    set_target_properties(Transcribe PROPERTIES
            MACOSX_BUNDLE_BUNDLE_NAME "Transcribe"
            MACOSX_BUNDLE_GUI_IDENTIFIER ${BUNDLE_ID}
            XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER ${BUNDLE_ID})
endif ()

whisper_set_default_target_setting(Transcribe)
