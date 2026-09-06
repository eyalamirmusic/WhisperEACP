add_executable(DeviceInfo Main.cpp)
target_link_libraries(DeviceInfo PRIVATE WhisperEACP)

set(BUNDLE_ID "com.eacp.whisper.deviceinfo")

if (APPLE)
    set_target_properties(DeviceInfo PROPERTIES
            MACOSX_BUNDLE_BUNDLE_NAME "DeviceInfo"
            MACOSX_BUNDLE_GUI_IDENTIFIER ${BUNDLE_ID}
            XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER ${BUNDLE_ID})
endif ()

whisper_set_default_target_setting(DeviceInfo)
