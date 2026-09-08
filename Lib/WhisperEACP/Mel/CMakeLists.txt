add_library(whisper-mel STATIC)

# whisper-audio, because the window and hop this module is written against are
# that module's constants rather than numbers repeated here; whisper-kernels
# for the tiled product the transform is.
target_link_libraries(whisper-mel PUBLIC whisper-audio whisper-kernels)

target_sources(whisper-mel PRIVATE
        Basis.cpp
        Mel.cpp
        MelSpectrogram.cpp
        Window.cpp)

whisper_set_default_target_setting(whisper-mel)
whisper_enable_unity_build(whisper-mel)
whisper_add_ide_sources(whisper-mel)
