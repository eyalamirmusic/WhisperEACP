add_subdirectory(Core)

# Above Core, below everything else: the mel front-end and the encoder are
# written against the format this module fixes, and it is the only place that
# talks to a capture device.
add_subdirectory(Audio)

# The whole library, for consumers that don't want to name the modules they use.
add_library(WhisperEACP INTERFACE)
target_link_libraries(WhisperEACP INTERFACE
        whisper-core
        whisper-audio)
