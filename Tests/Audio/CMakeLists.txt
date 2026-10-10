whisper_add_test_executable(AudioTests
        SOURCES
        CaptureTests.cpp
        FormatTests.cpp
        WavFileTests.cpp
        TARGETS whisper-audio)
