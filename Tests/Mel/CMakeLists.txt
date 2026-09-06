# NO_MAIN: these reach the GPU, so they run inside eacp's event loop through
# the shared entry point in Tests/Support.
nano_add_executable(MelTests
        NO_MAIN
        SOURCES
        FilterBankTests.cpp
        MelSpectrogramTests.cpp
        MelTests.cpp
        NormaliseTests.cpp
        StftTests.cpp
        WindowTests.cpp
        ${WHISPER_GPU_TEST_MAIN}
        TARGETS whisper-mel)
