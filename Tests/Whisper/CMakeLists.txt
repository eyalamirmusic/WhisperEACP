# NO_MAIN: transcribing is the whole GPU stack in one call, so every test here
# runs inside eacp's event loop through the shared entry point in Tests/Support.
#
# whisper-runtime brings every module it assembles with it, which is what lets
# these reuse Tests/Model/Common.h for the model directory rather than keeping a
# second copy of the same pair of environment variables — and lets
# PackedWeightsTests.cpp drive one decode step through Tests/Decoder/Common.h's
# harness rather than assembling a second one here.
nano_add_executable(WhisperTests
        NO_MAIN
        SOURCES
        LiveTranscriberTests.cpp
        PackedWeightsTests.cpp
        TranscribeTests.cpp
        WhisperTests.cpp
        ${WHISPER_GPU_TEST_MAIN}
        TARGETS whisper-runtime)

# Both set whether or not the fetch ran, so the tests compile against real paths
# either way and skip on a file's absence rather than on an #ifdef.
target_compile_definitions(WhisperTests PRIVATE
        WHISPER_EACP_MODEL_DIR="${WHISPER_EACP_MODEL_DIR}"
        WHISPER_EACP_SAMPLE_DIR="${WHISPER_EACP_SAMPLE_DIR}")
