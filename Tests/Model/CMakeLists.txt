# NO_MAIN, because a loaded tensor's whole point is the GPU buffer at the end of
# it: Tests/Support/GpuTestMain.cpp is the entry point that runs inside eacp's
# event loop, and the tests that reach the device return early when there is
# none.
#
# whisper-audio is linked as well as whisper-model so the tests can assert that
# preprocessor_config.json agrees with Audio/Format.h value for value. The
# library itself does not depend on Audio — that check belongs at the seam, not
# inside the loader.
nano_add_executable(ModelTests
        NO_MAIN
        SOURCES
        ConfigTests.cpp
        ModelIOTests.cpp
        SafeTensorsTests.cpp
        TinyEnTests.cpp
        ${WHISPER_GPU_TEST_MAIN}
        TARGETS whisper-model whisper-audio)

# Set whether or not the fetch ran, so the tests against the real files compile
# either way and skip on the file's absence rather than on an #ifdef.
target_compile_definitions(ModelTests PRIVATE
        WHISPER_EACP_MODEL_DIR="${WHISPER_EACP_MODEL_DIR}")
