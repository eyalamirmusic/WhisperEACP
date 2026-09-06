# NO_MAIN: the whole module is dispatches, so every test here runs inside
# eacp's event loop through the shared entry point in Tests/Support.
#
# whisper-encoder brings whisper-kernels and whisper-model with it, which is
# what lets these tests reuse Tests/Kernels/Common.h and Tests/Model/Common.h
# instead of keeping a third copy of the GPU plumbing and the safetensors
# assembly.
nano_add_executable(EncoderTests
        NO_MAIN
        SOURCES
        EncoderTests.cpp
        TinyEnEncoderTests.cpp
        ${WHISPER_GPU_TEST_MAIN}
        TARGETS whisper-encoder)

# Set whether or not the fetch ran, so the tests against the real weights
# compile either way and skip on the file's absence rather than on an #ifdef.
target_compile_definitions(EncoderTests PRIVATE
        WHISPER_EACP_MODEL_DIR="${WHISPER_EACP_MODEL_DIR}")
