# NO_MAIN: DecoderWeights uploads every tensor through makeBuffer, so this
# module needs a device and therefore eacp's event loop, through the shared
# entry point in Tests/Support.
#
# whisper-encoder is linked for its test suite rather than for the library:
# Tests/Decoder/Common.h includes Tests/Encoder/Common.h, which is where the
# safetensors builder, packHalf and the layer norm and linear references
# already live. The decoder's own target does not depend on the encoder's.
nano_add_executable(DecoderTests
        NO_MAIN
        SOURCES
        DecoderShapeTests.cpp
        DecoderTests.cpp
        DecoderWeightsTests.cpp
        ReferenceDecoderTests.cpp
        TinyEnDecoderTests.cpp
        ${WHISPER_GPU_TEST_MAIN}
        TARGETS whisper-decoder whisper-encoder)

# Set whether or not the fetch ran, so the tests against the real weights
# compile either way and skip on the file's absence rather than on an #ifdef.
target_compile_definitions(DecoderTests PRIVATE
        WHISPER_EACP_MODEL_DIR="${WHISPER_EACP_MODEL_DIR}")
