# whisper.cpp as an in-process oracle — the second tier of the validation order
# in CLAUDE.md. Behind WHISPER_EACP_ENABLE_WHISPER_CPP in the root CMakeLists,
# which is also where whisper.cpp itself is fetched: Benchmark/ links the same
# library, and its targets are named globally, so a tree holds one build of it.
#
# Which build this runs against is decided there too. Every backend off, so the
# reference computes exactly ggml's own CPU path — unless the benchmark is in
# the tree, in which case it is whisper.cpp's defaults with the GPU turned off
# at runtime, which on a Mac is ggml's CPU backend plus Accelerate through the
# BLAS one. The tolerances in these tests are measured against the reference's
# own one-ulp resolution rather than assumed, which is what lets them hold
# across both, and loadOracle() prints which build it loaded.
if (NOT WHISPER_EACP_ENABLE_WHISPER_CPP)
    return()
endif ()

# NO_MAIN: our mel runs on the GPU, so these run inside eacp's event loop
# through the shared entry point in Tests/Support.
#
# whisper-runtime is every module of ours with something to compare — the
# front-end, the filterbank the loader publishes, the vocabulary, and the two
# halves of the model with the greedy search on top. whisper is whisper.cpp's
# own library.
nano_add_executable(OracleTests
        NO_MAIN
        SOURCES
        DecoderOracleTests.cpp
        MelOracleTests.cpp
        TokenizerOracleTests.cpp
        TranscriptOracleTests.cpp
        ${WHISPER_GPU_TEST_MAIN}
        TARGETS whisper-runtime whisper)

# Set the same way Tests/Model and Tests/Encoder set theirs: the tests compile
# against a real path whether or not anything was fetched, and skip on the
# file's absence rather than on an #ifdef.
target_compile_definitions(OracleTests PRIVATE
        WHISPER_EACP_MODEL_DIR="${WHISPER_EACP_MODEL_DIR}"
        WHISPER_EACP_GGML_MODEL="${WHISPER_EACP_GGML_MODEL}"
        WHISPER_EACP_JFK_WAV="${whisper-cpp_SOURCE_DIR}/samples/jfk.wav")
