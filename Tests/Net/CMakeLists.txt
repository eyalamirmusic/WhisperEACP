nano_add_executable(NetTests
        NO_MAIN
        SOURCES
        CoreMLEncoderTests.cpp
        CoreMLNetTests.cpp
        ${WHISPER_MODEL_TEST_MAIN}
        TARGETS whisper-runtime)

target_compile_definitions(NetTests PRIVATE
        WHISPER_EACP_SAMPLE_DIR="${WHISPER_EACP_SAMPLE_DIR}")
