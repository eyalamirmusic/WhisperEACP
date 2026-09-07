# NO_MAIN, like every module that reaches the device: loading the embedded
# model and transcribing with it is the whole GPU stack, so these run inside
# eacp's event loop through the shared entry point in Tests/Support.
#
# whisper-embedded-model is linked whether or not it embeds anything — it is an
# empty INTERFACE target when WHISPER_EACP_EMBED_MODEL is off, which is the
# whole point of the first test below.
nano_add_executable(EmbeddedTests
        NO_MAIN
        SOURCES
        EmbeddedTests.cpp
        ${WHISPER_GPU_TEST_MAIN}
        TARGETS whisper-runtime whisper-embedded-model)

# Directly rather than through whisper-embedded-model, which carries ResEmbed
# only when it embeds something: the tests read the registry either way, and
# whisper-runtime keeps its own link to it PRIVATE.
find_package(ResEmbed REQUIRED)
target_link_libraries(EmbeddedTests PRIVATE ResEmbed)

# The two directories as Tests/Whisper has them, so this reuses its Common.h
# rather than keeping a second copy — plus the option's own value, which is the
# one thing here that cannot be discovered at runtime and must not skip.
target_compile_definitions(EmbeddedTests PRIVATE
        WHISPER_EACP_MODEL_DIR="${WHISPER_EACP_MODEL_DIR}"
        WHISPER_EACP_SAMPLE_DIR="${WHISPER_EACP_SAMPLE_DIR}"
        WHISPER_EACP_EMBEDS_MODEL=$<BOOL:${WHISPER_EACP_EMBED_MODEL}>)
