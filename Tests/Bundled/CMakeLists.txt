# NO_MAIN, like every module that reaches the device: loading the bundled model
# and transcribing with it is the whole GPU stack, so these run inside eacp's
# event loop through the shared entry point in Tests/Support.
nano_add_executable(BundledTests
        NO_MAIN
        SOURCES
        BundledTests.cpp
        ${WHISPER_GPU_TEST_MAIN}
        TARGETS whisper-runtime)

# The copy under test: the four fetched files beside this binary after every
# link. A no-op when the build fetched no model, which is what the option's own
# value below is for — the one thing here that cannot be discovered at runtime.
whisper_bundle_model(BundledTests)

# The two directories as Tests/Whisper has them, so this reuses its Common.h
# rather than keeping a second copy.
target_compile_definitions(BundledTests PRIVATE
        WHISPER_EACP_MODEL_DIR="${WHISPER_EACP_MODEL_DIR}"
        WHISPER_EACP_SAMPLE_DIR="${WHISPER_EACP_SAMPLE_DIR}"
        WHISPER_EACP_BUNDLES_MODEL=$<BOOL:${WHISPER_EACP_FETCH_MODEL}>)
