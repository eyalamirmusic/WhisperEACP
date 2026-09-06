nano_add_executable(TokenizerTests
        SOURCES
        TokenizerTests.cpp
        TARGETS whisper-tokenizer)

# A real tokenizer.json is a few megabytes, so it is a download and never a
# commit: the tests that need one skip when it is absent, the same way the GPU
# tests skip without a device. Point them at a downloaded copy with either
# -DWHISPER_EACP_TOKENIZER_JSON=<path> at configure time or the
# WHISPER_TOKENIZER_JSON environment variable at run time.
#
# WHISPER_EACP_MODEL_DIR is the fallback both of those now have, because
# tokenizer.json joined the model fetch: a build configured with
# -DWHISPER_EACP_FETCH_MODEL=ON has the file beside the other three, and a run
# with WHISPER_MODEL_DIR set has it wherever that points.
#
# The committed fixture beside it is reached by absolute path rather than by
# guessing a working directory.
set(WHISPER_EACP_TOKENIZER_JSON "" CACHE FILEPATH
        "A downloaded HuggingFace tokenizer.json to run the tokenizer tests against")

target_compile_definitions(TokenizerTests PRIVATE
        WHISPER_TOKENIZER_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/Fixtures"
        WHISPER_TOKENIZER_JSON_PATH="${WHISPER_EACP_TOKENIZER_JSON}"
        WHISPER_EACP_MODEL_DIR="${WHISPER_EACP_MODEL_DIR}")
