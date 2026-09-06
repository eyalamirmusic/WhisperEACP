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
# The committed fixture beside it is reached by absolute path rather than by
# guessing a working directory.
set(WHISPER_EACP_TOKENIZER_JSON "" CACHE FILEPATH
        "A downloaded HuggingFace tokenizer.json to run the tokenizer tests against")

target_compile_definitions(TokenizerTests PRIVATE
        WHISPER_TOKENIZER_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/Fixtures"
        WHISPER_TOKENIZER_JSON_PATH="${WHISPER_EACP_TOKENIZER_JSON}")
