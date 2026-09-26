# NO_MAIN, and the only module here that takes the model entry point without
# needing a device: the tokenizer tests want tokenizer.json, which is one of the
# four files the fetch fetches, so they take the main that fetches them.
# whisper-model comes along for ModelFetch rather than for anything the tokenizer
# itself reads.
nano_add_executable(TokenizerTests
        NO_MAIN
        SOURCES
        TokenizerTests.cpp
        ${WHISPER_MODEL_TEST_MAIN}
        TARGETS whisper-tokenizer whisper-model)

# A real tokenizer.json is a few megabytes, so it is a download and never a
# commit: the tests that need one skip when it is absent, the same way the GPU
# tests skip without a device. Point them at a copy somebody already has with
# either -DWHISPER_EACP_TOKENIZER_JSON=<path> at configure time or the
# WHISPER_TOKENIZER_JSON environment variable at run time.
#
# ModelFetch::directory() is the fallback both of those have, because
# tokenizer.json is one of the four files the runtime fetch gets: a run on a
# machine with a network has it beside the other three.
#
# The committed fixture beside it is reached by absolute path rather than by
# guessing a working directory.
set(WHISPER_EACP_TOKENIZER_JSON "" CACHE FILEPATH
        "A downloaded HuggingFace tokenizer.json to run the tokenizer tests against")

target_compile_definitions(TokenizerTests PRIVATE
        WHISPER_TOKENIZER_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/Fixtures"
        WHISPER_TOKENIZER_JSON_PATH="${WHISPER_EACP_TOKENIZER_JSON}")
