add_library(whisper-model STATIC)

# Miro rides along on eacp-core, which whisper-core links through eacp-gpu, so
# <Miro/Json.h> needs no dependency of its own here.
target_link_libraries(whisper-model PUBLIC whisper-core)

target_sources(whisper-model PRIVATE
        ModelConfig.cpp
        ModelIO.cpp
        PreprocessorConfig.cpp
        SafeTensors.cpp
        TensorLoader.cpp
        TensorType.cpp)

whisper_set_default_target_setting(whisper-model)
whisper_enable_unity_build(whisper-model)
whisper_add_ide_sources(whisper-model)

# The model is a download, never a commit — 151 MB of weights has no business
# in git. Off by default so an ordinary build stays cheap; on, CPM fetches the
# three files this module reads and the directory they are assembled into
# reaches Tests/Model as a compile definition.
option(WHISPER_EACP_FETCH_MODEL
        "Download the HuggingFace openai/whisper-tiny.en model files" OFF)

# Set even when the fetch is off, so Tests/Model always compiles against a real
# path and the tests that need a file skip on its absence rather than on an
# #ifdef. Cached because a plain set() would not survive out of this directory.
set(WHISPER_EACP_MODEL_DIR "${CMAKE_BINARY_DIR}/whisper-tiny.en"
        CACHE INTERNAL "Where the fetched whisper-tiny.en files are assembled")

# Each file separately, because CPM keys a package on its name and a HF repo is
# not an archive: DOWNLOAD_NO_EXTRACT stops FetchContent trying to unpack the
# download, and DOWNLOAD_ONLY says there is no CMakeLists to add. The hashes are
# the ones HuggingFace serves for the current revision, so a corrupted or
# swapped file fails the configure rather than the decode.
set(WHISPER_TINY_EN_URL
        "https://huggingface.co/openai/whisper-tiny.en/resolve/main")

function(whisper_fetch_model_file package_name file_name file_hash)
    CPMAddPackage(
            NAME ${package_name}
            URL ${WHISPER_TINY_EN_URL}/${file_name}
            URL_HASH SHA256=${file_hash}
            DOWNLOAD_NO_EXTRACT YES
            DOWNLOAD_ONLY YES)

    # CPM leaves the three downloads in three separate _deps directories, so
    # link them into one so a caller names a model directory rather than three
    # unrelated paths. COPY_ON_ERROR covers a Windows account without the
    # privilege to make a symlink.
    file(CREATE_LINK
            "${${package_name}_SOURCE_DIR}/${file_name}"
            "${WHISPER_EACP_MODEL_DIR}/${file_name}"
            COPY_ON_ERROR SYMBOLIC)
endfunction()

if (WHISPER_EACP_FETCH_MODEL)
    file(MAKE_DIRECTORY "${WHISPER_EACP_MODEL_DIR}")

    whisper_fetch_model_file(whisper-tiny-en-config config.json
            b5366317df280f79e6cc366f259f379674f81a8721d458acf2e00680ccb10b1d)

    whisper_fetch_model_file(whisper-tiny-en-preprocessor preprocessor_config.json
            9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666)

    whisper_fetch_model_file(whisper-tiny-en-weights model.safetensors
            db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad)

    # The vocabulary. Tokenizer/ reads it, and Tests/Oracle compares what it
    # reads against whisper.cpp's own copy of the same vocabulary, so it
    # belongs in the same directory as the three files above rather than behind
    # a second switch.
    whisper_fetch_model_file(whisper-tiny-en-tokenizer tokenizer.json
            5eb60cec1e77aeeb6869a2bb5a8e01a84c3fe5d072d75369343021fe6f5310d0)
endif ()

# The sample the end-to-end test transcribes, fetched the same way and behind
# the same switch — the model and a recording to run it on are one decision, and
# a test that needed the whisper.cpp oracle just to find a WAV would make the
# cheapest tier of the suite depend on the most expensive one.
#
# Its own directory rather than the model's, and the reason is what
# WHISPER_MODEL_DIR is for: that variable points at a HuggingFace repo, whether
# fetched here or a checkout somebody already has, and jfk.wav is not in one. A
# sample dropped in beside the four files would be missing from every such
# checkout and present only in ours.
set(WHISPER_EACP_SAMPLE_DIR "${CMAKE_BINARY_DIR}/whisper-samples"
        CACHE INTERNAL "Where the fetched audio samples are assembled")

function(whisper_fetch_sample_file package_name url file_name file_hash)
    CPMAddPackage(
            NAME ${package_name}
            URL ${url}
            URL_HASH SHA256=${file_hash}
            DOWNLOAD_NO_EXTRACT YES
            DOWNLOAD_ONLY YES)

    file(CREATE_LINK
            "${${package_name}_SOURCE_DIR}/${file_name}"
            "${WHISPER_EACP_SAMPLE_DIR}/${file_name}"
            COPY_ON_ERROR SYMBOLIC)
endfunction()

if (WHISPER_EACP_FETCH_MODEL)
    file(MAKE_DIRECTORY "${WHISPER_EACP_SAMPLE_DIR}")

    # 11 seconds of Kennedy's inaugural at 16 kHz mono PCM16, from whisper.cpp's
    # own samples directory at the tag Tests/Oracle builds. Pinned to a tag
    # rather than to a branch so the bytes the hash covers cannot move.
    whisper_fetch_sample_file(whisper-sample-jfk
            https://github.com/ggerganov/whisper.cpp/raw/v1.9.3/samples/jfk.wav
            jfk.wav
            59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e)
endif ()
