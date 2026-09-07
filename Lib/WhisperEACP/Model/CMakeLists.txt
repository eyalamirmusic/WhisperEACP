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
# in git. This switch is off by default so a build that only wants the fetched
# files on disk asks for them; on, CPM fetches the four files this module reads
# and the directory they are assembled into reaches Tests/Model as a compile
# definition.
option(WHISPER_EACP_FETCH_MODEL
        "Download the HuggingFace openai/whisper-tiny.en model files" OFF)

# Embedding needs the same four files, so it turns the fetch on below rather
# than duplicating it — which is why a default configure downloads them even
# though the switch above is off.
option(WHISPER_EACP_EMBED_MODEL
        "Embed the whisper-tiny.en model files into the binaries that link \
whisper-embedded-model" ON)

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

    # CPM leaves the four downloads in four separate _deps directories, so
    # link them into one so a caller names a model directory rather than four
    # unrelated paths. COPY_ON_ERROR covers a Windows account without the
    # privilege to make a symlink.
    file(CREATE_LINK
            "${${package_name}_SOURCE_DIR}/${file_name}"
            "${WHISPER_EACP_MODEL_DIR}/${file_name}"
            COPY_ON_ERROR SYMBOLIC)
endfunction()

if (WHISPER_EACP_FETCH_MODEL OR WHISPER_EACP_EMBED_MODEL)
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

# The same four files again, as bytes inside whatever links this — which is why
# it lives here rather than beside Whisper::loadEmbedded that reads them: this
# target is about the fetch above, and nothing but the file list connects the
# two.
#
# The name is load-bearing twice over. ResEmbed keys a resource by its basename,
# so the keys are exactly the HF file names loadEmbedded() looks up; and
# Whisper.cpp's error message tells the caller to link `whisper-embedded-model`.
#
# INTERFACE when the option is off rather than no target at all, so a consumer
# links it unconditionally and Whisper::hasEmbeddedModel() is what answers —
# the same shape as a test skipping on a missing file rather than on an #ifdef.
if (WHISPER_EACP_EMBED_MODEL)
    add_library(whisper-embedded-model STATIC)

    find_package(ResEmbed REQUIRED)
    res_embed_add(whisper-embedded-model
            FILES
            "${WHISPER_EACP_MODEL_DIR}/config.json"
            "${WHISPER_EACP_MODEL_DIR}/preprocessor_config.json"
            "${WHISPER_EACP_MODEL_DIR}/model.safetensors"
            "${WHISPER_EACP_MODEL_DIR}/tokenizer.json"
            NAMESPACE WhisperModel
            CATEGORY WhisperModel)

    # No warnings level, no LTO, no unity batch: this target is data. ResEmbed
    # emits the weights as a 657 MB decimal brace initializer, and nothing an
    # optimiser or a jumbo TU does to that is worth the compile.
else ()
    add_library(whisper-embedded-model INTERFACE)
endif ()
