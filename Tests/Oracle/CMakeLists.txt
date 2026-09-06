# whisper.cpp as an in-process oracle — the second tier of the validation order
# in CLAUDE.md. Off by default: it is a second inference stack to fetch and
# build, and a 75 MB model beside it, for assertions the scalar references in
# the rest of the suite cannot make on their own.
option(WHISPER_EACP_ENABLE_WHISPER_CPP
        "Fetch whisper.cpp and build the tests that compare against it" OFF)

if (NOT WHISPER_EACP_ENABLE_WHISPER_CPP)
    return()
endif ()

# whisper.cpp forces CMAKE_BUILD_TYPE to Release, in the cache, whenever it
# finds it empty — and the cache is the one scope an add_subdirectory does not
# contain, so a configure that named no build type would come back Release for
# this whole tree. Put back below.
set(whisperCppBuildTypeBefore "${CMAKE_BUILD_TYPE}")

# The leak also runs the other way. whisper_default_setup() puts
# _LIBCPP_REMOVE_TRANSITIVE_INCLUDES on the root directory, so every directory
# added under it inherits it — and ggml's gguf.cpp reaches errno through a
# transitive libc++ include and does not compile with it on. A directory
# property is copied into a child when the child is added, so clearing it here
# and putting it back afterwards leaves whisper.cpp's subtree without it and
# this directory's own targets with it.
get_directory_property(whisperCppDefinitionsBefore COMPILE_DEFINITIONS)
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "")

# CPU-only, every backend named off rather than left to the defaults, which on
# a Mac turn Metal, Accelerate and BLAS on. GGML_BLAS is its own switch and is
# not implied by GGML_ACCELERATE OFF: without it the configure still finds
# Accelerate and links BLAS into the reference, and an oracle that runs a
# different matmul from the one it documents is not one.
#
# BUILD_SHARED_LIBS is named for a reason of its own: whisper.cpp defaults it
# ON everywhere but Windows and Emscripten, and a shared ggml is a dylib the
# test executable would have to find at run time. These hold as plain options
# because CPM sets CMAKE_POLICY_DEFAULT_CMP0077 to NEW, so option() defers to
# the variable rather than writing a cache entry over it — whisper.cpp's own
# cmake_minimum_required of 3.5 would otherwise leave that policy unset. None
# of them reaches the cache, which is what keeps them out of the rest of the
# tree.
CPMAddPackage(
        NAME whisper-cpp
        GITHUB_REPOSITORY ggerganov/whisper.cpp
        GIT_TAG v1.9.3
        SYSTEM YES
        EXCLUDE_FROM_ALL YES
        OPTIONS
        "BUILD_SHARED_LIBS OFF"
        "WHISPER_BUILD_TESTS OFF"
        "WHISPER_BUILD_EXAMPLES OFF"
        "WHISPER_BUILD_SERVER OFF"
        "GGML_METAL OFF"
        "GGML_OPENMP OFF"
        "GGML_ACCELERATE OFF"
        "GGML_BLAS OFF")

set_directory_properties(PROPERTIES
        COMPILE_DEFINITIONS "${whisperCppDefinitionsBefore}")

if (NOT CMAKE_BUILD_TYPE STREQUAL whisperCppBuildTypeBefore)
    set(CMAKE_BUILD_TYPE "${whisperCppBuildTypeBefore}" CACHE STRING
            "Build type" FORCE)
endif ()

# The GGML model, fetched the way Lib/WhisperEACP/Model fetches the
# HuggingFace files: one CPM package per file, DOWNLOAD_NO_EXTRACT because a
# .bin is not an archive, DOWNLOAD_ONLY because there is no CMakeLists to add,
# and the hash pinned so a swapped file fails the configure rather than the
# comparison.
#
# It is a conversion of the same weights model.safetensors holds, which is why
# it is worth saying out loud: a disagreement against this reference has that
# conversion as a candidate explanation, and CLAUDE.md says to rule it out
# first.
set(WHISPER_EACP_GGML_MODEL_DIR "${CMAKE_BINARY_DIR}/whisper-cpp-tiny.en"
        CACHE INTERNAL "Where the fetched GGML model is assembled")

set(WHISPER_CPP_MODEL_URL
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main")

set(WHISPER_CPP_TINY_EN_SHA256
        921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f)

CPMAddPackage(
        NAME whisper-cpp-tiny-en-ggml
        URL ${WHISPER_CPP_MODEL_URL}/ggml-tiny.en.bin
        URL_HASH SHA256=${WHISPER_CPP_TINY_EN_SHA256}
        DOWNLOAD_NO_EXTRACT YES
        DOWNLOAD_ONLY YES)

file(MAKE_DIRECTORY "${WHISPER_EACP_GGML_MODEL_DIR}")

file(CREATE_LINK
        "${whisper-cpp-tiny-en-ggml_SOURCE_DIR}/ggml-tiny.en.bin"
        "${WHISPER_EACP_GGML_MODEL_DIR}/ggml-tiny.en.bin"
        COPY_ON_ERROR SYMBOLIC)

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
        WHISPER_EACP_GGML_MODEL="${WHISPER_EACP_GGML_MODEL_DIR}/ggml-tiny.en.bin"
        WHISPER_EACP_JFK_WAV="${whisper-cpp_SOURCE_DIR}/samples/jfk.wav")
