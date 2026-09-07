# whisper.cpp, fetched and added the one way it is safe to add it here. Two
# directories want it — Tests/Oracle as a CPU-only reference and Benchmark as
# a contestant built the way a user would build it — and the fetch has the
# same two hazards either way, so both call this rather than each carrying a
# copy of the workarounds.
#
# Not named WhisperCpp after anything in eacp/CMake or makeasound/CMake: this
# project's CMAKE_MODULE_PATH entry is appended before theirs, so a module here
# that shared a name with one of theirs would shadow it.
#
#   whisper_add_whisper_cpp([OPTIONS <option> ...])
#
# adds the `whisper` target (and ggml under it) at v1.9.3, static, with its
# tests, examples and server off. OPTIONS are forwarded to CPM as whisper.cpp's
# own switches, which is where a caller names the backends it wants: Tests/
# Oracle turns every one of them off, Benchmark leaves them at whisper.cpp's
# defaults. Sets whisper-cpp_SOURCE_DIR in the caller's scope.
#
#   whisper_fetch_ggml_tiny_en()
#
# fetches the GGML conversion of the same tiny.en weights model.safetensors
# holds, hash pinned, and sets WHISPER_EACP_GGML_MODEL to the file's path. It
# is a conversion, which is why it is worth saying out loud: a disagreement
# against whisper.cpp has that conversion as a candidate explanation, and
# CLAUDE.md says to rule it out first.

include(CPM)

function(whisper_add_whisper_cpp)
    cmake_parse_arguments(ARG "" "" "OPTIONS" ${ARGN})

    # whisper.cpp forces CMAKE_BUILD_TYPE to Release, in the cache, whenever it
    # finds it empty — and the cache is the one scope an add_subdirectory does
    # not contain, so a configure that named no build type would come back
    # Release for the whole tree. Put back below.
    set(buildTypeBefore "${CMAKE_BUILD_TYPE}")

    # The leak also runs the other way. whisper_default_setup() puts
    # _LIBCPP_REMOVE_TRANSITIVE_INCLUDES on the root directory, so every
    # directory added under it inherits it — and ggml's gguf.cpp reaches errno
    # through a transitive libc++ include and does not compile with it on. A
    # directory property is copied into a child when the child is added, so
    # clearing it here and putting it back afterwards leaves whisper.cpp's
    # subtree without it and the caller's own targets with it.
    get_directory_property(definitionsBefore COMPILE_DEFINITIONS)
    set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "")

    # BUILD_SHARED_LIBS is named because whisper.cpp defaults it ON everywhere
    # but Windows and Emscripten, and a shared ggml is a dylib the executable
    # would have to find at run time. It holds as a plain option because CPM
    # sets CMAKE_POLICY_DEFAULT_CMP0077 to NEW, so option() defers to the
    # variable rather than writing a cache entry over it — whisper.cpp's own
    # cmake_minimum_required of 3.5 would otherwise leave that policy unset.
    # None of these reaches the cache, which is what keeps them out of the rest
    # of the tree.
    #
    # WHISPER_BUILD_IS_DEV is on by default and only appends "-dev" to what
    # whisper_version() answers. This is the tagged release, so it says so.
    CPMAddPackage(
            NAME whisper-cpp
            GITHUB_REPOSITORY ggerganov/whisper.cpp
            GIT_TAG v1.9.3
            SYSTEM YES
            EXCLUDE_FROM_ALL YES
            OPTIONS
            "BUILD_SHARED_LIBS OFF"
            "WHISPER_BUILD_IS_DEV OFF"
            "WHISPER_BUILD_TESTS OFF"
            "WHISPER_BUILD_EXAMPLES OFF"
            "WHISPER_BUILD_SERVER OFF"
            ${ARG_OPTIONS})

    set_directory_properties(PROPERTIES
            COMPILE_DEFINITIONS "${definitionsBefore}")

    if (NOT CMAKE_BUILD_TYPE STREQUAL buildTypeBefore)
        set(CMAKE_BUILD_TYPE "${buildTypeBefore}" CACHE STRING
                "Build type" FORCE)
    endif ()

    set(whisper-cpp_SOURCE_DIR "${whisper-cpp_SOURCE_DIR}" PARENT_SCOPE)
endfunction()

# The GGML model, fetched the way Lib/WhisperEACP/Model fetches the
# HuggingFace files: one CPM package for the file, DOWNLOAD_NO_EXTRACT because
# a .bin is not an archive, DOWNLOAD_ONLY because there is no CMakeLists to
# add, and the hash pinned so a swapped file fails the configure rather than
# the comparison.
function(whisper_fetch_ggml_tiny_en)
    set(modelDirectory "${CMAKE_BINARY_DIR}/whisper-cpp-tiny.en")
    set(modelFile "ggml-tiny.en.bin")

    CPMAddPackage(
            NAME whisper-cpp-tiny-en-ggml
            URL https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${modelFile}
            URL_HASH SHA256=921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f
            DOWNLOAD_NO_EXTRACT YES
            DOWNLOAD_ONLY YES)

    file(MAKE_DIRECTORY "${modelDirectory}")

    file(CREATE_LINK
            "${whisper-cpp-tiny-en-ggml_SOURCE_DIR}/${modelFile}"
            "${modelDirectory}/${modelFile}"
            COPY_ON_ERROR SYMBOLIC)

    set(WHISPER_EACP_GGML_MODEL "${modelDirectory}/${modelFile}"
            CACHE INTERNAL "The fetched ggml-tiny.en.bin")
endfunction()
