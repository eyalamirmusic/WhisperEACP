# Target defaults, mirroring eacp's TargetSetup.cmake.
#
# Deliberately NOT named TargetSetup.cmake. CMAKE_MODULE_PATH is inherited by
# every subdirectory, and this project's entry is appended before eacp's — so a
# module of that name here shadows eacp's, and eacp's `include(TargetSetup)`
# silently picks up this file instead. Its `set_default_target_setting` then
# never gets defined, and eacp fails to configure on the first target that calls
# it. MakeASound's modules are on the same path and have the same problem.
#
# The functions carry a `whisper_` prefix for the same class of reason: CMake
# functions are global once defined, so a name declared both here and inside a
# fetched dependency resolves to whichever directory was processed last, which
# is not something a call site can see. Check `eacp/CMake/` and
# `makeasound/CMake/` before adding a module here.

function(whisper_set_default_warnings_level target)
    if (MSVC)
        target_compile_options(${target} PRIVATE /W4)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif ()
endfunction()

# The bundle plists come from eacp rather than from a copy here: an app of ours
# is an eacp app, and CMake/Findeacp.cmake points these at eacp's own templates.
function(whisper_set_default_target_setting target)
    whisper_set_default_warnings_level(${target})

    set_target_properties(${target} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

    if (IOS)
        set_target_properties(${target} PROPERTIES
                MACOSX_BUNDLE_INFO_PLIST "${EACP_IOS_PLIST}")
    elseif (APPLE)
        set_target_properties(${target} PROPERTIES
                MACOSX_BUNDLE_INFO_PLIST "${EACP_MACOS_PLIST}")
    endif ()
endfunction()

function(whisper_enable_unity_build target)
    if (WHISPER_EACP_UNITY_BUILD)
        set_target_properties(${target} PROPERTIES UNITY_BUILD ON)
    endif ()
endfunction()

# Headers are not compiled on their own, so nothing lists them — which also
# means an IDE never shows them next to the sources they belong to. Attach them
# as header-only sources so the module reads as one unit in the project tree.
function(whisper_add_ide_sources target)
    file(GLOB_RECURSE all_headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

    if (all_headers)
        target_sources(${target} PRIVATE ${all_headers})
        set_source_files_properties(${all_headers} PROPERTIES
                HEADER_FILE_ONLY TRUE)
    endif ()
endfunction()

# A macro rather than a function on purpose: everything below sets a variable
# the caller has to see afterwards, and a function's own scope would swallow
# every one of them.
macro(whisper_default_setup)
    set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
    add_compile_definitions(_LIBCPP_REMOVE_TRANSITIVE_INCLUDES)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_COMPILE_OPTIONS_IPO "-flto=full")
    endif ()
endmacro()
