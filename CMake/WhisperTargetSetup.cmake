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

# Every executable here carries eacp's AppInfo, named under the company the
# root CMakeLists sets in EACP_COMPANY_NAME, so the folders eacp derives from it
# are one per tree rather than one per binary: ML::defaultCacheDirectory() is
# <Caches>/<company>/CoreML, and the apps and the benchmark compile a Core ML
# model into it once between them. eacp_set_gui_subsystem embeds it already,
# which the AppInfo.json among the target's sources tells apart.
function(whisper_embed_app_info target)
    get_target_property(type ${target} TYPE)
    if (NOT type STREQUAL "EXECUTABLE")
        return()
    endif ()

    get_target_property(sources ${target} SOURCES)
    list(FILTER sources INCLUDE REGEX "/AppInfo\\.json$")
    if (NOT sources)
        eacp_embed_app_info(${target})
    endif ()
endfunction()

# The bundle plists come from eacp rather than from a copy here: an app of ours
# is an eacp app, and eacp's TargetSetup publishes EACP_MACOS_PLIST and
# EACP_IOS_PLIST when it is included, top-level project or not.
function(whisper_set_default_target_setting target)
    whisper_set_default_warnings_level(${target})
    whisper_embed_app_info(${target})

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

# Signing for an iPhone, under the Xcode generator only, the way MakeASound's
# plugin setup does it. eacp names a development team only when it is the
# top-level project, so a tree that fetches it signs nothing unless it says so:
# with WHISPER_EACP_IOS_TEAM set, Xcode signs automatically under that team;
# without one, device builds are left unsigned rather than failing on a missing
# identity.
function(whisper_sign_ios_app target)
    if (NOT IOS OR NOT CMAKE_GENERATOR STREQUAL "Xcode")
        return()
    endif ()

    set_target_properties(${target} PROPERTIES
            XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2")

    if (WHISPER_EACP_IOS_TEAM)
        set_target_properties(${target} PROPERTIES
                XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "${WHISPER_EACP_IOS_TEAM}"
                XCODE_ATTRIBUTE_CODE_SIGN_STYLE "Automatic"
                XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "Apple Development")
    else ()
        set_target_properties(${target} PROPERTIES
                XCODE_ATTRIBUTE_CODE_SIGN_STYLE "Manual"
                XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "-"
                "XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED[sdk=iphoneos*]" "NO")
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
