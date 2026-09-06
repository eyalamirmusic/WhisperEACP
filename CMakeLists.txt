cmake_minimum_required(VERSION 3.31)

# Claimed before project(), not after: CMake's Darwin module reads the
# deployment target while it is probing the compiler, so a set() further down is
# silently ignored and every object ends up built against whatever SDK the build
# machine happens to have. 11.0 is eacp's own floor.
if (APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING
            "Minimum macOS version WhisperEACP targets")
endif ()

project(WhisperEACP VERSION 0.1.0 LANGUAGES C CXX)

if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR
            "WhisperEACP requires a 64-bit toolchain (x64 or ARM64), but the "
            "selected compiler targets 32-bit "
            "(CMAKE_SIZEOF_VOID_P=${CMAKE_SIZEOF_VOID_P}, "
            "arch='${CMAKE_SYSTEM_PROCESSOR}'). On Windows, configure from an "
            "x64 or ARM64 Developer prompt, or pass an explicit 64-bit "
            "target, e.g. -DCMAKE_CXX_FLAGS=--target=x86_64-pc-windows-msvc.")
endif ()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# eacp links its static libraries into runtime-loaded plugins and webview
# helpers, so every static lib in the tree — ours and the CPM-fetched ones — has
# to be position-independent or the shared-object link fails on Linux. Set here
# as well as there so it reaches the dependencies configured below.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (APPLE)
    enable_language(OBJCXX)

    # CMake already puts -lobjc on the link line of everything that pulls in
    # Objective-C. Xcode adds -fobjc-link-runtime on top for any target with its
    # own .mm sources, and ld warns about the duplicate.
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_LINK_OBJC_RUNTIME NO)
endif ()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

option(WHISPER_EACP_UNITY_BUILD
        "Compile WhisperEACP libraries as CMake unity builds" OFF)
option(WHISPER_EACP_CI_BUILD "Mirror the exact flag set used by CI builds" OFF)

if (WHISPER_EACP_CI_BUILD)
    set(WHISPER_EACP_UNITY_BUILD ON CACHE BOOL
            "Compile WhisperEACP libraries as CMake unity builds" FORCE)
    set(EACP_UNITY_BUILD ON CACHE BOOL
            "Compile eacp libraries as CMake unity builds" FORCE)
    set(MAKEASOUND_UNITY_BUILD ON CACHE BOOL
            "Enable unity (jumbo) build for the MakeASound library" FORCE)
    set(MIRO_UNITY_BUILD ON CACHE BOOL
            "Enable CMake unity build for the Miro library" FORCE)
endif ()

include(WhisperTargetSetup)

if (PROJECT_IS_TOP_LEVEL)
    whisper_default_setup()
endif ()

set(CMAKE_CXX_STANDARD 20)

# eacp first: it defines nothing this project calls, but it brings Miro and
# ea_data_structures along, which MakeASound then finds already populated
# instead of fetching a second copy at a different tag.
find_package(eacp REQUIRED)
find_package(MakeASound REQUIRED)

add_subdirectory(Lib)

option(WHISPER_EACP_ENABLE_APPS "Build the WhisperEACP apps"
        ${PROJECT_IS_TOP_LEVEL})

if (WHISPER_EACP_ENABLE_APPS)
    add_subdirectory(Apps)
endif ()

option(WHISPER_EACP_ENABLE_TESTS "Enable tests for WhisperEACP"
        ${PROJECT_IS_TOP_LEVEL})

if (WHISPER_EACP_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(Tests)
endif ()
