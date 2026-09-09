include(CPM)

# WebView off: it is the one eacp module whose build shells out to npm/Vite, and
# nothing here draws a UI. Graphics has to stay on — eacp gates the whole GPU
# stack behind EACP_BUILD_GRAPHICS, so eacp-gpu does not exist without it, and
# eacp-gpu is the entire reason this project exists.
#
# eacp's own tests and examples default to off already, since it only enables
# them when it is the top-level project and here it never is.
CPMAddPackage(
        NAME eacp
        GITHUB_REPOSITORY eyalamirmusic/eacp
        GIT_TAG develop
        OPTIONS
        "EACP_BUILD_WEBVIEW OFF")

# eacp's SIMD-group matrix — SimdMatrix, simdGroupIndex(), multiplyAccumulate()
# — is what Kernels/SimdTiledMatMul.h is written against, and it is on a branch
# until it lands on develop. So the tree asks the eacp it was handed whether it
# has one rather than assuming, and an eacp without it gets the register-tiled
# product for every role, which is what every non-Metal backend gets anyway.
#
# The declaration in the header is the test: there is no version to compare
# against, eacp tagging none.
file(READ "${eacp_SOURCE_DIR}/Lib/eacp/GPU/Codegen/ShaderBuilder.h"
        whisper_eacp_shader_builder)

string(FIND "${whisper_eacp_shader_builder}" "SimdMatrix simdMatrix("
        whisper_eacp_simd_matrix_at)

if (whisper_eacp_simd_matrix_at EQUAL -1)
    set(WHISPER_EACP_HAS_SIMD_MATRIX OFF)
    message(STATUS "eacp has no SIMD-group matrix: the products run the "
            "register-tiled kernel, and the benchmark measures that")
else ()
    set(WHISPER_EACP_HAS_SIMD_MATRIX ON)
endif ()

# eacp appends its own CMake/ to CMAKE_MODULE_PATH inside its directory scope,
# which a parent project never inherits. Re-append it here so eacp's helper
# modules stay reachable by name from this project's CMakeLists.
list(APPEND CMAKE_MODULE_PATH "${eacp_SOURCE_DIR}/CMake")

# eacp fills these in from eacp_default_setup(), which only runs when eacp is
# the top-level project. Here it is a subproject, so point the bundle templates
# at its own copies — without them every .app falls back to CMake's default
# Info.plist.
if (APPLE AND eacp_SOURCE_DIR)
    if (IOS)
        set(EACP_IOS_PLIST
                "${eacp_SOURCE_DIR}/CMake/iOSBundleInfo.plist.in"
                CACHE INTERNAL "")
    else ()
        set(EACP_MACOS_PLIST
                "${eacp_SOURCE_DIR}/CMake/macOSBundleInfo.plist.in"
                CACHE INTERNAL "")
    endif ()
endif ()

# develop rather than main: main trails it, and the compute layer this project
# is written against moves on develop first. The fetch is the default and the
# configuration CI and every ordinary build use — nothing here points at a
# checkout on the machine.
#
# A local checkout is available for the case it is actually for: changing eacp
# itself, alongside a change here that needs it. It is a deliberate override
# for that session, not a standing setup, and the build goes back to the fetch
# once the eacp change is pushed.
#
#   cmake -B build -DCPM_eacp_SOURCE=$HOME/Code/eacp
#
# Use $HOME, not ~ — CMake does not expand a tilde, and the shell will not
# expand one inside quotes, so the path silently resolves to nothing and the
# failure surfaces much later as a missing eacp-gpu target. A local tree also
# brings whatever is uncommitted in it into this build, which is its own
# hazard: an unrelated refactor in progress there breaks every target here.
