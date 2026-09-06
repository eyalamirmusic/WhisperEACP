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
        GIT_TAG main
        OPTIONS
        "EACP_BUILD_WEBVIEW OFF")

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

# To develop against a local eacp checkout instead of the GitHub fetch — which
# is the normal setup here, since this project drives changes to eacp's compute
# layer — pass CPM's per-package source override at configure time:
#
#   cmake -B build -DCPM_eacp_SOURCE=$HOME/Code/eacp
#
# Use $HOME, not ~ — CMake does not expand a tilde, and the shell will not
# expand one inside quotes, so the path silently resolves to nothing and the
# failure surfaces much later as a missing eacp-gpu target.
