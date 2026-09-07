# A post-build copy of files to where a binary can find them at runtime. Not
# named Resources.cmake, for the reason WhisperTargetSetup.cmake gives: this
# project's CMAKE_MODULE_PATH entry is appended before eacp's and MakeASound's,
# so a module here that shares a name with one of theirs shadows it.
#
#   whisper_copy_resources(<target> FILES <file>... [DESTINATION <subdirectory>])
#
# The destination is the platform's idea of "beside the executable": the
# Contents/Resources directory of a MACOSX_BUNDLE target, and the directory the
# executable itself is written to otherwise — which is what a Windows build,
# and a macOS executable that is not a bundle, both are. DESTINATION names a
# subdirectory under that, so a set of files keeps its own directory rather
# than landing loose next to the binary.
#
# The copy runs after every link of the target, and copy_if_different keeps a
# rebuild that changed nothing from rewriting 151 MB. It is a copy of the bytes,
# not a link, whatever the source is — the fetched model directory is symlinks
# into CPM's download directories, and the binary should not depend on those
# staying where they are.
#
# The runtime half is WSP::resourcesDirectory(), which resolves the same two
# destinations from inside the running binary.
function(whisper_copy_resources target)
    cmake_parse_arguments(ARG "" "DESTINATION" "FILES" ${ARGN})

    if (NOT ARG_FILES)
        message(FATAL_ERROR
                "whisper_copy_resources(${target}) was given no FILES")
    endif ()

    if (APPLE)
        set(destination
                "$<IF:$<BOOL:$<TARGET_PROPERTY:${target},MACOSX_BUNDLE>>,\
$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources,$<TARGET_FILE_DIR:${target}>>")
    else ()
        set(destination "$<TARGET_FILE_DIR:${target}>")
    endif ()

    if (ARG_DESTINATION)
        string(APPEND destination "/${ARG_DESTINATION}")
    endif ()

    add_custom_command(TARGET ${target} POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${destination}"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            ${ARG_FILES} "${destination}"
            COMMENT "Copying resources beside ${target}"
            VERBATIM)
endfunction()
