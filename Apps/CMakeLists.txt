set(CMAKE_FOLDER Apps)

add_subdirectory(Console)

# eacp builds its widget tier on macOS and Windows, and there is no Linux target
# here for it to be missing from, so the gate is simply whether the tier is in
# the tree at all.
if (TARGET eacp-ui)
    add_subdirectory(Demo)
endif ()
