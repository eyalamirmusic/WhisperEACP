set(CMAKE_FOLDER Tests)

find_package(NanoTest REQUIRED)

# The entry point a GPU-touching test executable links instead of NanoTest's
# own — see Support/GpuTestMain.cpp. Named here so each such module lists it in
# SOURCES beside NO_MAIN rather than keeping a copy of its own.
set(WHISPER_GPU_TEST_MAIN "${CMAKE_CURRENT_LIST_DIR}/Support/GpuTestMain.cpp")

add_subdirectory(Core)
add_subdirectory(Audio)
add_subdirectory(GPU)
add_subdirectory(Kernels)
add_subdirectory(Mel)
add_subdirectory(Model)
add_subdirectory(Tokenizer)
