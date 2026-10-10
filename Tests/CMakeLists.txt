set(CMAKE_FOLDER Tests)

find_package(NanoTest REQUIRED)

# The entry point a GPU-touching test executable links instead of NanoTest's
# own — see Support/GpuTestMain.cpp. Named here so each such module lists it in
# SOURCES beside NO_MAIN rather than keeping a copy of its own.
set(WHISPER_GPU_TEST_MAIN "${CMAKE_CURRENT_LIST_DIR}/Support/GpuTestMain.cpp")

# The same entry point with the model fetch in front of it, for the modules that
# read the real tiny.en weights. One or the other, never both: it is that file
# plus a call, and the call has to be outside the event loop the other one opens.
set(WHISPER_MODEL_TEST_MAIN "${CMAKE_CURRENT_LIST_DIR}/Support/ModelTestMain.cpp")

# nano_add_executable with eacp's AppInfo stamped in, as every executable of
# the tree has it (whisper_embed_app_info). The embedding is a custom command,
# which reaches only targets of the directory that adds it, so it happens where
# each test executable is defined rather than once from here.
function(whisper_add_test_executable target)
    nano_add_executable(${target} ${ARGN})
    whisper_embed_app_info(${target})
endfunction()

add_subdirectory(Core)
add_subdirectory(Audio)
add_subdirectory(Decoder)
add_subdirectory(Encoder)
add_subdirectory(GPU)
add_subdirectory(Kernels)
add_subdirectory(Mel)
add_subdirectory(Model)
add_subdirectory(Net)
add_subdirectory(Oracle)
add_subdirectory(Tokenizer)
add_subdirectory(Whisper)
