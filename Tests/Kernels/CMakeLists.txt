set(kernel_test_sources
        AddTests.cpp
        ArgmaxTests.cpp
        AttentionTests.cpp
        Conv1dTests.cpp
        EmbedTests.cpp
        GeluTests.cpp
        LayerNormTests.cpp
        LinearTests.cpp
        MatMulTests.cpp
        SoftmaxTests.cpp)

# NO_MAIN: these reach the GPU, so they run inside eacp's event loop through
# the shared entry point in Tests/Support.
nano_add_executable(KernelTests
        NO_MAIN
        SOURCES
        ${kernel_test_sources}
        ${WHISPER_GPU_TEST_MAIN}
        TARGETS whisper-kernels)
