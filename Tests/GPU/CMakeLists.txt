# NO_MAIN, because these need an entry point of their own: anything that
# touches the GPU has to run inside eacp's event loop, which owns the run loop
# and the autorelease pool the Metal backend is written against.
# Tests/Support/GpuTestMain.cpp is that entry point, shared by every
# GPU-touching test module.
nano_add_executable(GPUTests
        NO_MAIN
        SOURCES
        ComputeSmokeTests.cpp
        ${WHISPER_GPU_TEST_MAIN}
        TARGETS whisper-core)
