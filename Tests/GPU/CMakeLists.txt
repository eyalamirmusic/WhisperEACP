set(gpu_test_sources
        ComputeSmokeTests.cpp
        TestMain.cpp)

# NO_MAIN, because these need an entry point of their own: anything that touches
# the GPU has to run inside eacp's event loop, which owns the run loop and the
# autorelease pool the Metal backend is written against. TestMain.cpp is that
# entry point.
nano_add_executable(GPUTests
        NO_MAIN
        SOURCES ${gpu_test_sources}
        TARGETS whisper-core)
