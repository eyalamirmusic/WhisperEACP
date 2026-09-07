# Our runtime against whisper.cpp, timed in one process: Main.cpp says what is
# measured, README.md how to read it. Behind WHISPER_EACP_ENABLE_BENCHMARK in
# the root CMakeLists, which is also where the whisper.cpp this links is
# fetched — at whisper.cpp's own defaults for the machine, Metal and Accelerate
# on a Mac, rather than the every-backend-off build the oracle alone would get.
#
# A benchmark of a Debug build measures the Debug build, and the tree's usual
# configure is Debug. The numbers come from a Release tree:
#
#   cmake -G Ninja -B build-release -DCMAKE_BUILD_TYPE=Release \
#         -DWHISPER_EACP_ENABLE_BENCHMARK=ON
#   cmake --build build-release --target Benchmark
#   ./build-release/Benchmark/Benchmark
#
# The binary prints the configuration it was built in above its numbers, so a
# Debug run says so rather than passing for a measurement.
set(CMAKE_FOLDER Benchmark)

add_executable(Benchmark Main.cpp)
target_link_libraries(Benchmark PRIVATE WhisperEACP whisper)

# The HuggingFace model beside the binary, for Whisper::loadBundled(). The
# GGML one and the sample are compile-time paths into the build and source
# trees: this is a tool of the build, not something that ships.
whisper_bundle_model(Benchmark)

target_compile_definitions(Benchmark PRIVATE
        WHISPER_EACP_GGML_MODEL="${WHISPER_EACP_GGML_MODEL}"
        WHISPER_EACP_SAMPLE_DIR="${WHISPER_EACP_SAMPLE_DIR}"
        WHISPER_EACP_BUILD_TYPE="$<CONFIG>")

whisper_set_default_target_setting(Benchmark)
