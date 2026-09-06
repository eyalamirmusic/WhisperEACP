// The entry point every GPU-touching test executable links instead of
// NanoTest's own: anything that reaches the device has to run inside eacp's
// event loop, which owns the run loop and the autorelease pool the Metal
// backend is written against.
//
// One copy, named by WHISPER_GPU_TEST_MAIN, rather than one per test module.

#include <eacp/Core/App/App.h>

#include <NanoTest/NanoTest.h>

namespace
{
int argCount = 0;
char** argValues = nullptr;
int exitCode = 0;

void runTests()
{
    exitCode = nano::run(argCount, argValues);
}
} // namespace

int main(int argc, char* argv[])
{
    argCount = argc;
    argValues = argv;

    eacp::Apps::run(runTests);
    return exitCode;
}
